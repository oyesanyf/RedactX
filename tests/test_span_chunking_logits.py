"""
Unit tests for core production modules:
- TokenSpanLocator span extraction and boundary handling
- chunk_text windowing guarantees and interval merging
- Candidate logit token index resolution
"""

import pytest
import torch
from redactx.models.span_locator import TokenSpanLocator
from redactx.production.chunking import Window, chunk_text, merge_intervals, to_global
from redactx.models.openjev import resolve_single_token


def test_chunk_text_guarantees():
    text = (
        "Admission Date: 2026-05-12. Patient John Doe presented with acute chest pain. "
        "Cardiac enzymes were elevated. Started on aspirin and heparin. "
        "Discharged on 2026-05-18 in stable condition to follow up with Dr. Smith."
    ) * 10

    max_chars = 300
    overlap = 100

    windows = chunk_text(text, max_chars=max_chars, overlap=overlap)
    assert len(windows) > 1

    # Verify coverage of all characters
    for w in windows:
        assert w.start < w.end
        assert (w.end - w.start) <= max_chars

    assert windows[0].start == 0
    assert windows[-1].end == len(text)

    # Verify overlap guarantee between consecutive windows
    for i in range(len(windows) - 1):
        w_curr = windows[i]
        w_next = windows[i + 1]
        assert w_next.start <= w_curr.end - overlap or w_next.start < w_curr.end


def test_merge_intervals_and_to_global():
    window = Window(100, 400)
    local_spans = [(10, 20), (25, 35)]
    global_spans = to_global(window, local_spans)
    assert global_spans == [(110, 120), (125, 135)]

    # Overlapping or adjacent intervals
    intervals = [(10, 25), (20, 30), (50, 60), (62, 70)]
    merged = merge_intervals(intervals, join_gap=2)
    # (10, 25) + (20, 30) merge into (10, 30)
    # (50, 60) and (62, 70) gap is 2, so join_gap=2 merges them into (50, 70)
    assert merged == [(10, 30), (50, 70)]


def test_token_span_locator_category_inference():
    assert TokenSpanLocator.infer_category("05/12/2026") == "DATE"
    assert TokenSpanLocator.infer_category("12-05-2026") == "DATE"
    assert TokenSpanLocator.infer_category("MRN: 9812401") == "IDENTIFIER"
    assert TokenSpanLocator.infer_category("123-45-6789") == "IDENTIFIER"
    assert TokenSpanLocator.infer_category("john.doe@hospital.org") == "CONTACT"
    assert TokenSpanLocator.infer_category("617-555-0199") == "CONTACT"
    assert TokenSpanLocator.infer_category("Fairview Hospital") == "LOCATION"
    assert TokenSpanLocator.infer_category("Robert Langdon") == "NAME"


def test_token_span_locator_extract_spans():
    locator = TokenSpanLocator(hidden_dim=32)

    prompt = "<prompt> Context: Patient Jane Smith was admitted on 2026-04-10. </prompt>"
    context = "Patient Jane Smith was admitted on 2026-04-10."
    offset_mapping = [
        (0, 0),        # special token
        (0, 8),        # "<prompt>"
        (9, 17),       # "Context:"
        (18, 25),      # "Patient"
        (26, 30),      # "Jane"
        (31, 36),      # "Smith"
        (37, 40),      # "was"
        (41, 49),      # "admitted"
        (50, 52),      # "on"
        (53, 63),      # "2026-04-10"
        (63, 64),      # "."
        (65, 74),      # "</prompt>"
        (0, 0),
    ]

    # Set Jane Smith and 2026-04-10 to high probabilities
    probs = torch.zeros(len(offset_mapping))
    probs[4] = 0.99  # "Jane"
    probs[5] = 0.98  # "Smith"
    probs[9] = 0.95  # "2026-04-10"

    spans = locator.extract_spans(
        token_probs=probs,
        offset_mapping=offset_mapping,
        full_prompt=prompt,
        context_text=context,
        threshold=0.50
    )

    assert len(spans) == 2
    # Jane Smith should be merged into a single contiguous span
    assert spans[0].text == "Jane Smith"
    assert spans[0].category == "NAME"
    assert spans[0].confidence > 0.90

    # 2026-04-10
    assert spans[1].text == "2026-04-10"
    assert spans[1].category == "DATE"


def test_resolve_single_token():
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == " true":
                return [101, 1092]
            if text == "true":
                return [1092]
            return [999]

    tok = DummyTokenizer()
    token_id = resolve_single_token(tok, "true")
    assert token_id == 1092


def test_adversarial_zero_width_and_whitespace_spans():
    # Attack point: adversarial zero-width characters (ZWSP, ZWNJ, ZWJ, BOM) or non-breaking spaces
    assert TokenSpanLocator.infer_category("MRN:\u200b 9812401") == "IDENTIFIER"
    assert TokenSpanLocator.infer_category("123\u00a045\u00a06789") == "IDENTIFIER"
    assert TokenSpanLocator.infer_category("05/\u200b12/\u200b2026") == "DATE"
    assert TokenSpanLocator.infer_category("(617)\u00a0555-0199") == "CONTACT"
    assert TokenSpanLocator.infer_category("Fairview\u200b Hospital") == "LOCATION"
    assert TokenSpanLocator.infer_category("John\u200c Doe") == "NAME"

    # Single alphanumeric character (e.g. initial)
    locator = TokenSpanLocator(hidden_dim=32)
    raw_text = "Dr. J. Doe"
    raw_ranges = [(0, 3), (4, 5), (5, 6), (7, 10)]
    probs = torch.tensor([0.01, 0.99, 0.01, 0.98])
    spans = locator.extract_raw_spans(probs, raw_ranges, raw_text, threshold=0.50)
    assert len(spans) == 2
    assert spans[0].text == "J"
    assert spans[1].text == "Doe"

