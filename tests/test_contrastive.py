"""
Tests for the contrastive training path: shared prompt builder, token span labels,
span-preserving text utilities, dataset collation, and raw-span extraction.
All tests run on real tokenizers and real strings (no model weights needed).
"""

import json
import os

import pytest
import torch
from transformers import AutoTokenizer

from redactx.data.prompting import (
    DEFAULT_NOUL_QUESTION, build_noul_prompt, token_raw_ranges, token_span_labels
)
from redactx.data.contrastive_corpus import (
    clean_spans, truncate_with_spans, replace_spans_with_generic, generic_phrase, make_record
)
from redactx.data.openjev_dataset import OpenJevCalibratedDataset, collate_openjev_batch
from redactx.models.span_locator import TokenSpanLocator

TRICKY_TEXTS = [
    "Patient John Smith, DOB 03/14/1962, MRN 448-22-9191.",
    'He said "call me at (555) 201-3344"\nthen left.',
    "Résumé of Zoë Ångström — email zoe@example.org\ttab",
    "",
]

LOCAL_GEMMA_TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "models", "RedactX")


@pytest.fixture(scope="module")
def distil_tok():
    return AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def _tokenizers(distil_tok):
    toks = [distil_tok]
    if os.path.exists(os.path.join(LOCAL_GEMMA_TOKENIZER, "tokenizer_config.json")):
        toks.append(AutoTokenizer.from_pretrained(LOCAL_GEMMA_TOKENIZER))
    return toks


@pytest.mark.parametrize("text", TRICKY_TEXTS)
def test_prompt_identical_to_json_dumps(text):
    prompt, raw_index = build_noul_prompt(text, DEFAULT_NOUL_QUESTION)
    expected = f"[STATE]: {json.dumps({'raw_text': text})}\n[DECISION]: {DEFAULT_NOUL_QUESTION}\n[VERDICT]:"
    assert prompt == expected
    assert len(raw_index) == len(prompt)
    # Every raw character is represented, in order
    mapped = [i for i in raw_index if i >= 0]
    assert sorted(set(mapped)) == list(range(len(text)))
    assert mapped == sorted(mapped)


@pytest.mark.parametrize("text", TRICKY_TEXTS[:3])
def test_token_raw_ranges_point_back_to_raw_text(text, distil_tok):
    for tok in _tokenizers(distil_tok):
        prompt, raw_index = build_noul_prompt(text)
        enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
        ranges = token_raw_ranges(enc["offset_mapping"], raw_index)
        covered = set()
        for rs, re_ in ranges:
            if rs >= 0:
                assert 0 <= rs < re_ <= len(text)
                covered.update(range(rs, re_))
        # all non-space raw characters must be covered by some token
        assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_token_span_labels_mark_exactly_the_pii(distil_tok):
    text = "Patient John Smith was seen on 03/14/1962 for chest pain."
    s_name = text.index("John Smith")
    s_date = text.index("03/14/1962")
    spans = [(s_name, s_name + len("John Smith")), (s_date, s_date + len("03/14/1962"))]
    for tok in _tokenizers(distil_tok):
        prompt, raw_index = build_noul_prompt(text)
        enc = tok(prompt, return_offsets_mapping=True)
        labels = token_span_labels(enc["offset_mapping"], raw_index, spans)
        ranges = token_raw_ranges(enc["offset_mapping"], raw_index)
        positive_text = "".join(
            text[rs:re_] for (rs, re_), lab in zip(ranges, labels) if lab == 1
        ).replace(" ", "").lower()
        assert "johnsmith" in positive_text and "03/14/1962" in positive_text
        assert "chest" not in positive_text and "patient" not in positive_text
        # Template tokens are ignored
        assert labels[0] == -100 and labels[-1] == -100
        assert 0 in labels


def test_clean_truncate_replace():
    text = "Call Jane Doe at 555-123-4567 tomorrow."
    a = text.index("Jane Doe"); b = text.index("555-123-4567")
    spans = clean_spans([(b, b + 12, "phone_number"), (a, a + 8, "first_name"), (a, a + 4, "first_name"),
                         (100, 120, "x")], len(text))
    assert spans == [(a, a + 8, "first_name"), (b, b + 12, "phone_number")]
    clean = replace_spans_with_generic(text, spans)
    assert clean == "Call the person at their phone number tomorrow."
    assert generic_phrase("date_of_birth") == "a recent date"
    t2, s2 = truncate_with_spans(text, spans, 20)
    assert len(t2) <= 20 and all(e <= len(t2) for _, e, _ in s2)
    assert t2[s2[0][0]:s2[0][1]] == "Jane Doe"


def test_dataset_emits_span_labels_and_collates(distil_tok):
    text = "Patient Mary Major, phone 555-867-5309, discharged."
    s = text.index("Mary Major"); p = text.index("555-867-5309")
    pos = make_record("r1", text, [(s, s + 10, "name"), (p, p + 12, "phone")], True, "test", "positive")
    neg = make_record("r2", replace_spans_with_generic(text, [(s, s + 10, "name"), (p, p + 12, "phone")]),
                      [], False, "test", "contrastive_negative")
    ds = OpenJevCalibratedDataset([pos, neg], distil_tok, max_length=128)
    a, b = ds[0], ds[1]
    assert "token_span_labels" in a and "token_span_labels" in b
    assert (torch.as_tensor(a["token_span_labels"]) == 1).sum() > 0
    assert (torch.as_tensor(b["token_span_labels"]) == 1).sum() == 0
    batch = collate_openjev_batch([a, b])
    assert batch["token_span_labels"].shape == batch["input_ids"].shape
    pad_positions = batch["attention_mask"] == 0
    assert (batch["token_span_labels"][pad_positions] == -100).all()


def test_extract_raw_spans_round_trip(distil_tok):
    text = 'Note: "Ann Lee" phoned from 555-222-3333.\nOK'
    a = text.index("Ann Lee"); b = text.index("555-222-3333")
    gold = [(a, a + 7), (b, b + 12)]
    prompt, raw_index = build_noul_prompt(text)
    enc = distil_tok(prompt, return_offsets_mapping=True)
    labels = token_span_labels(enc["offset_mapping"], raw_index, gold)
    ranges = token_raw_ranges(enc["offset_mapping"], raw_index)
    # Feed the gold labels as "probabilities": extraction must recover the gold spans exactly.
    probs = torch.tensor([1.0 if lab == 1 else 0.0 for lab in labels])
    loc = TokenSpanLocator(hidden_dim=8)
    spans = loc.extract_raw_spans(probs, ranges, text, threshold=0.5)
    assert [(sp.start, sp.end) for sp in spans] == gold
    assert [sp.text for sp in spans] == ["Ann Lee", "555-222-3333"]
