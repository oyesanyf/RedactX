"""
Unit tests for RedactX model heads and components.
"""

import pytest
import torch
from redactx.models.scorer import RedactXPointerScorer, PlattTemperatureScaler
from redactx.models.span_locator import TokenSpanLocator


def test_hobson_pointer_scorer():
    hidden_dim = 256
    num_classes = 2
    batch_size = 4

    scorer = RedactXPointerScorer(hidden_dim=hidden_dim, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, hidden_dim)

    logits = scorer(dummy_input)
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_token_span_locator():
    hidden_dim = 128
    batch_size = 2
    seq_len = 16

    locator = TokenSpanLocator(hidden_dim=hidden_dim)
    dummy_hidden = torch.randn(batch_size, seq_len, hidden_dim)

    logits = locator(dummy_hidden)
    assert logits.shape == (batch_size, seq_len)

    # Test span extraction logic
    token_probs = torch.tensor([0.1, 0.1, 0.95, 0.92, 0.1, 0.88, 0.1])
    offsets = [(0, 0), (0, 8), (9, 15), (16, 24), (25, 30), (31, 41), (42, 45)]
    prompt = "Context: Patient Marcus Kowalski visited 04/12/2026 for cough.\nQuery: PHI?\nDecision:"
    context = "Patient Marcus Kowalski visited 04/12/2026 for cough."

    spans = locator.extract_spans(
        token_probs=token_probs,
        offset_mapping=offsets,
        full_prompt=prompt,
        context_text=context,
        threshold=0.5
    )
    assert len(spans) >= 1
    for s in spans:
        assert s.confidence >= 0.5
        assert len(s.text) > 0


def test_temperature_scaler():
    scaler = PlattTemperatureScaler(initial_temperature=1.5)
    logits = torch.tensor([[2.0, -1.0], [0.0, 3.0]])
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 1.5)

    probs = scaler.predict_proba(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
