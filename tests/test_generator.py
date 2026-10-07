"""
Tests for RedactX synthetic dataset generator.
"""

import pytest
from redactx.data.generator import generate_redactx_corpus, HARD_NEGATIVE_TEMPLATES


def test_generator_corpus_balance():
    corpus = generate_redactx_corpus(num_samples=40, phi_ratio=0.5, seed=123)
    assert len(corpus) == 40

    clean_samples = [c for c in corpus if c["target"] == "Clean"]
    phi_samples = [c for c in corpus if c["target"] == "Contains_PHI_PII"]

    assert len(clean_samples) == 20
    assert len(phi_samples) == 20


def test_generator_span_alignment():
    corpus = generate_redactx_corpus(num_samples=20, phi_ratio=1.0, seed=42)
    for record in corpus:
        context = record["context"]
        spans = record["spans"]
        assert len(spans) > 0, "PHI record should contain spans"
        for s in spans:
            extracted = context[s["start"]:s["end"]]
            assert extracted == s["text"], f"Span mismatch: '{extracted}' != '{s['text']}'"
            assert s["category"] in ["NAME", "DATE", "LOCATION", "IDENTIFIER", "CONTACT"]


def test_generator_clean_spans_empty():
    corpus = generate_redactx_corpus(num_samples=20, phi_ratio=0.0, seed=42)
    for record in corpus:
        assert record["target"] == "Clean"
        assert record["spans"] == []
        assert record["primitive"] == "noul"
