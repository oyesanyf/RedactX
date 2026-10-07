"""
Tests for RedactX dataset collation and token alignment.
"""

import pytest
from transformers import AutoTokenizer
from redactx.data.generator import generate_redactx_corpus
from redactx.data.dataset import RedactXDataset


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def test_redactx_dataset_shapes(tokenizer):
    corpus = generate_redactx_corpus(num_samples=10, phi_ratio=0.5, seed=42)
    max_len = 128
    dataset = RedactXDataset(corpus, tokenizer, max_length=max_len, include_span_labels=True)

    assert len(dataset) == 10
    sample = dataset[0]

    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape == (max_len,)
    assert sample["attention_mask"].shape == (max_len,)
    assert sample["labels"].dtype == pytest.importorskip("torch").long

    if "token_span_labels" in sample:
        assert sample["token_span_labels"].shape == (max_len,)
