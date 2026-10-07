"""
Unit tests for RedactX dataset loader.
"""

import os
import json
import tempfile
import pytest
from redactx.data.loader import load_dataset_source, normalize_record


def test_normalize_record():
    raw = {
        "text": "Patient admitted on 05/12/2026.",
        "label": "1",
        "spans": [{"text": "05/12/2026", "category": "DATE", "start": 20, "end": 30}]
    }
    norm = normalize_record(raw, 0)
    assert norm["primitive"] == "noul"
    assert norm["target"] == "Contains_PHI_PII"
    assert len(norm["spans"]) == 1
    assert norm["spans"][0]["category"] == "DATE"


def test_load_dataset_source_synthetic():
    data = load_dataset_source(source="synthetic", num_samples=30, phi_ratio=0.5, seed=42)
    assert len(data) == 30
    assert sum(1 for d in data if d["target"] == "Contains_PHI_PII") == 15


def test_load_dataset_source_json():
    sample_records = [
        {"context": "Clean clinical report.", "target": "Clean"},
        {"context": "Discharge note for Arthur Patel.", "target": "Contains_PHI_PII"}
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_records, f)
        temp_path = f.name

    try:
        loaded = load_dataset_source(source=temp_path)
        assert len(loaded) == 2
        assert loaded[0]["target"] == "Clean"
        assert loaded[1]["target"] == "Contains_PHI_PII"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_recipe_60_20_20_corpus():
    from redactx.data.benchmark_loaders import load_recipe_60_20_20_corpus

    corpus, stats = load_recipe_60_20_20_corpus(total_samples=10, seed=42)
    assert len(corpus) > 0
    assert stats["persona_records"] > 0
    assert stats["hard_negative_records"] > 0
    assert "60% Persona" in stats["recipe"]

    sample = corpus[0]
    assert "id" in sample
    assert "prompt" in sample
    assert "candidate_tokens" in sample
    assert "target_dist" in sample

