"""
End-to-end unit and integration tests for RedactXDecisionEngine.
"""

import os
import shutil
import tempfile
import pytest
from redactx.config import RedactXConfig
from redactx.models.engine import RedactXDecisionEngine


@pytest.fixture(scope="module")
def engine():
    # Use lightweight local fallback for fast testing
    config = RedactXConfig(
        base_model_id="distilbert/distilbert-base-uncased",
        max_length=128
    )
    return RedactXDecisionEngine(config=config)


def test_engine_single_evaluation(engine):
    clean_text = (
        "Patient presented with seasonal allergies and mild rhinorrhea. "
        "Prescribed cetirizine 10mg PO daily. Symptoms improving."
    )
    decision = engine.evaluate(clean_text, decision_threshold=0.50)

    assert decision.primitive == "noul"
    assert decision.context == clean_text
    assert decision.verdict in ["Clean", "Contains_PHI_PII"]
    assert 0.0 <= decision.phi_probability <= 1.0
    assert 0.0 <= decision.clean_probability <= 1.0
    assert abs(decision.phi_probability + decision.clean_probability - 1.0) < 0.01
    assert decision.latency_ms > 0.0
    assert isinstance(decision.passed_gate, bool)


def test_engine_batch_evaluation(engine):
    texts = [
        "Routine lipid screening requested. Total cholesterol 180 mg/dL.",
        "Discharge notice for Devon Washington admitted on 05/14/2026."
    ]
    decisions = engine.evaluate_batch(texts, decision_threshold=0.50)
    assert len(decisions) == 2
    for d in decisions:
        assert d.primitive == "noul"
        assert d.verdict in ["Clean", "Contains_PHI_PII"]
        assert d.latency_ms > 0.0


def test_checkpoint_save_and_load(engine):
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine.save_checkpoint(tmp_dir)

        assert os.path.exists(os.path.join(tmp_dir, "redactx_scorer.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "redactx_calibrator.pt"))

        # Test loading into same engine
        engine.load_checkpoint(tmp_dir)
