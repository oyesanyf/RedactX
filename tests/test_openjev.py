"""
Tests for OpenJev VaultGemma Decision Engine, Primitives, and RLCD Loss.
"""

import pytest
import torch
import torch.nn.functional as F

from redactx.config import RedactXConfig
from redactx.primitives import QuestionPayload, DecisionRequest
from redactx.models.openjev import OpenJevVaultGemmaEngine, resolve_single_token
from redactx.training.losses import RLCDCalibrationLoss


def test_rlcd_calibration_loss():
    loss_fn = RLCDCalibrationLoss(lambda_brier=0.5, lambda_ece=0.25)
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.randint(0, 2, (8,))

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


@pytest.fixture(scope="module")
def openjev_engine():
    config = RedactXConfig(
        base_model_id="distilbert/distilbert-base-uncased",
        max_length=128
    )
    return OpenJevVaultGemmaEngine(config=config)


def test_openjev_noul_primitive(openjev_engine):
    req = DecisionRequest(
        state="Patient Marcus Kowalski admitted on 04/12/2026 at Bethesda Memorial with acute chest pain.",
        questions=[
            QuestionPayload(
                key="is_phi",
                type="noul",
                question="Does this document contain patient Protected Health Information (PHI)?"
            )
        ],
        samples=1
    )
    resp = openjev_engine.evaluate_decision(req)

    assert "is_phi" in resp.answers
    ans = resp.answers["is_phi"]
    assert ans.type == "noul"
    assert 0.0 <= ans.probability <= 1.0
    assert 0.0 <= ans.confidence <= 1.0
    assert isinstance(ans.value, bool)
    assert resp.latency_ms > 0.0


def test_openjev_choice_primitive(openjev_engine):
    req = DecisionRequest(
        state="Diagnostic biopsy confirms invasive ductal carcinoma. Commenced paclitaxel 80mg/m2 weekly.",
        questions=[
            QuestionPayload(
                key="department",
                type="choice",
                question="Which clinical service should manage this patient?",
                options=["Oncology", "Cardiology", "Orthopedics"]
            )
        ],
        samples=1
    )
    resp = openjev_engine.evaluate_decision(req)

    assert "department" in resp.answers
    ans = resp.answers["department"]
    assert ans.type == "choice"
    assert ans.selection in ["Oncology", "Cardiology", "Orthopedics"]
    assert len(ans.distribution) == 3
    # Check probabilities sum to 1.0
    total_prob = sum(ans.distribution.values())
    assert abs(total_prob - 1.0) < 0.01


def test_openjev_score_primitive(openjev_engine):
    req = DecisionRequest(
        state="Emergency trauma: BP 70/40, HR 138 bpm, massive hemothorax, unresponsive GCS 6.",
        questions=[
            QuestionPayload(
                key="acuity",
                type="score",
                question="Emergency severity index rating",
                scale_min=1,
                scale_max=5
            )
        ],
        samples=1
    )
    resp = openjev_engine.evaluate_decision(req)

    assert "acuity" in resp.answers
    ans = resp.answers["acuity"]
    assert ans.type == "score"
    assert 1.0 <= ans.expected_value <= 5.0
    assert len(ans.distribution) == 5


def test_openjev_active_rereads(openjev_engine):
    # Forcing entropy threshold to 0.0 to trigger re-reads
    req = DecisionRequest(
        state="Vital signs stable. Follow up in 3 months.",
        questions=[
            QuestionPayload(
                key="audit",
                type="noul",
                question="Is billing code valid?"
            )
        ],
        samples=3,
        entropy_threshold=-1.0  # Guarantees trigger
    )
    resp = openjev_engine.evaluate_decision(req)
    ans = resp.answers["audit"]
    assert ans.re_reads_executed == 3
