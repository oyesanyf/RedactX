"""
RedactX: Differentially Private OpenJev Decision Engine for Line-Rate PHI/PII Detection.
Implements the OpenJev non-generative architecture for Google's VaultGemma:
- Prefix conditioning with terminal logit projection (bypassing generation)
- Jev Primitives: Noul (binary hypothesis), Choice (constrained categorical), Score (ordinal expectation)
- Active re-reads & evidential calibration (entropy thresholding + MC perturbation)
- Wire-compatible OpenJev REST protocol (POST /v1/decision)
- RLCD combined calibration loss (NLL + Brier + ECE)
"""

__version__ = "0.2.0"

from redactx.config import RedactXConfig, TrainingConfig, GatewayConfig
from redactx.primitives import (
    NoulDecision, DetectedSpan, ChoiceDecision,
    QuestionPayload, DecisionRequest, DecisionResponse,
    NoulResult, ChoiceResult, ScoreResult
)
from redactx.models.engine import RedactXDecisionEngine, RedactX
from redactx.models.openjev import OpenJevVaultGemmaEngine
from redactx.training.losses import RLCDCalibrationLoss

# Convenient alias for OpenJev engine
OpenJev = OpenJevVaultGemmaEngine

__all__ = [
    "RedactX",
    "OpenJev",
    "OpenJevVaultGemmaEngine",
    "RedactXDecisionEngine",
    "RedactXConfig",
    "TrainingConfig",
    "GatewayConfig",
    "NoulDecision",
    "DetectedSpan",
    "ChoiceDecision",
    "QuestionPayload",
    "DecisionRequest",
    "DecisionResponse",
    "NoulResult",
    "ChoiceResult",
    "ScoreResult",
    "RLCDCalibrationLoss"
]
