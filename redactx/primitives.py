"""
Jev decision primitives and OpenJev wire protocol schemas.
Supports Noul (Binary Hypothesis), Choice (Constrained Categorical), and Score (Ordinal Expectation).
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class DetectedSpan(BaseModel):
    """Represents an identified Protected Health Information (PHI) or PII span."""
    text: str = Field(..., description="Exact substring identified as PHI/PII")
    start: int = Field(..., description="Character start offset (0-indexed)")
    end: int = Field(..., description="Character end offset (exclusive)")
    category: str = Field(..., description="HIPAA Safe Harbor category (NAME, DATE, MRN, LOCATION, etc.)")
    confidence: float = Field(..., description="Calibrated confidence score [0.0, 1.0]")


# ==============================================================================
# OpenJev Wire Protocol Request Models
# ==============================================================================
class QuestionPayload(BaseModel):
    """
    OpenJev Question definition.
    type: 'noul' | 'choice' | 'score'
    """
    key: str = Field(..., description="Unique slot identifier in the answers dictionary")
    type: str = Field("noul", description="'noul', 'choice', or 'score'")
    question: str = Field(..., description="The query, hypothesis, or rating question")
    options: Optional[List[str]] = Field(None, description="Candidate categories for 'choice' primitive")
    labels: Optional[List[str]] = Field(None, description="Optional custom anchor labels (e.g. ['clean', 'phi'])")
    scale_min: Optional[int] = Field(1, description="Minimum ordinal value for 'score' primitive")
    scale_max: Optional[int] = Field(5, description="Maximum ordinal value for 'score' primitive")


class DecisionRequest(BaseModel):
    """
    OpenJev REST request specification for POST /v1/decision.
    """
    model: Optional[str] = Field("openjev-latest", description="Target model variant")
    state: Union[Dict[str, Any], str] = Field(..., description="Shared state context (clinical payload, metrics, JSON)")
    questions: List[QuestionPayload] = Field(..., description="Parallel question slots to evaluate against state")
    samples: Optional[int] = Field(1, description="Re-read samples for evidential uncertainty reduction")
    entropy_threshold: Optional[float] = Field(0.10, description="Trigger automatic re-reads if Shannon entropy > threshold")


# ==============================================================================
# OpenJev Wire Protocol Response Results
# ==============================================================================
class NoulResult(BaseModel):
    """
    Jev 'noul' primitive (Bernoulli trial): Binary hypothesis evaluation.
    """
    type: str = "noul"
    probability: float = Field(..., description="P(true) in [0.0, 1.0]")
    confidence: float = Field(..., description="1.0 - normalized Shannon entropy in [0.0, 1.0]")
    value: bool = Field(..., description="True if probability >= threshold")
    re_reads_executed: int = Field(0, description="Number of active re-reads performed")


class ChoiceResult(BaseModel):
    """
    Jev 'choice' primitive: Constrained categorical routing over dynamic options.
    """
    type: str = "choice"
    selection: str = Field(..., description="Winning option label")
    distribution: Dict[str, float] = Field(..., description="Masked Softmax normalized probability distribution")
    confidence: float = Field(..., description="Confidence score")
    re_reads_executed: int = Field(0, description="Number of active re-reads performed")


class ScoreResult(BaseModel):
    """
    Jev 'score' primitive: Ordinal threshold expectation E[score] over discrete scale.
    """
    type: str = "score"
    expected_value: float = Field(..., description="Mathematical expectation sum(m * P(level_m))")
    distribution: Dict[str, float] = Field(..., description="Categorical probability distribution over scale levels")
    confidence: float = Field(..., description="Confidence score")
    re_reads_executed: int = Field(0, description="Number of active re-reads performed")


class DecisionResponse(BaseModel):
    """
    OpenJev REST response specification.
    """
    model: str = "vaultgemma-openjev"
    answers: Dict[str, Union[NoulResult, ChoiceResult, ScoreResult, Dict[str, Any]]]
    latency_ms: float
    spans: Optional[List[DetectedSpan]] = Field(default_factory=list)


# Backward-compatible NoulDecision
class NoulDecision(BaseModel):
    id: Optional[str] = None
    primitive: str = "noul"
    context: str
    query: str
    verdict: str
    phi_probability: float
    clean_probability: float
    passed_gate: bool
    calibrated: bool = True
    confidence: float = 1.0
    latency_ms: float
    spans: List[DetectedSpan] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChoiceDecision(BaseModel):
    id: Optional[str] = None
    primitive: str = "choice"
    context: str
    query: str
    selection: str
    probabilities: Dict[str, float]
    confidence: float
    latency_ms: float
