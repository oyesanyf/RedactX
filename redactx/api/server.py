"""
FastAPI Line-Rate Decision Gateway for OpenJev & RedactX.
Exposes wire-compatible OpenJev REST endpoints (/v1/decision) and batch scanning endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from redactx.primitives import (
    NoulDecision, DecisionRequest, DecisionResponse, QuestionPayload
)
from redactx.config import RedactXConfig
from redactx.models.engine import RedactXDecisionEngine
from redactx.models.openjev import OpenJevVaultGemmaEngine


class DecideRequest(BaseModel):
    id: Optional[str] = None
    context: str = Field(..., description="The clinical document or payload to evaluate")
    query: str = Field(
        "Does this text contain protected health information or personal identifiers?",
        description="Jev query"
    )
    primitive: str = Field("noul", description="Jev primitive type ('noul')")
    threshold: Optional[float] = Field(0.50, description="Decision threshold for PHI gating")
    extract_spans: Optional[bool] = Field(True, description="Whether to locate and return PHI spans")


class BatchScanRequest(BaseModel):
    contexts: List[str] = Field(..., description="List of clinical texts to scan in batch")
    threshold: Optional[float] = Field(0.50, description="Decision threshold for PHI gating")


class DecisionPayload(BaseModel):
    state: Optional[Union[Dict[str, Any], str]] = Field(None, description="State payload containing raw_text or metadata")
    text: Optional[str] = Field(None, description="Raw text context to evaluate")
    check_phi: bool = Field(True, description="Whether to check for PHI/PII")


def create_app(
    engine: Optional[Union[RedactXDecisionEngine, OpenJevVaultGemmaEngine]] = None
) -> FastAPI:
    app = FastAPI(
        title="OpenJev VaultGemma Decision Gateway",
        description="Wire-compatible Jev & OpenJev line-rate decision API with differential privacy",
        version="0.2.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize OpenJev engine by default
    if engine is not None:
        app_engine = engine
    else:
        app_engine = OpenJevVaultGemmaEngine()

    @app.get("/health")
    def health():
        return {
            "status": "online",
            "model_id": getattr(app_engine, "resolved_model_id", "vaultgemma"),
            "device": str(getattr(app_engine, "device", "cpu")),
            "engine": type(app_engine).__name__,
            "openjev_primitives_supported": ["noul", "choice", "score"]
        }

    # ==============================================================================
    # 1. Official OpenJev Wire-Compatible Endpoint
    # ==============================================================================
    @app.post("/v1/decision", response_model=DecisionResponse)
    def decision_endpoint(req: DecisionRequest) -> DecisionResponse:
        """
        OpenJev REST Wire Protocol:
        Evaluates Noul, Choice, and Score primitives in parallel against a shared state.
        Bypasses autoregressive loop, extracts terminal logits, and applies active re-reads.
        """
        if isinstance(app_engine, OpenJevVaultGemmaEngine):
            return app_engine.evaluate_decision(req)
        else:
            # Fallback wrapper
            state_str = str(req.state)
            ans = {}
            for q in req.questions:
                dec = app_engine.evaluate(state_str, query=q.question)
                ans[q.key] = {
                    "type": "noul",
                    "probability": dec.phi_probability,
                    "confidence": 1.0 - dec.clean_probability,
                    "value": not dec.passed_gate
                }
            return DecisionResponse(
                model=getattr(app_engine, "resolved_model_id", "vaultgemma"),
                answers=ans,
                latency_ms=10.0
            )

    # ==============================================================================
    # 2. RedactX Decision Evaluation Endpoint (Section 4 Contract)
    # ==============================================================================
    @app.post("/v1/redactx/evaluate")
    def evaluate_redaction_endpoint(payload: DecisionPayload) -> Dict[str, Any]:
        """
        Sub-40ms single-forward-pass redaction evaluation with differential privacy.
        Locks outputs to deterministic status ('REDACT', 'ESCALATE', 'PASS').
        """
        payload_data = payload.state if payload.state is not None else (payload.text or "")
        if hasattr(app_engine, "evaluate_redaction"):
            return app_engine.evaluate_redaction(payload_data)
        elif isinstance(app_engine, OpenJevVaultGemmaEngine):
            state_str = str(payload_data)
            req = DecisionRequest(
                state=state_str,
                questions=[QuestionPayload(
                    key="phi_gate",
                    type="noul",
                    question="Does this text contain Protected Health Information or Personal Identifiable Information?"
                )]
            )
            res = app_engine.evaluate_decision(req)
            ans = res.answers["phi_gate"]
            p_phi = ans.probability
            entropy = 1.0 - ans.confidence
            return {
                "is_phi_pii": p_phi >= 0.70,
                "phi_probability": round(p_phi, 4),
                "calibration_entropy": round(entropy, 4),
                "action": "REDACT" if p_phi >= 0.70 else ("ESCALATE" if entropy > 0.45 else "PASS")
            }
        else:
            state_str = str(payload.state)
            dec = app_engine.evaluate(state_str)
            p_phi = dec.phi_probability
            return {
                "is_phi_pii": p_phi >= 0.70,
                "phi_probability": round(p_phi, 4),
                "calibration_entropy": 0.1,
                "action": "REDACT" if p_phi >= 0.70 else "PASS"
            }
    @app.post("/v1/decide", response_model=NoulDecision)
    def decide(req: DecideRequest) -> NoulDecision:
        if isinstance(app_engine, OpenJevVaultGemmaEngine):
            d_req = DecisionRequest(
                state=req.context,
                questions=[QuestionPayload(key="gate", type="noul", question=req.query)]
            )
            resp = app_engine.evaluate_decision(d_req)
            ans = resp.answers["gate"]
            verdict = "Contains_PHI_PII" if ans.probability >= (req.threshold or 0.5) else "Clean"
            return NoulDecision(
                id=req.id,
                primitive="noul",
                context=req.context,
                query=req.query,
                verdict=verdict,
                phi_probability=ans.probability,
                clean_probability=round(1.0 - ans.probability, 4),
                passed_gate=(verdict == "Clean"),
                calibrated=True,
                latency_ms=resp.latency_ms,
                spans=resp.spans or []
            )
        else:
            return app_engine.evaluate(
                text=req.context,
                query=req.query,
                decision_threshold=req.threshold,
                extract_spans=req.extract_spans,
                payload_id=req.id
            )

    @app.post("/v1/scan", response_model=List[NoulDecision])
    def scan_batch(req: BatchScanRequest) -> List[NoulDecision]:
        decisions = []
        for ctx in req.contexts:
            d_req = DecisionRequest(
                state=ctx,
                questions=[QuestionPayload(
                    key="gate",
                    type="noul",
                    question="Does this text contain protected health information or personal identifiers?"
                )]
            )
            if isinstance(app_engine, OpenJevVaultGemmaEngine):
                resp = app_engine.evaluate_decision(d_req)
                ans = resp.answers["gate"]
                verdict = "Contains_PHI_PII" if ans.probability >= (req.threshold or 0.5) else "Clean"
                decisions.append(NoulDecision(
                    primitive="noul",
                    context=ctx,
                    query="Does this text contain protected health information or personal identifiers?",
                    verdict=verdict,
                    phi_probability=ans.probability,
                    clean_probability=round(1.0 - ans.probability, 4),
                    passed_gate=(verdict == "Clean"),
                    latency_ms=resp.latency_ms,
                    spans=resp.spans or []
                ))
            else:
                decisions.append(app_engine.evaluate(ctx, decision_threshold=req.threshold))
        return decisions

    return app
