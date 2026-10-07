"""
End-to-end tests of the production API server and of the engine/detector integration (CPU only).

* Server tests use the real Presidio detector (mode=presidio) through FastAPI's TestClient.
* Engine tests build a real RedactX checkpoint from the gpt2 backbone with a freshly initialised span head;
  they check batching parity, long-document stitching and the hybrid server path - not detection quality.
"""

import json
import os

import pytest

pytest.importorskip("presidio_analyzer")
from fastapi.testclient import TestClient  # noqa: E402

from redactx.production.settings import Settings, hash_api_key  # noqa: E402

KEY = "test-key-0123456789abcdef"
PHI = "Please call Jennifer Okonkwo at 312-555-0198 about her biopsy results."


def _settings(tmp_path, **env):
    base = {"REDACTX_MODE": "presidio", "REDACTX_API_KEY_HASHES": hash_api_key(KEY),
            "REDACTX_AUDIT_LOG": str(tmp_path / "audit.jsonl")}
    base.update({k: str(v) for k, v in env.items()})
    return Settings.from_env(env=base, dotenv_path=None)


@pytest.fixture(scope="module")
def presidio_service():
    from redactx.production.server import Service
    s = Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_ALLOW_NO_AUTH": "1"}, dotenv_path=None)
    svc = Service.from_settings(s)
    svc.warmup()
    return svc


def _client(tmp_path, service, **env):
    from redactx.production.server import create_production_app
    settings = _settings(tmp_path, **env)
    service.settings = settings
    return TestClient(create_production_app(settings, service=service)), settings


H = {"X-API-Key": KEY}


def test_health_ready_and_auth(tmp_path, presidio_service):
    c, _ = _client(tmp_path, presidio_service)
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert c.post("/v2/redact", json={"text": PHI}).status_code == 401
    assert c.post("/v2/redact", json={"text": PHI}, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = c.post("/v2/redact", json={"text": PHI}, headers={"Authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200
    assert c.get("/metrics").status_code == 401


def test_redact_detect_batch_and_no_phi_leaks(tmp_path, presidio_service):
    c, settings = _client(tmp_path, presidio_service)
    r = c.post("/v2/redact", json={"text": PHI}, headers={**H, "X-Request-ID": "req-123"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "req-123"
    body = r.json()
    assert body["action"] == "REDACTED"
    assert "Jennifer" not in body["redacted_text"] and "312-555-0198" not in body["redacted_text"]
    assert all("text" not in f for f in body["findings"]), "finding values must not be returned by default"
    assert body["model_version"].startswith("presidio:")

    d = c.post("/v2/detect", json={"text": PHI}, headers=H).json()
    assert d["contains_phi"] and "redacted_text" not in d

    b = c.post("/v2/redact/batch", json={"texts": [PHI, "The server restarted cleanly."]}, headers=H).json()
    assert [x["action"] for x in b["results"]] == ["REDACTED", "PASS"]

    s = c.post("/v2/redact", json={"text": PHI, "strategy": "char"}, headers=H).json()
    assert len(s["redacted_text"]) == len(PHI)
    assert c.post("/v2/redact", json={"text": PHI, "strategy": "pseudonym"}, headers=H).status_code == 422

    audit = open(settings.audit_log, encoding="utf-8").read()
    assert "Jennifer" not in audit and "312-555" not in audit and "biopsy" not in audit
    events = [json.loads(line) for line in audit.splitlines()]
    assert events and events[0]["request_id"] == "req-123" and events[0]["categories"].get("NAME") == 1

    m = c.get("/metrics", headers=H).text
    assert "redactx_requests_total" in m and 'redactx_findings_total{category="NAME"}' in m
    assert "Jennifer" not in m

    info = c.get("/v2/info", headers=H).json()
    assert info["mode"] == "presidio" and info["limits"]["max_batch"] == settings.max_batch


def test_validation_errors_do_not_echo_input(tmp_path, presidio_service):
    c, _ = _client(tmp_path, presidio_service)
    r = c.post("/v2/redact", json={"txt": PHI}, headers=H)
    assert r.status_code == 422 and "Jennifer" not in r.text
    r2 = c.post("/v2/redact", content=b"not json " + PHI.encode(), headers={**H, "Content-Type": "application/json"})
    assert r2.status_code == 422 and "Jennifer" not in r2.text


def test_size_limits(tmp_path, presidio_service):
    c, _ = _client(tmp_path, presidio_service, REDACTX_MAX_CHARS=100, REDACTX_MAX_BATCH=2,
                   REDACTX_MAX_BODY_BYTES=2048)
    r = c.post("/v2/redact", json={"text": "a" * 101}, headers=H)
    assert r.status_code == 413 and r.json()["error"] == "document_too_large"
    r = c.post("/v2/redact/batch", json={"texts": ["a", "b", "c"]}, headers=H)
    assert r.status_code == 413 and r.json()["error"] == "batch_too_large"
    r = c.post("/v2/redact", json={"text": "a" * 5000}, headers=H)
    assert r.status_code == 413 and r.json()["error"] == "payload_too_large"

    def chunks():  # chunked upload without Content-Length must also be rejected
        yield b'{"text": "'
        for _ in range(10):
            yield b"a" * 500
        yield b'"}'
    r = c.post("/v2/redact", content=chunks(), headers={**H, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_rate_limit(tmp_path, presidio_service):
    c, _ = _client(tmp_path, presidio_service, REDACTX_RATE_LIMIT_PER_MIN=1, REDACTX_RATE_BURST=2)
    codes = [c.post("/v2/detect", json={"text": "hello"}, headers=H).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/v2/detect", json={"text": "hello"}, headers=H)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_detection_failure_fails_closed(tmp_path, presidio_service):
    c, _ = _client(tmp_path, presidio_service)
    original = presidio_service.redactor.detector.analyzer
    try:
        presidio_service.redactor.detector.analyzer = None  # real failure: analyzer unavailable -> AttributeError
        r = c.post("/v2/redact", json={"text": PHI}, headers=H)
        assert r.status_code == 503 and r.json()["error"] == "detection_failed"
        assert "Jennifer" not in r.text
    finally:
        presidio_service.redactor.detector.analyzer = original


def test_server_refuses_to_start_without_auth():
    with pytest.raises(ValueError, match="No API keys"):
        Settings.from_env(env={"REDACTX_MODE": "presidio"}, dotenv_path=None)


# ============================================================================ engine integration
# gpt2_checkpoint fixture: tests/conftest.py


@pytest.fixture(scope="module")
def engine(gpt2_checkpoint):
    from redactx.models.openjev import OpenJevVaultGemmaEngine
    return OpenJevVaultGemmaEngine.from_pretrained(gpt2_checkpoint, device="cpu")


def test_engine_runs_fp32_on_cpu_and_batch_matches_single(engine):
    import torch
    assert next(engine.model.parameters()).dtype == torch.float32
    texts = [PHI, "The server restarted after the configuration change.",
             'He said "call me at (555) 201-3344"\nthen left. Résumé of Zoë.']
    batch = engine.score_texts(texts, batch_size=3)
    for t, b in zip(texts, batch):
        single = engine.evaluate_text(t)
        assert abs(b["p_phi"] - single.phi_probability) < 1e-3
        assert [(s.start, s.end) for s in b["spans"]] == [(s.start, s.end) for s in single.spans]
        for s in b["spans"]:
            assert t[s.start:s.end] == s.text


def test_redactx_detector_long_document_stitching(engine):
    from redactx.production.detectors import RedactXDetector
    det = RedactXDetector(engine, max_chars=300, overlap=100, batch_size=4)
    doc = " ".join([PHI] * 25)
    d = det.detect(doc)
    assert d.windows > 1 and d.doc_score is not None
    prev_end = -1
    for f in d.findings:
        assert doc[f.start:f.end] == f.text and f.start >= prev_end
        prev_end = f.end
    many = det.detect_many([PHI, doc, ""])
    assert len(many) == 3 and many[2].findings == []


def test_hybrid_server_end_to_end(tmp_path, gpt2_checkpoint):
    from redactx.production.server import create_production_app
    settings = Settings.from_env(env={
        "REDACTX_MODE": "hybrid", "REDACTX_MODEL_DIR": gpt2_checkpoint, "REDACTX_DEVICE": "cpu",
        "REDACTX_API_KEY_HASHES": hash_api_key(KEY), "REDACTX_AUDIT_LOG": str(tmp_path / "a.jsonl"),
        "REDACTX_CHUNK_CHARS": "300", "REDACTX_CHUNK_OVERLAP": "100"}, dotenv_path=None)
    c = TestClient(create_production_app(settings))
    assert c.get("/readyz").status_code == 200
    r = c.post("/v2/redact", json={"text": PHI}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["detectors"] == ["redactx", "presidio"] and body["model_version"].startswith("hybrid:")
    assert "Jennifer" not in body["redacted_text"]  # Presidio alone guarantees this; union can only add
    info = c.get("/v2/info", headers=H).json()
    assert info["has_span_head"] is True and info["chunk_chars"] == 300
