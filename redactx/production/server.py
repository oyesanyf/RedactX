"""
Production RedactX API server.

Security / reliability properties
  * Authentication: API keys (X-API-Key or Authorization: Bearer), stored only as SHA-256 hashes and
    compared in constant time. The server refuses to start without keys unless REDACTX_ALLOW_NO_AUTH=1.
  * Rate limiting: token bucket per API key -> 429 with Retry-After.
  * Size limits: request body bytes (streaming check), characters per document, documents per batch -> 413.
  * Backpressure: bounded inference concurrency + bounded queue -> 503 when saturated; per-request timeout -> 504.
  * Fail-closed: if detection fails, the request fails (5xx). Unredacted text is never returned on error.
  * No PHI in logs or errors: the audit log records counts/categories/latency only; validation errors are
    returned without echoing input; finding values are omitted from responses unless explicitly enabled.
  * Observability: request IDs, /healthz (liveness), /readyz (model loaded + warmed up), /metrics (Prometheus).
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter as CCounter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from redactx.production.metrics import Registry
from redactx.production.redactor import STRATEGIES, Redactor
from redactx.production.settings import Settings, hash_api_key

API_VERSION = "2.0.0"
_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
logger = logging.getLogger("redactx.server")


# ====================================================================== service (model + redactor)
def model_fingerprint(model_dir: Optional[str]) -> str:
    """Short, stable fingerprint of the deployed model artefacts (config, thresholds, span head, weights head)."""
    if not model_dir:
        return "none"
    h = hashlib.sha256()
    for name in ("config.json", "redactx_thresholds.json", "temperature.json", "redactx_span_locator.pt",
                 "model.safetensors"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            h.update(name.encode())
            h.update(str(os.path.getsize(p)).encode())
            with open(p, "rb") as f:
                h.update(f.read(1 << 20))  # first 1 MiB is enough to distinguish checkpoints
    return h.hexdigest()[:12]


class Service:
    """Owns the detector + redactor. Built once at startup."""

    def __init__(self, settings: Settings, redactor: Redactor, model_version: str):
        self.settings = settings
        self.redactor = redactor
        self.model_version = model_version
        self.ready = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "Service":
        from redactx.production.detectors import HybridDetector, PresidioDetector, RedactXDetector
        rx = None
        if settings.mode in ("hybrid", "redactx"):
            from redactx.models.openjev import OpenJevVaultGemmaEngine
            engine = OpenJevVaultGemmaEngine.from_pretrained(settings.model_dir, device=settings.device)
            rx = RedactXDetector(engine, max_chars=settings.chunk_chars, overlap=settings.chunk_overlap,
                                 batch_size=settings.batch_size)
            if not rx.has_span_head:
                logger.warning("Checkpoint has no trained span head: RedactX can flag PHI but cannot localize it; "
                               "flagged documents follow REDACTX_UNLOCALIZED_POLICY=%s", settings.unlocalized_policy)
        if settings.mode == "hybrid":
            detector = HybridDetector(rx, PresidioDetector(score_threshold=settings.presidio_score))
        elif settings.mode == "redactx":
            detector = rx
        else:
            detector = PresidioDetector(score_threshold=settings.presidio_score)
        redactor = Redactor(detector, strategy=settings.strategy, unlocalized_policy=settings.unlocalized_policy,
                            hmac_key=settings.hmac_key, hipaa_only=settings.hipaa_only)
        return cls(settings, redactor, f"{settings.mode}:{model_fingerprint(settings.model_dir)}")

    def warmup(self) -> None:
        self.redactor.redact_many(["Warm-up request: patient seen in clinic for follow-up."])
        self.ready = True

    def describe(self) -> Dict[str, Any]:
        det = self.redactor.detector
        rx = getattr(det, "redactx", det if hasattr(det, "engine") else None)
        info: Dict[str, Any] = {"mode": self.settings.mode, "model_version": self.model_version,
                                "strategy": self.redactor.strategy,
                                "unlocalized_policy": self.redactor.unlocalized_policy,
                                "hipaa_only": self.redactor.hipaa_only}
        if rx is not None and hasattr(rx, "engine"):
            th = getattr(rx.engine, "thresholds", None)
            info.update({"has_span_head": rx.has_span_head, "doc_threshold": rx.doc_threshold,
                         "span_threshold": getattr(rx.engine, "span_threshold", None),
                         "thresholds_calibrated_on": getattr(th, "calibrated_on", None),
                         "chunk_chars": rx.max_chars, "chunk_overlap": rx.overlap})
        return info


# ====================================================================== rate limiting
class TokenBucketLimiter:
    def __init__(self, per_minute: int, burst: int):
        self.rate = per_minute / 60.0
        self.capacity = float(burst)
        self._state: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> Optional[float]:
        """Returns None if allowed, else seconds to wait."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= cost:
                self._state[key] = [tokens - cost, now]
                return None
            self._state[key] = [tokens, now]
            return (cost - tokens) / self.rate if self.rate > 0 else 60.0


# ====================================================================== body size limit (ASGI)
class BodySizeLimitMiddleware:
    """Rejects bodies larger than max_bytes, including chunked uploads without Content-Length."""

    def __init__(self, app, max_bytes: int, on_reject=None):
        self.app, self.max_bytes, self.on_reject = app, max_bytes, on_reject

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > self.max_bytes:
            return await self._reject(scope, send)
        received = 0
        too_big = False

        async def limited_receive():
            nonlocal received, too_big
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_big = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        started = False

        async def guarded_send(message):
            nonlocal started
            if too_big and not started:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
        if too_big and not started:
            await self._reject(scope, send)

    async def _reject(self, scope, send):
        if self.on_reject:
            self.on_reject()
        body = json.dumps({"error": "payload_too_large", "max_bytes": self.max_bytes}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


# ====================================================================== audit log
def make_audit_logger(target: str) -> logging.Logger:
    log = logging.getLogger("redactx.audit")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout) if target in ("-", "", "stdout") else logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    return log


# ====================================================================== request models
class RedactRequest(BaseModel):
    text: str = Field(..., description="Document to redact")
    strategy: Optional[str] = Field(None, description=f"Override strategy: one of {STRATEGIES}")


class BatchRedactRequest(BaseModel):
    texts: List[str] = Field(..., description="Documents to redact")
    strategy: Optional[str] = None


class DetectRequest(BaseModel):
    text: str


# ====================================================================== app factory
def create_production_app(settings: Settings, service: Optional[Service] = None, warmup: bool = True) -> FastAPI:
    for w in settings.warnings:
        logger.warning(w)
    metrics = Registry()
    audit = make_audit_logger(settings.audit_log)
    limiter = TokenBucketLimiter(settings.rate_limit_per_min, settings.rate_burst)
    executor = ThreadPoolExecutor(max_workers=settings.max_concurrency, thread_name_prefix="redactx-infer")
    admission_lock = threading.Lock()
    state = {"admitted": 0}
    key_hashes = tuple(settings.api_key_hashes)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=False)

    app = FastAPI(title="RedactX", version=API_VERSION,
                  description="PHI/PII detection and redaction (RedactX + Presidio).",
                  docs_url="/docs", redoc_url=None, lifespan=lifespan)

    if service is None:
        service = Service.from_settings(settings)
    app.state.service = service
    if warmup and not service.ready:
        service.warmup()
    metrics.ready.set(1.0 if service.ready else 0.0)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "X-API-Key", "Content-Type"])
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes,
                       on_reject=lambda: metrics.rejected.inc(reason="body_too_large"))

    # ------------------------------------------------------------------ middleware: request id + metrics
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if _REQ_ID.match(rid) else uuid.uuid4().hex
        request.state.request_id = rid
        request.state.key_id = "-"
        t0 = time.perf_counter()
        metrics.inflight.inc(1)
        try:
            response = await call_next(request)
        finally:
            metrics.inflight.inc(-1)
        elapsed = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.requests.inc(endpoint=path, status=str(response.status_code))
        metrics.latency.observe(elapsed, endpoint=path)
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # ------------------------------------------------------------------ error handlers (no PHI echo)
    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid_request", "details": errors,
                                                      "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        headers = getattr(exc, "headers", None)
        content = dict(exc.detail) if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        content["request_id"] = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=exc.status_code, content=content, headers=headers)

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        metrics.errors.inc(type=type(exc).__name__)
        logger.error("unhandled %s (request_id=%s)", type(exc).__name__, getattr(request.state, "request_id", None))
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                      "request_id": getattr(request.state, "request_id", None)})

    # ------------------------------------------------------------------ auth + rate limit dependency
    def authenticate(request: Request) -> str:
        if not key_hashes:
            if settings.allow_no_auth:
                request.state.key_id = "anonymous"
                return "anonymous"
            raise HTTPException(status_code=401, detail={"error": "unauthorized"})
        supplied = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not supplied and auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if not supplied:
            metrics.rejected.inc(reason="unauthorized")
            raise HTTPException(status_code=401, detail={"error": "unauthorized"},
                                headers={"WWW-Authenticate": "Bearer"})
        digest = hash_api_key(supplied)
        match = None
        for h in key_hashes:  # constant-time comparison against every configured key
            if hmac.compare_digest(digest, h):
                match = h
        if match is None:
            metrics.rejected.inc(reason="unauthorized")
            raise HTTPException(status_code=401, detail={"error": "unauthorized"},
                                headers={"WWW-Authenticate": "Bearer"})
        key_id = match[:8]
        request.state.key_id = key_id
        return key_id

    def rate_limited(cost: float):
        def dep(request: Request, key_id: str = Depends(authenticate)) -> str:
            wait = limiter.allow(key_id, cost)
            if wait is not None:
                metrics.rejected.inc(reason="rate_limited")
                raise HTTPException(status_code=429, detail={"error": "rate_limited"},
                                    headers={"Retry-After": str(max(1, int(wait + 0.999)))})
            return key_id
        return dep

    # ------------------------------------------------------------------ inference with backpressure
    def _release_slot(_f=None) -> None:
        with admission_lock:
            state["admitted"] -= 1

    async def run_inference(fn):
        if not service.ready:
            raise HTTPException(status_code=503, detail={"error": "not_ready"})
        # Admission control: at most max_concurrency running (executor size) + max_queue waiting.
        with admission_lock:
            admitted = state["admitted"] < settings.max_concurrency + settings.max_queue
            if admitted:
                state["admitted"] += 1
        if not admitted:
            metrics.rejected.inc(reason="overloaded")
            raise HTTPException(status_code=503, detail={"error": "overloaded"}, headers={"Retry-After": "1"})
        cfut = executor.submit(fn)
        cfut.add_done_callback(_release_slot)  # slot is freed only when the work really ends (even after a timeout)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(cfut), timeout=settings.request_timeout_s)
        except asyncio.TimeoutError:
            metrics.errors.inc(type="timeout")
            raise HTTPException(status_code=504, detail={"error": "timeout"})
        except HTTPException:
            raise
        except Exception as exc:  # fail closed: never return unredacted text on error
            metrics.errors.inc(type=type(exc).__name__)
            logger.error("inference failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail={"error": "detection_failed"})

    def check_texts(texts: List[str]) -> None:
        if not texts:
            raise HTTPException(status_code=422, detail={"error": "empty_batch"})
        if len(texts) > settings.max_batch:
            metrics.rejected.inc(reason="batch_too_large")
            raise HTTPException(status_code=413, detail={"error": "batch_too_large", "max_batch": settings.max_batch})
        for i, t in enumerate(texts):
            if len(t) > settings.max_chars:
                metrics.rejected.inc(reason="document_too_large")
                raise HTTPException(status_code=413, detail={"error": "document_too_large", "index": i,
                                                             "max_chars": settings.max_chars})

    def check_strategy(strategy: Optional[str]) -> None:
        if strategy is None:
            return
        if strategy not in STRATEGIES:
            raise HTTPException(status_code=422, detail={"error": "invalid_strategy", "allowed": list(STRATEGIES)})
        if strategy == "pseudonym" and not service.redactor.hmac_key:
            raise HTTPException(status_code=422, detail={"error": "pseudonym_not_configured"})

    def audit_event(request: Request, endpoint: str, results, latency_s: float, status: int = 200) -> None:
        actions = CCounter(r.action for r in results)
        cats = CCounter()
        for r in results:
            cats.update(r.category_counts)
        for a, n in actions.items():
            metrics.documents.inc(n, action=a)
        for c, n in cats.items():
            metrics.findings.inc(n, category=c)
        audit.info(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request.state.request_id, "key_id": request.state.key_id, "endpoint": endpoint,
            "status": status, "documents": len(results), "actions": dict(actions), "categories": dict(cats),
            "latency_ms": round(latency_s * 1000, 1), "model_version": service.model_version,
        }, sort_keys=True))

    # ------------------------------------------------------------------ endpoints
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if not service.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "model_version": service.model_version}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint(_: str = Depends(authenticate)):
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v2/info")
    def info(_: str = Depends(authenticate)):
        d = service.describe()
        d.update({"api_version": API_VERSION,
                  "limits": {"max_chars": settings.max_chars, "max_batch": settings.max_batch,
                             "max_body_bytes": settings.max_body_bytes,
                             "rate_limit_per_min": settings.rate_limit_per_min}})
        return d

    @app.post("/v2/detect")
    async def detect(req: DetectRequest, request: Request, _: str = Depends(rate_limited(1.0))):
        check_texts([req.text])
        t0 = time.perf_counter()
        result = (await run_inference(lambda: service.redactor.redact_many([req.text])))[0]
        audit_event(request, "/v2/detect", [result], time.perf_counter() - t0)
        d = result.to_dict(include_finding_text=settings.return_finding_text)
        d.pop("redacted_text")
        d["model_version"] = service.model_version
        return d

    @app.post("/v2/redact")
    async def redact(req: RedactRequest, request: Request, _: str = Depends(rate_limited(1.0))):
        check_texts([req.text])
        check_strategy(req.strategy)
        t0 = time.perf_counter()
        result = (await run_inference(lambda: service.redactor.redact_many([req.text], req.strategy)))[0]
        audit_event(request, "/v2/redact", [result], time.perf_counter() - t0)
        d = result.to_dict(include_finding_text=settings.return_finding_text)
        d["model_version"] = service.model_version
        return d

    @app.post("/v2/redact/batch")
    async def redact_batch(req: BatchRedactRequest, request: Request,
                           key_id: str = Depends(authenticate)):
        check_texts(req.texts)
        check_strategy(req.strategy)
        wait = limiter.allow(key_id, float(len(req.texts)))
        if wait is not None:
            metrics.rejected.inc(reason="rate_limited")
            raise HTTPException(status_code=429, detail={"error": "rate_limited"},
                                headers={"Retry-After": str(max(1, int(wait + 0.999)))})
        t0 = time.perf_counter()
        results = await run_inference(lambda: service.redactor.redact_many(req.texts, req.strategy))
        audit_event(request, "/v2/redact/batch", results, time.perf_counter() - t0)
        return {"model_version": service.model_version,
                "results": [r.to_dict(include_finding_text=settings.return_finding_text) for r in results]}

    return app
