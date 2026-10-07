"""
Production settings, read from the process environment (and optionally a local .env file, which is only
READ — nothing is ever written to the user/system environment). Every value is validated at startup so a
misconfigured server refuses to start instead of running unsafely.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

MODES = ("hybrid", "redactx", "presidio")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def hash_api_key(key: str) -> str:
    """SHA-256 hex digest used to store API keys (the server never stores plaintext keys)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def read_dotenv(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    model_dir: Optional[str] = None
    mode: str = "hybrid"
    device: Optional[str] = None
    strategy: str = "tag"
    unlocalized_policy: str = "redact_all"
    hmac_key: Optional[bytes] = None
    api_key_hashes: Tuple[str, ...] = ()
    allow_no_auth: bool = False
    max_chars: int = 100_000
    max_batch: int = 32
    max_body_bytes: int = 4_000_000
    rate_limit_per_min: int = 120
    rate_burst: int = 30
    request_timeout_s: float = 60.0
    max_concurrency: int = 1
    max_queue: int = 16
    cors_origins: Tuple[str, ...] = ()
    audit_log: str = "-"
    presidio_score: float = 0.35
    chunk_chars: int = 600
    chunk_overlap: int = 150
    batch_size: int = 8
    hipaa_only: bool = False
    return_finding_text: bool = False
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, dotenv_path: Optional[str] = ".env") -> "Settings":
        merged: Dict[str, str] = dict(read_dotenv(dotenv_path)) if dotenv_path else {}
        merged.update(dict(os.environ if env is None else env))
        g = lambda k, d=None: merged.get(f"REDACTX_{k}", d)  # noqa: E731

        hmac_raw = g("HMAC_KEY")
        hmac_key = None
        if hmac_raw:
            hmac_key = bytes.fromhex(hmac_raw) if re.fullmatch(r"[0-9a-fA-F]{32,}", hmac_raw) else hmac_raw.encode()

        s = cls(
            model_dir=g("MODEL_DIR"),
            mode=(g("MODE", "hybrid") or "hybrid").lower(),
            device=g("DEVICE") or None,
            strategy=(g("STRATEGY", "tag") or "tag").lower(),
            unlocalized_policy=(g("UNLOCALIZED_POLICY", "redact_all") or "redact_all").lower(),
            hmac_key=hmac_key,
            api_key_hashes=tuple(h.strip().lower() for h in (g("API_KEY_HASHES", "") or "").split(",") if h.strip()),
            allow_no_auth=_bool(g("ALLOW_NO_AUTH", "0")),
            max_chars=int(g("MAX_CHARS", 100_000)),
            max_batch=int(g("MAX_BATCH", 32)),
            max_body_bytes=int(g("MAX_BODY_BYTES", 4_000_000)),
            rate_limit_per_min=int(g("RATE_LIMIT_PER_MIN", 120)),
            rate_burst=int(g("RATE_BURST", 30)),
            request_timeout_s=float(g("REQUEST_TIMEOUT_S", 60)),
            max_concurrency=int(g("MAX_CONCURRENCY", 1)),
            max_queue=int(g("MAX_QUEUE", 16)),
            cors_origins=tuple(o.strip() for o in (g("CORS_ORIGINS", "") or "").split(",") if o.strip()),
            audit_log=g("AUDIT_LOG", "-") or "-",
            presidio_score=float(g("PRESIDIO_SCORE", 0.35)),
            chunk_chars=int(g("CHUNK_CHARS", 600)),
            chunk_overlap=int(g("CHUNK_OVERLAP", 150)),
            batch_size=int(g("BATCH_SIZE", 8)),
            hipaa_only=_bool(g("HIPAA_ONLY", "0")),
            return_finding_text=_bool(g("RETURN_FINDING_TEXT", "0")),
        )
        s.validate()
        return s

    def validate(self) -> None:
        errors: List[str] = []
        if self.mode not in MODES:
            errors.append(f"REDACTX_MODE must be one of {MODES}")
        if self.mode in ("hybrid", "redactx"):
            if not self.model_dir:
                errors.append("REDACTX_MODEL_DIR is required for mode 'hybrid' or 'redactx'")
            elif not os.path.isdir(self.model_dir):
                errors.append(f"REDACTX_MODEL_DIR does not exist: {self.model_dir}")
            elif not os.path.isfile(os.path.join(self.model_dir, "config.json")):
                # The engine loader would otherwise fall back to another directory or an untrained base model.
                errors.append(f"REDACTX_MODEL_DIR is not an exported RedactX checkpoint (no config.json): "
                              f"{self.model_dir}")
        from redactx.production.redactor import STRATEGIES, UNLOCALIZED_POLICIES
        if self.strategy not in STRATEGIES:
            errors.append(f"REDACTX_STRATEGY must be one of {STRATEGIES}")
        if self.unlocalized_policy not in UNLOCALIZED_POLICIES:
            errors.append(f"REDACTX_UNLOCALIZED_POLICY must be one of {UNLOCALIZED_POLICIES}")
        if self.strategy == "pseudonym" and not self.hmac_key:
            errors.append("REDACTX_STRATEGY=pseudonym requires REDACTX_HMAC_KEY")
        if self.hmac_key is not None and len(self.hmac_key) < 16:
            errors.append("REDACTX_HMAC_KEY must be at least 16 bytes (or 32+ hex characters)")
        bad = [h for h in self.api_key_hashes if not _HEX64.match(h)]
        if bad:
            errors.append("REDACTX_API_KEY_HASHES must be SHA-256 hex digests (use `redactx hash-key`)")
        if not self.api_key_hashes and not self.allow_no_auth:
            errors.append("No API keys configured: set REDACTX_API_KEY_HASHES "
                          "(or REDACTX_ALLOW_NO_AUTH=1 for local development only)")
        for name, v, lo in [("MAX_CHARS", self.max_chars, 1), ("MAX_BATCH", self.max_batch, 1),
                            ("MAX_BODY_BYTES", self.max_body_bytes, 1024), ("RATE_LIMIT_PER_MIN", self.rate_limit_per_min, 1),
                            ("RATE_BURST", self.rate_burst, 1), ("MAX_CONCURRENCY", self.max_concurrency, 1),
                            ("MAX_QUEUE", self.max_queue, 0), ("BATCH_SIZE", self.batch_size, 1),
                            ("CHUNK_CHARS", self.chunk_chars, 50)]:
            if v < lo:
                errors.append(f"REDACTX_{name} must be >= {lo}")
        if not 0 <= self.chunk_overlap <= self.chunk_chars // 2:
            errors.append("REDACTX_CHUNK_OVERLAP must be in [0, CHUNK_CHARS/2]")
        if self.request_timeout_s <= 0:
            errors.append("REDACTX_REQUEST_TIMEOUT_S must be > 0")
        if not 0.0 <= self.presidio_score <= 1.0:
            errors.append("REDACTX_PRESIDIO_SCORE must be in [0, 1]")
        if "*" in self.cors_origins:
            errors.append("REDACTX_CORS_ORIGINS must list explicit origins ('*' is not allowed)")
        if errors:
            raise ValueError("Invalid RedactX settings:\n  - " + "\n  - ".join(errors))
        if self.allow_no_auth:
            self.warnings.append("Authentication is DISABLED (REDACTX_ALLOW_NO_AUTH=1). Development only.")
        if self.return_finding_text:
            self.warnings.append("REDACTX_RETURN_FINDING_TEXT=1: API responses include the original PHI values.")
