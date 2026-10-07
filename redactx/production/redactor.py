"""
Redaction: turns a Detection into redacted text.

Strategies
  tag        "John Smith" -> "[NAME]"
  mask       "John Smith" -> "[REDACTED]"
  char       "John Smith" -> "**********"            (length-preserving)
  pseudonym  "John Smith" -> "[NAME_3f9a1c2b]"        (keyed HMAC-SHA256: the same value always maps to the
                                                      same token under one key, so records stay linkable
                                                      without exposing the value; requires a secret key)

Fail-closed policy for PHI the model detects but cannot localize (`unlocalized_policy`)
  redact_all  replace the whole document with "[REDACTED_DOCUMENT]"   (default)
  review      redact every localized finding (e.g. Presidio's) and return action="REVIEW" so a human
              checks the document for the identifier RedactX detected but could not place
"""

import hashlib
import hmac
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from redactx.production.detectors import Detection, Finding

STRATEGIES = ("tag", "mask", "char", "pseudonym")
UNLOCALIZED_POLICIES = ("redact_all", "review")
REDACTED_DOCUMENT = "[REDACTED_DOCUMENT]"


@dataclass
class RedactionResult:
    redacted_text: str
    action: str                       # PASS | REDACTED | REDACTED_DOCUMENT | REVIEW
    contains_phi: bool
    findings: List[Finding]
    doc_score: Optional[float]
    detectors: List[str]
    windows: int
    latency_ms: float
    category_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self, include_finding_text: bool = False) -> Dict:
        return {
            "redacted_text": self.redacted_text,
            "action": self.action,
            "contains_phi": self.contains_phi,
            "doc_score": self.doc_score,
            "findings": [f.to_dict(include_text=include_finding_text) for f in self.findings],
            "category_counts": self.category_counts,
            "detectors": self.detectors,
            "windows": self.windows,
            "latency_ms": round(self.latency_ms, 2),
        }


class Redactor:
    def __init__(self, detector, strategy: str = "tag", unlocalized_policy: str = "redact_all",
                 hmac_key: Optional[bytes] = None, hipaa_only: bool = False, char: str = "*"):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if unlocalized_policy not in UNLOCALIZED_POLICIES:
            raise ValueError(f"unlocalized_policy must be one of {UNLOCALIZED_POLICIES}")
        if strategy == "pseudonym" and not hmac_key:
            raise ValueError("strategy 'pseudonym' requires a secret hmac_key")
        if hmac_key is not None and len(hmac_key) < 16:
            raise ValueError("hmac_key must be at least 16 bytes")
        self.detector = detector
        self.strategy = strategy
        self.unlocalized_policy = unlocalized_policy
        self.hmac_key = hmac_key
        self.hipaa_only = hipaa_only
        self.char = char

    # -------------------------------------------------------------- replacement
    def _pseudonym(self, f: Finding) -> str:
        norm = unicodedata.normalize("NFKC", f.text).strip().casefold()
        digest = hmac.new(self.hmac_key, f"{f.category.value}\x1f{norm}".encode("utf-8"), hashlib.sha256)
        return f"[{f.category.value}_{digest.hexdigest()[:8]}]"

    def replacement(self, f: Finding, strategy: Optional[str] = None) -> str:
        s = strategy or self.strategy
        if s == "tag":
            return f"[{f.category.value}]"
        if s == "mask":
            return "[REDACTED]"
        if s == "char":
            return "".join(ch if ch.isspace() else self.char for ch in f.text)
        return self._pseudonym(f)

    def apply(self, text: str, detection: Detection, latency_ms: float = 0.0,
              strategy: Optional[str] = None) -> RedactionResult:
        findings = [f for f in detection.findings if (f.is_hipaa_identifier or not self.hipaa_only)]
        counts = dict(Counter(f.category.value for f in findings))
        base = dict(contains_phi=detection.contains_phi, findings=findings, doc_score=detection.doc_score,
                    detectors=detection.detectors, windows=detection.windows, latency_ms=latency_ms,
                    category_counts=counts)
        if detection.unlocalized_phi and self.unlocalized_policy == "redact_all":
            return RedactionResult(redacted_text=REDACTED_DOCUMENT, action="REDACTED_DOCUMENT", **base)
        if not findings:
            action = "REVIEW" if detection.unlocalized_phi else "PASS"
            return RedactionResult(redacted_text=text, action=action, **base)
        parts, cursor = [], 0
        for f in sorted(findings, key=lambda x: x.start):
            if f.start < cursor:          # defensive: findings are already merged/non-overlapping
                continue
            parts.append(text[cursor:f.start])
            parts.append(self.replacement(f, strategy))
            cursor = f.end
        parts.append(text[cursor:])
        # Localized findings are always redacted; an unlocalized RedactX flag additionally requires human review.
        action = "REVIEW" if detection.unlocalized_phi else "REDACTED"
        return RedactionResult(redacted_text="".join(parts), action=action, **base)

    # -------------------------------------------------------------- end-to-end
    def redact_many(self, texts: Sequence[str], strategy: Optional[str] = None) -> List[RedactionResult]:
        t0 = time.perf_counter()
        detections = self.detector.detect_many(list(texts))
        per_doc_ms = (time.perf_counter() - t0) * 1000.0 / max(len(texts), 1)
        return [self.apply(t, d, per_doc_ms, strategy) for t, d in zip(texts, detections)]

    def redact(self, text: str, strategy: Optional[str] = None) -> RedactionResult:
        return self.redact_many([text], strategy)[0]
