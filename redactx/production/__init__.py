"""
RedactX production layer: long-document chunking, hybrid detection (RedactX + Presidio), redaction
strategies, calibrated thresholds, and a hardened API server.

Submodules are imported lazily by callers (this file must stay light: the model engine imports
`redactx.production.thresholds`, so importing heavy modules here would create import cycles).
"""

__all__ = ["chunking", "detectors", "hipaa", "metrics", "redactor", "server", "settings", "thresholds"]
