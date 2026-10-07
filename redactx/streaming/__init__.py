"""
RedactX Streaming Engine: Unlimited-length document redaction and decision processing.
Bypasses fixed-length parser memory limits (such as spaCy's 1M character constraint)
via overlapping sliding-window chunking, sub-50ms line-rate inference, and seamless span deduplication.
"""

from redactx.streaming.streaming_engine import StreamingRedactXEngine, StreamingDecisionResult

__all__ = ["StreamingRedactXEngine", "StreamingDecisionResult"]
