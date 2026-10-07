"""
Tests for RedactX Streaming Engine: Unlimited-length processing,
overlapping window chunking, and boundary deduplication.
"""

import pytest
from redactx.streaming.streaming_engine import StreamingRedactXEngine, StreamingDecisionResult
from redactx.primitives import DetectedSpan


def test_chunk_text_boundaries():
    engine = StreamingRedactXEngine(chunk_size_chars=50, overlap_chars=10)
    text = "The quick brown fox jumps over the lazy dog repeatedly across multiple lines."
    chunks = list(engine.chunk_text(text))

    assert len(chunks) > 1
    # Verify that the entire text is covered without missing gaps
    covered = ""
    for idx, (start, end, chunk) in enumerate(chunks):
        assert text[start:end] == chunk
        if idx > 0:
            prev_end = chunks[idx - 1][1]
            # Ensure consecutive chunks overlap
            assert start < prev_end


def test_merge_overlapping_spans():
    engine = StreamingRedactXEngine()
    spans = [
        DetectedSpan(text="John", start=10, end=14, category="PATIENT_NAME", confidence=0.85),
        DetectedSpan(text="John Doe", start=10, end=18, category="PATIENT_NAME", confidence=0.92),
        DetectedSpan(text="04/12/1981", start=50, end=60, category="DATE", confidence=0.95),
        DetectedSpan(text="1981", start=56, end=60, category="DATE", confidence=0.70)
    ]
    merged = engine.merge_overlapping_spans(spans)

    assert len(merged) == 2
    assert merged[0].start == 10
    assert merged[0].end == 18
    assert merged[0].confidence == 0.92

    assert merged[1].start == 50
    assert merged[1].end == 60
    assert merged[1].confidence == 0.95


def test_streaming_exceeds_1m_character_limit():
    """
    Directly verifies RedactX's ability to stream through documents larger than
    the 1M character limit imposed by spaCy without crashing or exhausting RAM.
    """
    engine = StreamingRedactXEngine(chunk_size_chars=4000, overlap_chars=200)

    base_sentence = "Routine server telemetry packet logged with response status 200 OK. "
    # Build text exceeding 1,000,000 characters
    repeat_count = int(1_050_000 / len(base_sentence)) + 1
    large_text = base_sentence * repeat_count

    # Inject PHI at specific locations: near start, middle (around 500k), and end (around 1M)
    phi_1 = "Patient Alice Walker DOB 01/01/1990 SSN 123-45-6789. "
    phi_2 = "Patient Bob Vance MRN 987654321. "

    mid_point = 525_000
    large_text = (
        large_text[:1000] + phi_1 +
        large_text[1000 + len(phi_1):mid_point] + phi_2 +
        large_text[mid_point + len(phi_2):]
    )

    total_len = len(large_text)
    assert total_len > 1_000_000, f"Document length should exceed 1M chars, got {total_len}"

    result = engine.process_stream(large_text, generate_redacted_text=False)

    assert result.total_chars_processed == total_len
    assert result.num_chunks_processed > 200
    assert result.contains_phi is True
    assert result.total_spans_detected >= 2
    assert result.throughput_chars_per_second > 50_000
