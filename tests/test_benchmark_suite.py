"""
Tests for RedactX Comparative Benchmark Suite:
IoU calculation, Adjusted Recall, Exact Precision/Recall, and comparative reporting.
"""

import pytest
from redactx.evaluation.benchmark_suite import (
    BenchmarkSuite,
    compute_token_overlap_iou,
    PresidioBaselineSimulator
)
from redactx.primitives import DetectedSpan


def test_compute_token_overlap_iou():
    # Exact match
    assert compute_token_overlap_iou((10, 20), (10, 20)) == 1.0

    # No overlap
    assert compute_token_overlap_iou((10, 20), (30, 40)) == 0.0

    # Partial overlap: [10, 20] and [15, 25] -> intersection = 5, union = 15 -> 5/15 = 0.333
    iou = compute_token_overlap_iou((10, 20), (15, 25))
    assert abs(iou - 0.3333) < 0.01


def test_adjusted_recall_logic():
    suite = BenchmarkSuite()

    # Ground truth: 2 spans (One Name, One Date)
    ground_truth = [
        {"start": 10, "end": 22, "category": "PATIENT_NAME", "text": "Elena Watson"},
        {"start": 40, "end": 50, "category": "DATE", "text": "04/12/1981"}
    ]

    # Predicted spans:
    # 1. Elena Watson is predicted as generic "PHI" (misclassified category, but caught!)
    # 2. 04/12/1981 is predicted with exact category "DATE"
    predicted = [
        DetectedSpan(text="Elena Watson", start=10, end=22, category="PHI", confidence=0.90),
        DetectedSpan(text="04/12/1981", start=40, end=50, category="DATE", confidence=0.95)
    ]

    exact_tp, adj_tp, fp = suite.evaluate_spans(predicted, ground_truth)

    # Both are caught under adjusted recall (100% adjusted recall)
    assert adj_tp == 2
    # Date is exact match, Name is category overlap
    assert exact_tp >= 1
    assert fp == 0


def test_presidio_simulator():
    sim = PresidioBaselineSimulator()
    text = "Contact Dr. Smith at 555-123-4567 or email john.doe@example.com with SSN 123-45-6789 on 05/20/2026."
    spans = sim.analyze(text)

    categories = {s.category for s in spans}
    assert "SSN" in categories
    assert "EMAIL" in categories
    assert "PHONE" in categories
    assert "DATE" in categories
    assert len(spans) >= 4
