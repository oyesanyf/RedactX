"""
RedactX Evaluation and Comparative Benchmarking Suite.
Computes Standard Precision, Standard Recall, Adjusted Recall (Lenient Overlap),
F1 Score, Latency, and comparative analysis against Presidio and rule-based baselines.
"""

from redactx.evaluation.benchmark_suite import (
    BenchmarkSuite,
    BenchmarkMetrics,
    ComparativeBenchmarkResult,
    compute_token_overlap_iou
)

__all__ = [
    "BenchmarkSuite",
    "BenchmarkMetrics",
    "ComparativeBenchmarkResult",
    "compute_token_overlap_iou"
]
