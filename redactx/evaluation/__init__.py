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
from redactx.evaluation.curves import (
    CurveData,
    OperatingPointMetrics,
    compute_auprc,
    compute_auroc,
    compute_brier_score,
    compute_ece_10bin,
    compute_pareto_frontier,
    compute_precision_recall_curve,
    compute_roc_curve,
    evaluate_dual_operating_modes,
    export_curve_json,
    wilson_lower_bound
)

__all__ = [
    "BenchmarkSuite",
    "BenchmarkMetrics",
    "ComparativeBenchmarkResult",
    "compute_token_overlap_iou",
    "CurveData",
    "OperatingPointMetrics",
    "compute_auprc",
    "compute_auroc",
    "compute_brier_score",
    "compute_ece_10bin",
    "compute_pareto_frontier",
    "compute_precision_recall_curve",
    "compute_roc_curve",
    "evaluate_dual_operating_modes",
    "export_curve_json",
    "wilson_lower_bound"
]

