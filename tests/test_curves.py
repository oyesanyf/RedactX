"""
Unit tests for Precision-Recall (PR) and ROC curve computation, AUPRC, AUROC,
Brier score, 10-bin ECE, Pareto frontier, and dual operating mode evaluation.
"""

import math
import pytest
from redactx.evaluation.curves import (
    compute_auprc,
    compute_auroc,
    compute_brier_score,
    compute_ece_10bin,
    compute_pareto_frontier,
    compute_precision_recall_curve,
    compute_roc_curve,
    evaluate_dual_operating_modes,
    wilson_lower_bound
)


def test_wilson_lower_bound_math():
    # 0 successes out of 100
    lb_0 = wilson_lower_bound(0, 100, confidence=0.95)
    assert lb_0 == 0.0

    # 100 successes out of 100
    lb_100 = wilson_lower_bound(100, 100, confidence=0.95)
    assert 0.95 < lb_100 < 1.0

    # 99 out of 100
    lb_99 = wilson_lower_bound(99, 100, confidence=0.95)
    assert 0.94 < lb_99 < 0.99

    # Errors on invalid inputs
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 0)
    with pytest.raises(ValueError):
        wilson_lower_bound(-1, 10)
    with pytest.raises(ValueError):
        wilson_lower_bound(11, 10)
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 10, confidence=1.5)
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 10, confidence=-0.1)


def test_curve_length_validations():
    with pytest.raises(ValueError):
        compute_auprc([0.1, 0.5], [0.9])
    with pytest.raises(ValueError):
        compute_auroc([0.1, 0.5], [0.9])
    with pytest.raises(ValueError):
        compute_pareto_frontier([0.5], [0.9], [0.1, 0.2])


def test_brier_score():
    y_true = [1, 0, 1, 0]
    # Perfect probabilities
    assert compute_brier_score(y_true, [1.0, 0.0, 1.0, 0.0]) == 0.0
    # Worst probabilities
    assert compute_brier_score(y_true, [0.0, 1.0, 0.0, 1.0]) == 1.0
    # Constant 0.5
    assert compute_brier_score(y_true, [0.5, 0.5, 0.5, 0.5]) == 0.25

    # Length mismatch
    with pytest.raises(ValueError):
        compute_brier_score([1, 0], [0.5])


def test_ece_10bin():
    # Perfectly calibrated predictions
    y_true = [0] * 50 + [1] * 50
    y_prob = [0.05] * 50 + [0.95] * 50
    ece = compute_ece_10bin(y_true, y_prob, num_bins=10)
    assert round(ece, 3) <= 0.06

    # Empty inputs
    assert compute_ece_10bin([], []) == 0.0

    # Length mismatch
    with pytest.raises(ValueError):
        compute_ece_10bin([1], [0.5, 0.5])


def test_precision_recall_and_auprc_perfect_separation():
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    precs, recs, threshs = compute_precision_recall_curve(y_true, y_score)
    assert len(precs) == len(recs) == len(threshs)
    # Since separation is perfect, precision should be 1.0 for all recalls
    auprc = compute_auprc(recs, precs)
    assert auprc >= 0.95


def test_roc_and_auroc_perfect_separation():
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    fprs, tprs, threshs = compute_roc_curve(y_true, y_score)
    assert fprs[0] == 0.0 and tprs[0] == 0.0
    auroc = compute_auroc(fprs, tprs)
    assert auroc == 1.0


def test_roc_random_chance():
    fprs = [0.0, 0.5, 1.0]
    tprs = [0.0, 0.5, 1.0]
    auroc = compute_auroc(fprs, tprs)
    assert math.isclose(auroc, 0.5, abs_tol=1e-5)


def test_pareto_frontier():
    recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
    precisions = [1.0, 0.95, 0.90, 0.85, 0.70]
    threshs = [0.9, 0.8, 0.7, 0.6, 0.5]

    pareto = compute_pareto_frontier(recalls, precisions, threshs)
    assert len(pareto) > 0
    # Every point on this synthetic frontier is non-dominated
    assert len(pareto) == 5
    assert pareto[-1]["recall"] == 1.0


def test_evaluate_dual_operating_modes():
    # 50 positives, 50 negatives
    y_true = [1] * 50 + [0] * 50
    # High scores for positives, low scores for negatives
    y_score = [0.95] * 45 + [0.85] * 4 + [0.05] * 1 + [0.02] * 45 + [0.15] * 4 + [0.90] * 1

    curve_data = evaluate_dual_operating_modes(
        y_true, y_score, target_recall=0.95, confidence=0.95, mode_b_threshold=0.50
    )

    assert curve_data.n_samples == 100
    assert curve_data.n_positives == 50
    assert curve_data.n_negatives == 50
    assert curve_data.mode_a.mode == "zero_leakage"
    assert curve_data.mode_b.mode == "balanced_utility"

    # Mode A must have high recall
    assert curve_data.mode_a.recall >= 0.95
    # Mode B must have high specificity
    assert curve_data.mode_b.specificity >= 0.90
    assert curve_data.auprc > 0.80
    assert curve_data.auroc > 0.80
    assert 0.0 <= curve_data.brier_score <= 1.0
    assert 0.0 <= curve_data.ece_10bin <= 1.0
