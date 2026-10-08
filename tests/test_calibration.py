"""
Unit test for Platt temperature calibration optimization.
"""

import pytest
import torch
from redactx.models.scorer import PlattTemperatureScaler


def test_temperature_calibration_convergence():
    scaler = PlattTemperatureScaler(initial_temperature=1.0)

    # Simulate overconfident logits
    torch.manual_seed(42)
    val_labels = torch.randint(0, 2, (100,))
    # Overconfident scale
    val_logits = torch.randn(100, 2) * 5.0

    optimal_t = scaler.fit(val_logits, val_labels, lr=0.1, max_iter=25)
    assert optimal_t > 0.0
    assert isinstance(optimal_t, float)


def test_wilson_lower_bound_properties():
    from redactx.production.thresholds import wilson_lower_bound

    # Higher confidence implies lower (more conservative) lower bound
    lb_90 = wilson_lower_bound(95, 100, confidence=0.90)
    lb_99 = wilson_lower_bound(95, 100, confidence=0.99)
    assert lb_99 < lb_90

    # Bounds must be strictly in [0, 1]
    assert 0.0 <= wilson_lower_bound(0, 50) <= 1.0
    assert 0.0 <= wilson_lower_bound(50, 50) <= 1.0

    # Invalid successes bounds
    with pytest.raises(ValueError):
        wilson_lower_bound(-1, 50)
    with pytest.raises(ValueError):
        wilson_lower_bound(51, 50)


def test_conservative_threshold_selection():
    from redactx.production.thresholds import conservative_threshold_for_recall, threshold_for_recall

    # 100 scores from 0.01 to 1.00
    pos_scores = [i / 100.0 for i in range(1, 101)]

    # Point estimate threshold
    point_t = threshold_for_recall(pos_scores, target_recall=0.95)
    assert point_t == 0.06

    # Conservative threshold (Wilson 95% lower bound)
    cons_t, lb, ok = conservative_threshold_for_recall(pos_scores, target_recall=0.95, confidence=0.95)
    # Conservative threshold must be <= point estimate (more cautious, lower threshold flags more)
    assert cons_t <= point_t
    assert ok is True
    assert lb >= 0.95


def test_operating_point_metrics():
    from redactx.production.thresholds import operating_point

    pos = [0.9, 0.8, 0.7, 0.6]
    neg = [0.4, 0.3, 0.2, 0.1]

    pt = operating_point(pos, neg, threshold=0.5, confidence=0.95)
    assert pt["recall"] == 1.0
    assert pt["specificity"] == 1.0
    assert pt["precision"] == 1.0
    assert pt["n_pos"] == 4
    assert pt["n_neg"] == 4
    assert pt["recall_lower_bound"] is not None

