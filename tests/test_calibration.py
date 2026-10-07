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
