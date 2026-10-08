"""
Unit tests for Dual Operating Modes (Mode A: Zero-Leakage Compliance Mode vs Mode B: Balanced Utility Mode).
"""

import json
import os
import pytest
from redactx.production.thresholds import ThresholdConfig


def test_threshold_config_mode_switching(tmp_path):
    cfg = ThresholdConfig(
        doc_threshold=0.009281,
        span_threshold=0.003928,
        target_recall=0.98,
        confidence=0.95,
        calibrated_on="test-held-out"
    )

    # Initial state should default to zero_leakage mode
    assert cfg.operating_mode == "zero_leakage"
    assert cfg.doc_threshold == 0.009281
    assert cfg.span_threshold == 0.003928

    # Switch to balanced utility mode
    cfg.set_operating_mode("balanced_utility")
    assert cfg.operating_mode == "balanced_utility"
    assert cfg.doc_threshold == 0.50
    assert cfg.span_threshold == 0.50

    # Switch back using alias "compliance"
    cfg.set_operating_mode("compliance")
    assert cfg.operating_mode == "zero_leakage"
    assert cfg.doc_threshold == 0.009281
    assert cfg.span_threshold == 0.003928

    # Switch using alias "utility"
    cfg.set_operating_mode("utility")
    assert cfg.operating_mode == "balanced_utility"
    assert cfg.doc_threshold == 0.50

    # Invalid mode should raise ValueError
    with pytest.raises(ValueError):
        cfg.set_operating_mode("invalid_mode")


def test_threshold_config_save_load(tmp_path):
    save_dir = str(tmp_path / "model_chk")
    os.makedirs(save_dir, exist_ok=True)

    cfg = ThresholdConfig(
        doc_threshold=0.0123,
        span_threshold=0.0045,
        target_recall=0.98,
        confidence=0.95
    )
    cfg.save(save_dir)

    loaded = ThresholdConfig.load(save_dir)
    assert loaded.doc_threshold == 0.0123
    assert loaded.span_threshold == 0.0045
    assert "zero_leakage" in loaded.modes
    assert "balanced_utility" in loaded.modes

    # Switch loaded config
    loaded.set_operating_mode("balanced_utility")
    assert loaded.doc_threshold == 0.50
    assert loaded.span_threshold == 0.50


def test_get_mode_thresholds_without_mutation():
    cfg = ThresholdConfig(
        doc_threshold=0.009281,
        span_threshold=0.003928
    )
    doc_b, span_b = cfg.get_mode_thresholds("balanced_utility")
    assert doc_b == 0.50 and span_b == 0.50
    # Original state unchanged
    assert cfg.doc_threshold == 0.009281
    assert cfg.operating_mode == "zero_leakage"
