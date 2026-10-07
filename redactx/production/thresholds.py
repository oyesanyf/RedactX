"""
Operating thresholds.

A redaction system is judged by its misses, so thresholds are chosen for a *target recall* on held-out
data rather than fixed at 0.5. `calibrate_thresholds.py` measures scores on held-out documents, picks the
thresholds here, and writes them to `<model_dir>/redactx_thresholds.json`, which the engine loads.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

THRESHOLDS_FILE = "redactx_thresholds.json"


def threshold_for_recall(positive_scores: Sequence[float], target_recall: float) -> float:
    """
    Largest threshold t such that the fraction of positives with score >= t is at least target_recall.
    """
    if not positive_scores:
        raise ValueError("need at least one positive score")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    ordered = sorted(positive_scores, reverse=True)
    k = math.ceil(target_recall * len(ordered))
    return float(ordered[k - 1])


def operating_point(positive_scores: Sequence[float], negative_scores: Sequence[float],
                    threshold: float) -> Dict[str, Optional[float]]:
    tp = sum(1 for s in positive_scores if s >= threshold)
    fn = len(positive_scores) - tp
    fp = sum(1 for s in negative_scores if s >= threshold)
    tn = len(negative_scores) - fp
    return {
        "threshold": round(float(threshold), 6),
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "n_pos": len(positive_scores),
        "n_neg": len(negative_scores),
    }


@dataclass
class ThresholdConfig:
    doc_threshold: float = 0.5
    span_threshold: float = 0.5
    target_recall: Optional[float] = None
    calibrated_on: str = "default (not calibrated)"
    doc_operating_point: Dict[str, Any] = field(default_factory=dict)
    span_operating_point: Dict[str, Any] = field(default_factory=dict)

    def save(self, model_dir: str) -> str:
        path = os.path.join(model_dir, THRESHOLDS_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, model_dir: str) -> "ThresholdConfig":
        path = os.path.join(model_dir, THRESHOLDS_FILE)
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        for name in ("doc_threshold", "span_threshold"):
            v = getattr(cfg, name)
            if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
                raise ValueError(f"{path}: {name} must be a number in [0, 1], got {v!r}")
        return cfg
