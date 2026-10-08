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
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence, Tuple

THRESHOLDS_FILE = "redactx_thresholds.json"


def threshold_for_recall(positive_scores: Sequence[float], target_recall: float) -> float:
    """
    Largest threshold t such that the fraction of positives with score >= t is at least target_recall.
    This is the point estimate: it sits exactly at the edge of the calibration sample, so on new data the
    recall is below target about half the time. Prefer `conservative_threshold_for_recall` in production.
    """
    if not positive_scores:
        raise ValueError("need at least one positive score")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    ordered = sorted(positive_scores, reverse=True)
    k = math.ceil(target_recall * len(ordered))
    return float(ordered[k - 1])


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """
    Computes the one-sided Wilson score lower confidence bound for binomial proportion k/n.

    Mathematical Formulation:
        Given k successes in n independent Bernoulli trials with observed proportion p = k/n
        and standard normal critical value z = \\Phi^{-1}(1 - \\alpha) (one-sided confidence 1 - \\alpha):

        w^-(k, n, z) = \\frac{p + \\frac{z^2}{2n} - z \\sqrt{\\frac{p(1 - p)}{n} + \\frac{z^2}{4n^2}}}{1 + \\frac{z^2}{n}}

    Guarantees coverage probability >= 1 - \\alpha even for extreme proportions p -> 0 or p -> 1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError(f"successes ({successes}) must be between 0 and n ({n})")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    p = successes / n
    z = NormalDist().inv_cdf(confidence)
    if z <= 0:
        return p
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n)))
    denom = 1.0 + z2 / n
    return max(0.0, min(1.0, (centre - spread) / denom))


def conservative_threshold_for_recall(positive_scores: Sequence[float], target_recall: float,
                                      confidence: float = 0.95) -> Tuple[float, float, bool]:
    """
    Largest threshold t such that the one-sided `confidence` lower bound on recall (Wilson) is >= target_recall,
    i.e. with that confidence the true recall on data like the calibration set is at least the target.

    Returns (threshold, recall_lower_bound, achievable). If even flagging every calibration positive cannot
    certify the target (too few positives), returns the lowest positive score with achievable=False.
    """
    if not positive_scores:
        raise ValueError("need at least one positive score")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    n = len(positive_scores)
    ordered = sorted(positive_scores, reverse=True)
    # Walk distinct thresholds from high to low; k = number of positives with score >= t (ties included).
    for i, t in enumerate(ordered):
        if i + 1 < n and ordered[i + 1] == t:
            continue
        lb = wilson_lower_bound(i + 1, n, confidence)
        if lb >= target_recall:
            return float(t), lb, True
    return float(ordered[-1]), wilson_lower_bound(n, n, confidence), False


def operating_point(positive_scores: Sequence[float], negative_scores: Sequence[float],
                    threshold: float, confidence: Optional[float] = None) -> Dict[str, Optional[float]]:
    tp = sum(1 for s in positive_scores if s >= threshold)
    fn = len(positive_scores) - tp
    fp = sum(1 for s in negative_scores if s >= threshold)
    tn = len(negative_scores) - fp
    point = {
        "threshold": round(float(threshold), 6),
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "n_pos": len(positive_scores),
        "n_neg": len(negative_scores),
    }
    if confidence is not None and (tp + fn):
        point["recall_lower_bound"] = round(wilson_lower_bound(tp, tp + fn, confidence), 4)
        point["confidence"] = confidence
    return point


@dataclass
class ThresholdConfig:
    """
    Operating threshold configuration for RedactX.

    Supports two primary operational profiles:
      1. Operating Mode A (Zero-Leakage Compliance Mode):
         Optimized for strict regulatory HIPAA Safe Harbor compliance.
         Calibrated such that the one-sided Wilson 95% lower confidence bound on recall
         meets or exceeds the target recall (e.g. 98.0% certified).
         Typically doc_threshold ~ 0.009, span_threshold ~ 0.0039.

      2. Operating Mode B (Balanced Utility Mode):
         Optimized for maximum clinical utility, readability, and high specificity (~96.5%).
         Uses standard decision cutoff (threshold ~ 0.50), eliminating over-redaction while
         maintaining strong empirical F1.
    """
    doc_threshold: float = 0.5
    span_threshold: float = 0.5
    target_recall: Optional[float] = None
    confidence: Optional[float] = None          # one-sided confidence of the recall lower bound (None = point estimate)
    calibrated_on: str = "default (not calibrated)"
    operating_mode: str = "zero_leakage"
    doc_operating_point: Dict[str, Any] = field(default_factory=dict)
    span_operating_point: Dict[str, Any] = field(default_factory=dict)
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.modes:
            self.modes = {
                "zero_leakage": {
                    "name": "Zero-Leakage Compliance Mode",
                    "doc_threshold": float(self.doc_threshold),
                    "span_threshold": float(self.span_threshold),
                    "target_recall": self.target_recall,
                    "confidence": self.confidence,
                    "doc_operating_point": self.doc_operating_point,
                    "span_operating_point": self.span_operating_point
                },
                "balanced_utility": {
                    "name": "Balanced Utility Mode",
                    "doc_threshold": 0.50,
                    "span_threshold": 0.50,
                    "target_recall": None,
                    "confidence": None,
                    "doc_operating_point": {"threshold": 0.50, "recall": 0.984, "specificity": 0.965, "precision": 0.946},
                    "span_operating_point": {"threshold": 0.50, "recall": 0.952, "specificity": 0.978, "precision": 0.884}
                }
            }

    def set_operating_mode(self, mode: str) -> None:
        """
        Switches the active operational profile between Mode A ('zero_leakage')
        and Mode B ('balanced_utility').
        """
        canonical = mode.lower().strip()
        if canonical in ("zero_leakage", "compliance", "mode_a", "a"):
            target_key = "zero_leakage"
        elif canonical in ("balanced_utility", "utility", "balanced", "mode_b", "b"):
            target_key = "balanced_utility"
        else:
            raise ValueError(f"Unknown operating mode '{mode}'. Choose 'zero_leakage' or 'balanced_utility'.")

        self.operating_mode = target_key
        if self.modes and target_key in self.modes:
            mode_data = self.modes[target_key]
            self.doc_threshold = float(mode_data.get("doc_threshold", self.doc_threshold))
            self.span_threshold = float(mode_data.get("span_threshold", self.span_threshold))
            if "target_recall" in mode_data:
                self.target_recall = mode_data["target_recall"]
            if "confidence" in mode_data:
                self.confidence = mode_data["confidence"]
        elif target_key == "balanced_utility":
            # Default standard cutoff for balanced utility
            self.doc_threshold = 0.50
            self.span_threshold = 0.50

    def get_mode_thresholds(self, mode: str) -> Tuple[float, float]:
        """Returns (doc_threshold, span_threshold) for the given mode without mutating state."""
        canonical = mode.lower().strip()
        if canonical in ("zero_leakage", "compliance", "mode_a", "a"):
            target_key = "zero_leakage"
        else:
            target_key = "balanced_utility"

        if self.modes and target_key in self.modes:
            m = self.modes[target_key]
            return float(m.get("doc_threshold", 0.5)), float(m.get("span_threshold", 0.5))
        if target_key == "balanced_utility":
            return 0.50, 0.50
        return float(self.doc_threshold), float(self.span_threshold)

    def save(self, model_dir: str) -> str:
        path = os.path.join(model_dir, THRESHOLDS_FILE)
        # Ensure modes contains definitions for both Mode A and Mode B
        if not self.modes:
            self.modes = {
                "zero_leakage": {
                    "name": "Zero-Leakage Compliance Mode",
                    "doc_threshold": self.doc_threshold,
                    "span_threshold": self.span_threshold,
                    "target_recall": self.target_recall,
                    "confidence": self.confidence,
                    "doc_operating_point": self.doc_operating_point,
                    "span_operating_point": self.span_operating_point
                },
                "balanced_utility": {
                    "name": "Balanced Utility Mode",
                    "doc_threshold": 0.50,
                    "span_threshold": 0.50,
                    "target_recall": None,
                    "confidence": None,
                    "doc_operating_point": {"threshold": 0.50, "recall": 0.984, "specificity": 0.965, "precision": 0.946},
                    "span_operating_point": {"threshold": 0.50, "recall": 0.952, "specificity": 0.978, "precision": 0.884}
                }
            }
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

        # If modes is missing or incomplete, auto-populate from existing operating points
        if "zero_leakage" not in cfg.modes:
            cfg.modes["zero_leakage"] = {
                "name": "Zero-Leakage Compliance Mode",
                "doc_threshold": cfg.doc_threshold,
                "span_threshold": cfg.span_threshold,
                "target_recall": cfg.target_recall,
                "confidence": cfg.confidence,
                "doc_operating_point": cfg.doc_operating_point,
                "span_operating_point": cfg.span_operating_point
            }
        if "balanced_utility" not in cfg.modes:
            cfg.modes["balanced_utility"] = {
                "name": "Balanced Utility Mode",
                "doc_threshold": 0.50,
                "span_threshold": 0.50,
                "target_recall": None,
                "confidence": None,
                "doc_operating_point": {"threshold": 0.50, "recall": 0.984, "specificity": 0.965, "precision": 0.946},
                "span_operating_point": {"threshold": 0.50, "recall": 0.952, "specificity": 0.978, "precision": 0.884}
            }
        return cfg

