"""
Precision-Recall (PR) and ROC curve computation with Pareto frontier and dual operating modes.

Mathematical Formulations:
1. Precision and Recall at threshold t:
   Precision(t) = TP(t) / (TP(t) + FP(t))
   Recall(t)    = TP(t) / (TP(t) + FN(t))
   Specificity(t) = TN(t) / (TN(t) + FP(t))

2. Area Under the Precision-Recall Curve (AUPRC):
   AUPRC = sum_{k=1}^K (R_k - R_{k-1}) * (P_k + P_{k-1}) / 2

3. Area Under the Receiver Operating Characteristic Curve (AUROC):
   AUROC = sum_{k=1}^K (FPR_k - FPR_{k-1}) * (TPR_k + TPR_{k-1}) / 2

4. Expected Calibration Error (ECE, M=10 bins):
   ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
   where B_m = {i : p_i in ((m-1)/M, m/M]}

5. Brier Score:
   Brier = (1 / N) * sum_{i=1}^N (p_i - y_i)^2

6. One-sided Wilson Score Lower Confidence Bound:
   w^-(k, n, z) = (p + z^2/(2n) - z * sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)
   where p = k/n and z = Phi^{-1}(1 - alpha)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class OperatingPointMetrics:
    mode: str
    name: str
    threshold: float
    precision: float
    recall: float
    specificity: float
    f1: float
    n_pos: int
    n_neg: int
    recall_lower_bound: Optional[float] = None
    confidence: Optional[float] = None
    target_certified: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurveData:
    thresholds: List[float]
    precisions: List[float]
    recalls: List[float]
    fprs: List[float]
    tprs: List[float]
    auprc: float
    auroc: float
    brier_score: float
    ece_10bin: float
    mode_a: OperatingPointMetrics
    mode_b: OperatingPointMetrics
    pareto_frontier: List[Dict[str, float]]
    n_samples: int
    n_positives: int
    n_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def wilson_lower_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """
    Computes the one-sided Wilson score lower confidence bound for binomial proportion k/n.

    Formula:
        w^- = (p + z^2/(2n) - z * sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)
        where p = k/n, z = NormalDist().inv_cdf(confidence)
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    if not (0 <= k <= n):
        raise ValueError(f"Success count k ({k}) must satisfy 0 <= k <= n ({n}).")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence must be strictly between 0 and 1.")
    p = k / n
    z = NormalDist().inv_cdf(confidence)
    if z <= 0:
        return p
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n)))
    denom = 1.0 + z2 / n
    return max(0.0, min(1.0, (centre - spread) / denom))


def compute_brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """
    Computes Brier score: (1/N) * sum_{i=1}^N (p_i - y_i)^2
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have identical lengths.")
    if not y_true:
        return 0.0
    total = sum((float(p) - float(y)) ** 2 for y, p in zip(y_true, y_prob))
    return float(total / len(y_true))


def compute_ece_10bin(y_true: Sequence[int], y_prob: Sequence[float], num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across M equal-width confidence bins:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have identical lengths.")
    n = len(y_true)
    if n == 0:
        return 0.0

    bins: List[List[Tuple[int, float]]] = [[] for _ in range(num_bins)]
    for y, p in zip(y_true, y_prob):
        p_val = max(0.0, min(1.0, float(p)))
        bin_idx = min(int(p_val * num_bins), num_bins - 1)
        bins[bin_idx].append((int(y), p_val))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_size = len(b)
        bin_acc = sum(y for y, _ in b) / bin_size
        bin_conf = sum(p for _, p in b) / bin_size
        ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_precision_recall_curve(
    y_true: Sequence[int],
    y_score: Sequence[float]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Computes Precision-Recall curve points for binary classification.

    Returns:
        (precisions, recalls, thresholds)
        ordered from highest threshold to lowest threshold.
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have identical lengths.")
    if not y_true:
        return [1.0], [0.0], [1.0]

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    total_positives = sum(y for _, y in paired)
    if total_positives == 0:
        return [0.0], [0.0], [0.5]

    precisions: List[float] = [1.0]
    recalls: List[float] = [0.0]
    thresholds: List[float] = [float(paired[0][0]) + 1e-6]

    tp = 0
    fp = 0
    i = 0
    n = len(paired)

    while i < n:
        curr_thresh = paired[i][0]
        while i < n and paired[i][0] == curr_thresh:
            if paired[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / total_positives
        precisions.append(float(prec))
        recalls.append(float(rec))
        thresholds.append(float(curr_thresh))

    return precisions, recalls, thresholds


def compute_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Computes Receiver Operating Characteristic (ROC) curve points.

    Returns:
        (fprs, tprs, thresholds)
        starting from (0, 0) up to (1, 1).
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have identical lengths.")
    if not y_true:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    total_pos = sum(y for _, y in paired)
    total_neg = len(paired) - total_pos

    if total_pos == 0 or total_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    fprs: List[float] = [0.0]
    tprs: List[float] = [0.0]
    thresholds: List[float] = [float(paired[0][0]) + 1e-6]

    tp = 0
    fp = 0
    i = 0
    n = len(paired)

    while i < n:
        curr_thresh = paired[i][0]
        while i < n and paired[i][0] == curr_thresh:
            if paired[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        fpr = fp / total_neg
        tpr = tp / total_pos
        fprs.append(float(fpr))
        tprs.append(float(tpr))
        thresholds.append(float(curr_thresh))

    return fprs, tprs, thresholds


def compute_auprc(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """
    Computes Area Under the Precision-Recall Curve using the trapezoidal rule.
    """
    if len(recalls) != len(precisions):
        raise ValueError("recalls and precisions must have identical lengths.")
    if len(recalls) < 2:
        return 0.0

    # Ensure sorted by recall ascending
    points = sorted(zip(recalls, precisions), key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(points)):
        r_prev, p_prev = points[i - 1]
        r_curr, p_curr = points[i]
        dr = r_curr - r_prev
        if dr > 0:
            area += dr * (p_prev + p_curr) / 2.0
    return float(max(0.0, min(1.0, area)))


def compute_auroc(fprs: Sequence[float], tprs: Sequence[float]) -> float:
    """
    Computes Area Under Receiver Operating Characteristic (AUROC) using trapezoidal rule.
    """
    if len(fprs) != len(tprs):
        raise ValueError("fprs and tprs must have identical lengths.")
    if len(fprs) < 2:
        return 0.5

    points = sorted(zip(fprs, tprs), key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(points)):
        fpr_prev, tpr_prev = points[i - 1]
        fpr_curr, tpr_curr = points[i]
        dfpr = fpr_curr - fpr_prev
        if dfpr > 0:
            area += dfpr * (tpr_prev + tpr_curr) / 2.0
    return float(max(0.0, min(1.0, area)))


def compute_pareto_frontier(
    recalls: Sequence[float],
    precisions: Sequence[float],
    thresholds: Sequence[float]
) -> List[Dict[str, float]]:
    """
    Extracts the non-dominated Pareto frontier points where no other point
    achieves strictly higher recall AND higher precision.
    """
    if not (len(recalls) == len(precisions) == len(thresholds)):
        raise ValueError("recalls, precisions, and thresholds must have identical lengths.")
    points = sorted(zip(recalls, precisions, thresholds), key=lambda x: (-x[0], -x[1]))
    pareto: List[Dict[str, float]] = []
    max_prec = -1.0

    for r, p, t in points:
        if p > max_prec:
            pareto.append({
                "recall": round(float(r), 4),
                "precision": round(float(p), 4),
                "threshold": round(float(t), 6),
                "f1": round(2.0 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)
            })
            max_prec = p

    pareto.sort(key=lambda d: d["recall"])
    return pareto


def evaluate_dual_operating_modes(
    y_true: Sequence[int],
    y_score: Sequence[float],
    target_recall: float = 0.98,
    confidence: float = 0.95,
    mode_b_threshold: float = 0.50
) -> CurveData:
    """
    Evaluates binary classification scores across both:
      - Operating Mode A: Zero-Leakage Compliance Mode (Certified Wilson 95% recall >= target_recall)
      - Operating Mode B: Balanced Utility Mode (threshold = mode_b_threshold, high specificity and precision)

    Returns:
        CurveData object with full curves, metrics, and operating points.
    """
    precisions, recalls, thresholds = compute_precision_recall_curve(y_true, y_score)
    fprs, tprs, _ = compute_roc_curve(y_true, y_score)

    auprc = compute_auprc(recalls, precisions)
    auroc = compute_auroc(fprs, tprs)
    brier = compute_brier_score(y_true, y_score)
    ece = compute_ece_10bin(y_true, y_score, num_bins=10)
    pareto = compute_pareto_frontier(recalls, precisions, thresholds)

    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos

    # Mode A: Wilson 95% certified threshold for target_recall
    pos_scores = [float(s) for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [float(s) for y, s in zip(y_true, y_score) if y == 0]
    pos_sorted = sorted(pos_scores, reverse=True)

    mode_a_thresh = float(pos_sorted[-1]) if pos_sorted else 0.01
    mode_a_lb = 0.0
    mode_a_ok = False

    for i, t in enumerate(pos_sorted):
        if i + 1 < len(pos_sorted) and pos_sorted[i + 1] == t:
            continue
        lb = wilson_lower_bound(i + 1, len(pos_sorted), confidence)
        if lb >= target_recall:
            mode_a_thresh = float(t)
            mode_a_lb = lb
            mode_a_ok = True
            break
    if not mode_a_ok and pos_sorted:
        mode_a_lb = wilson_lower_bound(len(pos_sorted), len(pos_sorted), confidence)

    # Evaluate Mode A counts
    tp_a = sum(1 for s in pos_scores if s >= mode_a_thresh)
    fn_a = n_pos - tp_a
    fp_a = sum(1 for s in neg_scores if s >= mode_a_thresh)
    tn_a = n_neg - fp_a

    rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else 0.0
    spec_a = tn_a / (tn_a + fp_a) if (tn_a + fp_a) else 0.0
    f1_a = (2.0 * prec_a * rec_a) / (prec_a + rec_a) if (prec_a + rec_a) else 0.0

    mode_a = OperatingPointMetrics(
        mode="zero_leakage",
        name="Operating Mode A: Zero-Leakage Compliance Mode",
        threshold=round(mode_a_thresh, 6),
        precision=round(prec_a, 4),
        recall=round(rec_a, 4),
        specificity=round(spec_a, 4),
        f1=round(f1_a, 4),
        n_pos=n_pos,
        n_neg=n_neg,
        recall_lower_bound=round(mode_a_lb, 4),
        confidence=confidence,
        target_certified=mode_a_ok
    )

    # Mode B: Balanced Utility Mode (threshold ~ 0.50)
    tp_b = sum(1 for s in pos_scores if s >= mode_b_threshold)
    fn_b = n_pos - tp_b
    fp_b = sum(1 for s in neg_scores if s >= mode_b_threshold)
    tn_b = n_neg - fp_b

    rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) else 0.0
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) else 0.0
    spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) else 0.0
    f1_b = (2.0 * prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) else 0.0

    mode_b = OperatingPointMetrics(
        mode="balanced_utility",
        name="Operating Mode B: Balanced Utility Mode",
        threshold=round(mode_b_threshold, 6),
        precision=round(prec_b, 4),
        recall=round(rec_b, 4),
        specificity=round(spec_b, 4),
        f1=round(f1_b, 4),
        n_pos=n_pos,
        n_neg=n_neg,
        recall_lower_bound=round(wilson_lower_bound(tp_b, n_pos, confidence), 4) if n_pos else None,
        confidence=confidence,
        target_certified=(rec_b >= target_recall)
    )

    return CurveData(
        thresholds=[round(t, 6) for t in thresholds],
        precisions=[round(p, 4) for p in precisions],
        recalls=[round(r, 4) for r in recalls],
        fprs=[round(f, 4) for f in fprs],
        tprs=[round(t, 4) for t in tprs],
        auprc=round(auprc, 4),
        auroc=round(auroc, 4),
        brier_score=round(brier, 4),
        ece_10bin=round(ece, 4),
        mode_a=mode_a,
        mode_b=mode_b,
        pareto_frontier=pareto,
        n_samples=len(y_true),
        n_positives=n_pos,
        n_negatives=n_neg
    )


def export_curve_json(curve_data: CurveData, out_path: str) -> str:
    """Exports CurveData to a structured JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curve_data.to_dict(), f, indent=2)
    return out_path
