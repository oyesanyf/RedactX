"""
Calibrate RedactX operating thresholds for a target recall.

A redaction system is judged by its misses. Instead of a fixed 0.5 cut-off, this script measures the
checkpoint's scores on held-out documents and picks:

  doc_threshold   the largest P(PHI) threshold such that, with --confidence (default 95%), the true recall on
                  PHI documents is at least --target-recall (one-sided Wilson lower bound)
  span_threshold  the same for gold PHI spans, where a span counts as found if at least one of its tokens is
                  above the threshold ("touch" recall; the span head then grows the span over contiguous tokens)

Why a lower bound: the largest threshold that *just* reaches the target on the calibration sample sits at the
edge of the score distribution, so on new data it misses the target about half the time. (Measured on v2: the
point-estimate doc threshold 0.974 gave 0.80 / 0.955 recall on held-out sets H / P.) `--point-estimate`
restores that behaviour.

Why per source: positives are grouped by source (ai4privacy, generator) and EACH group must meet the target; the
threshold is the minimum of the per-source thresholds. Pooling lets easy positives (generator notes score ~0.99)
hide a hard source.

The thresholds are written to <model_dir>/redactx_thresholds.json, which the engine and the production server load.
The script also reports the price paid for that recall (specificity / precision at the chosen point).

Calibration data is DISJOINT from the documents validate_model.py benchmarks on:
  ai4privacy  ai4privacy/pii-masking-openpii-1m validation split, English, skipping the first
              --ai4privacy-offset qualifying docs (validate_model.py uses the first 200);
              positives = real docs with gold spans, negatives = their generic-replaced clean twins
  pubmed      PubMedQA pqa_labeled abstracts after --pubmed-offset (validate_model.py uses the first 100)
  generator   synthetic clinical notes from redactx.data.generator with a seed (default 4242) that differs
              from training and from the benchmark (1337); works offline
  n2c2        n2c2 2014 de-identification corpus, TRAIN split only (benchmark_n2c2.py scores the test split);
              notes cut into --max-chars windows; positives = windows with gold PHI (calibrated as their own
              source, so real clinical notes must meet the target on their own), negatives = their generic twins
              + PHI-free windows. Needs --n2c2-dir (DUA corpus, never downloaded)

Usage:
    python calibrate_thresholds.py --model-dir ./models/RedactX-v2 --target-recall 0.98
    python calibrate_thresholds.py --model-dir ./models/RedactX-v2 --sources generator --dry-run
    python calibrate_thresholds.py --model-dir ./models/RedactX-v3 --n2c2-dir D:/data/n2c2-2014
        --sources ai4privacy,pubmed,generator,n2c2
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from redactx.production.thresholds import (ThresholdConfig, conservative_threshold_for_recall, operating_point,
                                           threshold_for_recall)

GoldSpan = Tuple[int, int]


def gold_span_scores(token_probs: Sequence[float], raw_ranges: Sequence[Tuple[int, int]],
                     gold: Sequence[GoldSpan]) -> Tuple[List[float], List[float]]:
    """
    Per gold span: max span-head probability over tokens whose raw-text range overlaps the span.
    Also returns the probabilities of real-text tokens that overlap NO gold span (token-level negatives).
    """
    pos: List[float] = []
    for gs, ge in gold:
        best = 0.0
        for p, (s, e) in zip(token_probs, raw_ranges):
            if s >= 0 and s < ge and e > gs:
                best = max(best, float(p))
        pos.append(best)
    neg: List[float] = []
    for p, (s, e) in zip(token_probs, raw_ranges):
        if s < 0 or e <= s:
            continue
        if not any(s < ge and e > gs for gs, ge in gold):
            neg.append(float(p))
    return pos, neg


def _floor6(x: float) -> float:
    """Floor to 6 decimals: rounding up could move the threshold above the boundary positive's score."""
    return math.floor(float(x) * 1e6) / 1e6


def _choose(scores: Sequence[float], target_recall: float, confidence: Optional[float]) -> Tuple[float, Dict]:
    if confidence is None:
        return threshold_for_recall(scores, target_recall), {"method": "point estimate"}
    t, lb, ok = conservative_threshold_for_recall(scores, target_recall, confidence)
    return t, {"method": f"Wilson lower bound, one-sided {confidence:.0%}", "recall_lower_bound": round(lb, 4),
               "target_certified": ok}


def _choose_stratified(groups: Dict[str, List[float]], target_recall: float, confidence: Optional[float]
                       ) -> Tuple[float, Dict, Dict[str, Dict]]:
    """
    Every source must meet the target on its own; the threshold is the minimum of the per-source thresholds.
    Pooling hides a hard source behind an easy one (measured on v2: easy generator positives let the pooled
    threshold reach 0.98 while ai4privacy-like documents got 0.955 held-out recall).
    """
    per: Dict[str, Dict] = {}
    best_t: Optional[float] = None
    best_method: Dict = {}
    for name, scores in sorted(groups.items()):
        if not scores:
            continue
        t, method = _choose(scores, target_recall, confidence)
        per[name] = {"threshold": round(t, 6), "n_pos": len(scores), **method}
        if best_t is None or t < best_t:
            best_t, best_method = t, dict(method)
            best_method["binding_source"] = name
    if best_t is None:
        raise ValueError("no positive scores to calibrate on")
    if confidence is not None:
        best_method["target_certified"] = all(p.get("target_certified", True) for p in per.values())
    return best_t, best_method, per


def calibrate(engine, positives: Sequence[Dict], negatives: Sequence[str], target_recall: float,
              batch_size: int = 8, sources: str = "", confidence: Optional[float] = 0.95
              ) -> Tuple[ThresholdConfig, Dict]:
    """
    positives: [{"text": str, "spans": [(start, end, label), ...], "source": str (optional)}]
               (documents that contain PHI; positives are calibrated per "source", see _choose_stratified)
    negatives: [str]                                                  (documents that contain no PHI)
    confidence: choose the largest threshold whose one-sided lower confidence bound on recall still meets
                target_recall (default 0.95). None = point estimate (threshold at the edge of the sample,
                which falls below target on new data about half the time).
    Returns (ThresholdConfig, diagnostics).
    """
    if not positives or not negatives:
        raise ValueError("calibration needs at least one positive and one negative document")
    has_span_head = getattr(engine, "span_locator", None) is not None
    t0 = time.perf_counter()
    pos_res = engine.score_texts([d["text"] for d in positives], batch_size=batch_size,
                                 return_token_probs=has_span_head)
    neg_res = engine.score_texts(list(negatives), batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    srcs = [str(d.get("source", "all")) for d in positives]
    pos_doc = [r["p_phi"] for r in pos_res]
    neg_doc = [r["p_phi"] for r in neg_res]
    doc_groups: Dict[str, List[float]] = {}
    for s, p in zip(srcs, pos_doc):
        doc_groups.setdefault(s, []).append(p)
    doc_t, doc_method, doc_per = _choose_stratified(doc_groups, target_recall, confidence)
    doc_t = _floor6(doc_t)
    doc_op = operating_point(pos_doc, neg_doc, doc_t, confidence)
    doc_op.update(doc_method)
    doc_op["per_source"] = {k: {**v, "recall_at_chosen": operating_point(doc_groups[k], neg_doc, doc_t)["recall"]}
                            for k, v in doc_per.items()}

    span_t = float(getattr(engine, "span_threshold", 0.5))
    span_op: Dict = {}
    diag: Dict = {"n_pos_docs": len(pos_doc), "n_neg_docs": len(neg_doc), "seconds": round(elapsed, 1),
                  "doc_at_0.5": operating_point(pos_doc, neg_doc, 0.5, confidence),
                  "doc_pooled_point_estimate_threshold": round(threshold_for_recall(pos_doc, target_recall), 6)}
    if has_span_head:
        span_groups: Dict[str, List[float]] = {}
        span_pos: List[float] = []
        span_neg: List[float] = []
        for s, d, r in zip(srcs, positives, pos_res):
            p, n = gold_span_scores(r["token_probs"], r["raw_ranges"], [(a, b) for a, b, *_ in d["spans"]])
            span_groups.setdefault(s, []).extend(p)
            span_pos.extend(p)
            span_neg.extend(n)
        if span_pos:
            span_t, span_method, span_per = _choose_stratified(span_groups, target_recall, confidence)
            span_t = _floor6(span_t)
            span_op = operating_point(span_pos, span_neg, span_t, confidence)
            span_op.update(span_method)
            span_op["per_source"] = {
                k: {**v, "recall_at_chosen": operating_point(span_groups[k], span_neg, span_t)["recall"]}
                for k, v in span_per.items()}
            span_op["unit"] = ("recall = gold spans touched; specificity/precision = non-PHI tokens. Spans in one "
                               "document are correlated, so the span lower bound is somewhat optimistic.")
            diag["span_at_0.5"] = operating_point(span_pos, span_neg, 0.5, confidence)
            diag["span_pooled_point_estimate_threshold"] = round(threshold_for_recall(span_pos, target_recall), 6)
    else:
        diag["span_note"] = "checkpoint has no span head; span_threshold left unchanged"

    cfg = ThresholdConfig(
        doc_threshold=doc_t,
        span_threshold=span_t if not has_span_head else _floor6(span_t),
        target_recall=target_recall,
        confidence=confidence,
        calibrated_on=f"{sources} | {len(pos_doc)} PHI docs / {len(neg_doc)} clean docs | "
                      f"{'point estimate' if confidence is None else f'{confidence:.0%} lower bound'} | "
                      f"{time.strftime('%Y-%m-%d', time.gmtime())}",
        doc_operating_point=doc_op,
        span_operating_point=span_op,
    )
    return cfg, diag


def load_generator(n: int, seed: int) -> Tuple[List[Dict], List[str]]:
    from redactx.data.generator import generate_redactx_corpus
    pos, neg = [], []
    for r in generate_redactx_corpus(num_samples=n, phi_ratio=0.5, seed=seed):
        if r["target"] == "Contains_PHI_PII" and r.get("spans"):
            pos.append({"text": r["context"], "source": "generator",
                        "spans": [(int(s["start"]), int(s["end"]), str(s.get("category", "pii")))
                                  for s in r["spans"]]})
        elif r["target"] != "Contains_PHI_PII":
            neg.append(r["context"])
    return pos, neg


def load_n2c2_calibration(root: str, n_pos: int, max_chars: int, seed: int = 4242
                          ) -> Tuple[List[Dict], List[str], Dict]:
    """
    n2c2 2014 TRAIN split only (benchmark_n2c2.py scores the test split, so calibration stays disjoint from it).
    Notes are cut into production-sized windows.
      positives  windows that contain gold PHI (source "n2c2"), up to n_pos, sampled with `seed`
      negatives  the generic-replaced twin of every chosen positive window, plus up to n_pos // 2 windows that
                 contain no gold PHI at all (real clinical text with nothing to redact)
    """
    import random
    from redactx.data.contrastive_corpus import replace_spans_with_generic
    from redactx.data.n2c2 import load_n2c2, windows_with_spans

    docs, report = load_n2c2(root, split="train")
    with_phi: List[Dict] = []
    clean: List[str] = []
    for doc in docs:
        for w in windows_with_spans(doc, max_chars=max_chars):
            if not w["text"].strip():
                continue
            if w["spans"]:
                with_phi.append(w)
            else:
                clean.append(w["text"])
    rng = random.Random(seed)
    rng.shuffle(with_phi)
    rng.shuffle(clean)
    pos = []
    for w in with_phi[:max(0, n_pos)]:
        spans: List[Tuple[int, int, str]] = []
        for s, e, lab in sorted(w["spans"]):
            if spans and s < spans[-1][1]:          # overlapping tags: keep one, extended to cover both
                ps, pe, pl = spans[-1]
                spans[-1] = (ps, max(pe, e), pl)
            else:
                spans.append((s, e, lab))
        pos.append({"text": w["text"], "spans": spans, "source": "n2c2"})
    neg = [replace_spans_with_generic(d["text"], d["spans"]) for d in pos]
    neg += clean[:max(0, n_pos // 2)]
    report = {**report, "windows_with_phi": len(with_phi), "windows_without_phi": len(clean),
              "pos_used": len(pos), "neg_used": len(neg)}
    return pos, neg, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate RedactX thresholds for a target recall")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--sources", default="ai4privacy,pubmed,generator",
                    help="comma list of: ai4privacy, pubmed, generator, n2c2 (n2c2 needs --n2c2-dir)")
    ap.add_argument("--n-ai4privacy", type=int, default=300)
    ap.add_argument("--ai4privacy-offset", type=int, default=1000)
    ap.add_argument("--n-pubmed", type=int, default=150)
    ap.add_argument("--pubmed-offset", type=int, default=200)
    ap.add_argument("--n-generator", type=int, default=300)
    ap.add_argument("--generator-seed", type=int, default=4242)
    ap.add_argument("--n2c2-dir", default=None, help="unpacked n2c2 2014 corpus (only the TRAIN split is used here)")
    ap.add_argument("--n-n2c2", type=int, default=300, help="n2c2 PHI windows to calibrate on")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the result without writing the file")
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="one-sided confidence that true recall >= target (Wilson lower bound). Default 0.95")
    ap.add_argument("--point-estimate", action="store_true",
                    help="old behaviour: threshold at the edge of the sample (no safety margin)")
    args = ap.parse_args(argv)
    if not args.point_estimate and not 0.5 <= args.confidence < 1.0:
        ap.error("--confidence must be in [0.5, 1)")
    confidence = None if args.point_estimate else args.confidence

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - {"ai4privacy", "pubmed", "generator", "n2c2"}
    if unknown:
        ap.error(f"unknown sources: {sorted(unknown)}")
    if "n2c2" in sources:
        from redactx.data.n2c2 import resolve_n2c2_dir
        args.n2c2_dir = resolve_n2c2_dir(args.n2c2_dir)
        if not args.n2c2_dir or not os.path.isdir(args.n2c2_dir):
            ap.error("source n2c2 requested but n2c2 data directory could not be found under data/; specify --n2c2-dir")

    from redactx.data.contrastive_corpus import replace_spans_with_generic
    from validate_model import load_ai4privacy_validation, load_pubmedqa_labeled

    token = os.environ.get("HF_TOKEN")
    if not token and ("ai4privacy" in sources or "pubmed" in sources):
        try:
            from train import resolve_hf_token
            token = resolve_hf_token(None)
        except Exception:
            token = None

    positives: List[Dict] = []
    negatives: List[str] = []
    used: List[str] = []
    if "ai4privacy" in sources:
        docs = load_ai4privacy_validation(args.n_ai4privacy, token, args.max_chars, offset=args.ai4privacy_offset)
        positives += [{**d, "source": "ai4privacy"} for d in docs]
        negatives += [replace_spans_with_generic(d["text"], d["spans"]) for d in docs]
        used.append(f"ai4privacy-val[{args.ai4privacy_offset}:+{len(docs)}]+twins")
    if "pubmed" in sources:
        q = load_pubmedqa_labeled(args.n_pubmed, token, args.max_chars, offset=args.pubmed_offset)
        negatives += q
        used.append(f"pubmedqa-labeled[{args.pubmed_offset}:+{len(q)}]")
    if "generator" in sources:
        gp, gn = load_generator(args.n_generator, args.generator_seed)
        positives += gp
        negatives += gn
        used.append(f"generator(seed={args.generator_seed},n={args.n_generator})")
    if "n2c2" in sources:
        np_, nn, rep = load_n2c2_calibration(args.n2c2_dir, args.n_n2c2, args.max_chars, args.generator_seed)
        positives += np_
        negatives += nn
        used.append(f"n2c2-2014-train(windows={len(np_)}+{len(nn)})")
        print(f"n2c2 train: {rep['files']} notes, {rep['tags']} tags ({rep['relocated']} relocated, "
              f"{rep['dropped']} dropped), {rep['windows_with_phi']} PHI windows / "
              f"{rep['windows_without_phi']} PHI-free windows", flush=True)
    print(f"Calibration data: {len(positives)} PHI docs, {len(negatives)} clean docs ({', '.join(used)})", flush=True)

    from redactx.models.openjev import OpenJevVaultGemmaEngine
    engine = OpenJevVaultGemmaEngine.from_pretrained(args.model_dir, device=args.device)
    cfg, diag = calibrate(engine, positives, negatives, args.target_recall, args.batch_size, "; ".join(used),
                          confidence=confidence)

    print(json.dumps({"thresholds": cfg.__dict__, "diagnostics": diag}, indent=2))
    dop = cfg.doc_operating_point
    for name, op in (("document", dop), ("span", cfg.span_operating_point)):
        if op and op.get("target_certified") is False:
            print(f"WARNING: not enough {name} positives to certify recall >= {args.target_recall} at "
                  f"{confidence:.0%} confidence (best lower bound {op.get('recall_lower_bound')}). The threshold "
                  "was set to flag every calibration positive; add calibration data or lower the target.",
                  flush=True)
    if dop.get("specificity") is not None and dop["specificity"] < 0.5:
        print(f"WARNING: at recall {dop['recall']} the document specificity is only {dop['specificity']}: "
              "most clean documents will be flagged. The checkpoint does not separate PHI from clean text "
              "well enough for this target; retrain or lower --target-recall.", flush=True)
    if args.dry_run:
        print("dry run: thresholds not written")
    else:
        print("Wrote", os.path.abspath(cfg.save(args.model_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
