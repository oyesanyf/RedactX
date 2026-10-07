"""
Calibrate RedactX operating thresholds for a target recall.

A redaction system is judged by its misses. Instead of a fixed 0.5 cut-off, this script measures the
checkpoint's scores on held-out documents and picks:

  doc_threshold   the largest P(PHI) threshold that still flags >= target_recall of PHI documents
  span_threshold  the largest span-head token-probability threshold such that >= target_recall of gold
                  PHI spans have at least one token above it ("touch" recall: the span is at least
                  partially found; the span head then grows the span over contiguous tokens)

and writes them to <model_dir>/redactx_thresholds.json, which the engine and the production server load.
The script also reports the price paid for that recall (specificity / precision at the chosen point).

Calibration data is DISJOINT from the documents validate_model.py benchmarks on:
  ai4privacy  ai4privacy/pii-masking-openpii-1m validation split, English, skipping the first
              --ai4privacy-offset qualifying docs (validate_model.py uses the first 200);
              positives = real docs with gold spans, negatives = their generic-replaced clean twins
  pubmed      PubMedQA pqa_labeled abstracts after --pubmed-offset (validate_model.py uses the first 100)
  generator   synthetic clinical notes from redactx.data.generator with a seed (default 4242) that differs
              from training and from the benchmark (1337); works offline

Usage:
    python calibrate_thresholds.py --model-dir ./models/RedactX-v2 --target-recall 0.98
    python calibrate_thresholds.py --model-dir ./models/RedactX-v2 --sources generator --dry-run
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from redactx.production.thresholds import ThresholdConfig, operating_point, threshold_for_recall

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


def calibrate(engine, positives: Sequence[Dict], negatives: Sequence[str], target_recall: float,
              batch_size: int = 8, sources: str = "") -> Tuple[ThresholdConfig, Dict]:
    """
    positives: [{"text": str, "spans": [(start, end, label), ...]}]   (documents that contain PHI)
    negatives: [str]                                                  (documents that contain no PHI)
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

    pos_doc = [r["p_phi"] for r in pos_res]
    neg_doc = [r["p_phi"] for r in neg_res]
    doc_t = threshold_for_recall(pos_doc, target_recall)
    doc_op = operating_point(pos_doc, neg_doc, doc_t)
    default_doc_op = operating_point(pos_doc, neg_doc, 0.5)

    span_t = float(getattr(engine, "span_threshold", 0.5))
    span_op: Dict = {}
    diag: Dict = {"n_pos_docs": len(pos_doc), "n_neg_docs": len(neg_doc), "seconds": round(elapsed, 1),
                  "doc_at_0.5": default_doc_op}
    if has_span_head:
        span_pos: List[float] = []
        span_neg: List[float] = []
        for d, r in zip(positives, pos_res):
            p, n = gold_span_scores(r["token_probs"], r["raw_ranges"], [(s, e) for s, e, *_ in d["spans"]])
            span_pos.extend(p)
            span_neg.extend(n)
        if span_pos:
            span_t = threshold_for_recall(span_pos, target_recall)
            span_op = operating_point(span_pos, span_neg, span_t)
            span_op["unit"] = "recall = gold spans touched; specificity/precision = non-PHI tokens"
            diag["span_at_0.5"] = operating_point(span_pos, span_neg, 0.5)
    else:
        diag["span_note"] = "checkpoint has no span head; span_threshold left unchanged"

    cfg = ThresholdConfig(
        doc_threshold=round(doc_t, 6),
        span_threshold=round(span_t, 6),
        target_recall=target_recall,
        calibrated_on=f"{sources} | {len(pos_doc)} PHI docs / {len(neg_doc)} clean docs | "
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
            pos.append({"text": r["context"],
                        "spans": [(int(s["start"]), int(s["end"]), str(s.get("category", "pii")))
                                  for s in r["spans"]]})
        elif r["target"] != "Contains_PHI_PII":
            neg.append(r["context"])
    return pos, neg


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate RedactX thresholds for a target recall")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--sources", default="ai4privacy,pubmed,generator",
                    help="comma list of: ai4privacy, pubmed, generator")
    ap.add_argument("--n-ai4privacy", type=int, default=300)
    ap.add_argument("--ai4privacy-offset", type=int, default=1000)
    ap.add_argument("--n-pubmed", type=int, default=150)
    ap.add_argument("--pubmed-offset", type=int, default=200)
    ap.add_argument("--n-generator", type=int, default=300)
    ap.add_argument("--generator-seed", type=int, default=4242)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the result without writing the file")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - {"ai4privacy", "pubmed", "generator"}
    if unknown:
        ap.error(f"unknown sources: {sorted(unknown)}")

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
        positives += docs
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
    print(f"Calibration data: {len(positives)} PHI docs, {len(negatives)} clean docs ({', '.join(used)})", flush=True)

    from redactx.models.openjev import OpenJevVaultGemmaEngine
    engine = OpenJevVaultGemmaEngine.from_pretrained(args.model_dir, device=args.device)
    cfg, diag = calibrate(engine, positives, negatives, args.target_recall, args.batch_size, "; ".join(used))

    print(json.dumps({"thresholds": cfg.__dict__, "diagnostics": diag}, indent=2))
    dop = cfg.doc_operating_point
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
