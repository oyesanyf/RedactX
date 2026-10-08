"""
Benchmark RedactX against Microsoft Presidio on the n2c2 2014 de-identification corpus (real clinical notes).

    python benchmark_n2c2.py --model-dir ./models/RedactX-v3 --n2c2-dir D:\\data\\n2c2-2014
    python benchmark_n2c2.py --model-dir ./models/RedactX-v3 --n2c2-dir D:\\data\\n2c2-2014 --split test --limit 100

The corpus needs a Data Use Agreement (https://portal.dbmi.hms.harvard.edu); nothing is downloaded here.
Default split is `test` (514 notes); calibrate on `train` (calibrate_thresholds.py --sources n2c2,...) so the two
never overlap.

Systems (all run live on the same notes):
  redactx             the model alone (span head, production chunking)
  redactx+validators  the model plus verified SSN / e-mail / phone / MRN / NPI validators (production default)
  validators          the validators alone (what they contribute)
  presidio            presidio_analyzer + spaCy en_core_web_lg
  hybrid              redactx+validators UNION presidio (recommended deployment mode)

Metrics:
  char P / R / F1     character level vs ALL n2c2 PHI, whitespace ignored (boundary tolerant)
  HIPAA recall        recall restricted to Safe Harbor identifiers (redactx/data/n2c2.py mapping; ages only > 89)
  per HIPAA category  touched recall (gold span overlapped) and char recall, per redactx Category
  per n2c2 type       the same per original n2c2 TYPE (PATIENT, DOCTOR, HOSPITAL, MEDICALRECORD, ...)
  windows             600-char windows: recall = PHI windows with any prediction; specificity = PHI-free windows
                      with no prediction (over-redaction on real clinical text)
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List

from validate_model import _span_bounds as _bounds, char_metrics, per_category_recall  # Finding / dict / tuple


def per_type_recall(docs: List[Dict], key: str) -> Dict[str, Dict]:
    acc: Dict[str, Dict] = {}
    for d in docs:
        text = d["text"]
        preds = [_bounds(f) for f in d[key]]
        pchars = {i for a, b in preds for i in range(a, b)}
        for g in d["spans"]:
            st = acc.setdefault(g["type"], {"n": 0, "touched": 0, "chars": 0, "covered": 0})
            st["n"] += 1
            st["touched"] += int(any(a < g["end"] and b > g["start"] for a, b in preds))
            idx = [i for i in range(g["start"], g["end"]) if not text[i].isspace()]
            st["chars"] += len(idx)
            st["covered"] += sum(1 for i in idx if i in pchars)
    return {t: {"n": v["n"], "touched_recall": round(v["touched"] / v["n"], 4),
                "char_recall": round(v["covered"] / v["chars"], 4) if v["chars"] else None}
            for t, v in sorted(acc.items(), key=lambda kv: -kv[1]["n"])}


def hipaa_recall(docs: List[Dict], key: str) -> Dict:
    n = touched = chars = covered = 0
    for d in docs:
        text = d["text"]
        preds = [_bounds(f) for f in d[key]]
        pchars = {i for a, b in preds for i in range(a, b)}
        for g in d["spans"]:
            if not g["hipaa"]:
                continue
            n += 1
            touched += int(any(a < g["end"] and b > g["start"] for a, b in preds))
            idx = [i for i in range(g["start"], g["end"]) if not text[i].isspace()]
            chars += len(idx)
            covered += sum(1 for i in idx if i in pchars)
    return {"n_spans": n, "touched_recall": round(touched / n, 4) if n else None,
            "char_recall": round(covered / chars, 4) if chars else None}


def window_metrics(docs: List[Dict], key: str, max_chars: int) -> Dict:
    from redactx.production.chunking import chunk_text
    tp = fn = fp = tn = 0
    for d in docs:
        gold = {i for g in d["spans"] for i in range(g["start"], g["end"])}
        pred = {i for f in d[key] for i in range(*_bounds(f))}
        for w in chunk_text(d["text"], max_chars, 0):
            g = any(i in gold for i in range(w.start, w.end))
            p = any(i in pred for i in range(w.start, w.end))
            tp += g and p; fn += g and not p; fp += (not g) and p; tn += (not g) and not p
    return {"phi_windows": tp + fn, "clean_windows": fp + tn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None}


def run(args) -> Dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from redactx.data.n2c2 import load_n2c2, resolve_n2c2_dir
    from redactx.production.detectors import (PresidioDetector, RedactXDetector, merge_findings,
                                              validator_findings)
    from redactx.production.validators import StructuredIdValidator

    n2c2_dir = resolve_n2c2_dir(args.n2c2_dir)
    print(f"Using n2c2 corpus: {os.path.abspath(n2c2_dir)}", flush=True)
    docs, report = load_n2c2(n2c2_dir, args.split, args.limit, part=args.part)
    print(f"n2c2 2014 {args.split} (part: {args.part}): {report['files']} notes, {report['tags']} PHI tags "
          f"(exact offsets {report['exact']}, relocated {report['relocated']}, dropped {report['dropped']})", flush=True)

    OUT: Dict = {"split": args.split, "load_report": report, "n_docs": len(docs),
                 "mean_chars": round(sum(len(d["text"]) for d in docs) / max(len(docs), 1), 1), "seconds": {}}
    validator = StructuredIdValidator()
    OUT["validators"] = {"kinds": list(validator.kinds), "phone_backend": validator.phone_backend}

    t0 = time.perf_counter()
    for d in docs:
        d["validators"] = validator_findings(d["text"], validator)
    OUT["seconds"]["validators"] = round(time.perf_counter() - t0, 2)

    systems: Dict[str, Callable] = {}
    if not args.skip_redactx:
        from redactx.models.openjev import OpenJevVaultGemmaEngine
        engine = OpenJevVaultGemmaEngine.from_pretrained(args.model_dir, device=args.device)
        OUT.update({"model_dir": os.path.abspath(args.model_dir),
                    "doc_threshold": float(getattr(engine, "doc_threshold", 0.5)),
                    "span_threshold": float(getattr(engine, "span_threshold", 0.5)),
                    "thresholds_calibrated_on": getattr(getattr(engine, "thresholds", None), "calibrated_on", None)})
        det = RedactXDetector(engine, max_chars=args.max_chars, overlap=args.overlap, batch_size=args.batch_size,
                              validators=None)
        t0 = time.perf_counter()
        for i in range(0, len(docs), args.doc_batch):
            part = docs[i:i + args.doc_batch]
            for d, r in zip(part, det.detect_many([d["text"] for d in part])):
                d["redactx"] = r.findings
            print(f"  redactx {min(i + args.doc_batch, len(docs))}/{len(docs)}", flush=True)
        OUT["seconds"]["redactx"] = round(time.perf_counter() - t0, 1)
        for d in docs:
            d["redactx+validators"] = merge_findings(d["text"], list(d["redactx"]) + list(d["validators"]))
        systems["redactx"] = "redactx"
        systems["redactx+validators"] = "redactx+validators"
    systems["validators"] = "validators"

    if not args.skip_presidio:
        try:
            pres = PresidioDetector(score_threshold=args.presidio_score)
            t0 = time.perf_counter()
            for d in docs:
                d["presidio"] = pres.detect(d["text"]).findings
            OUT["seconds"]["presidio"] = round(time.perf_counter() - t0, 1)
            systems["presidio"] = "presidio"
            if "redactx+validators" in systems:
                for d in docs:
                    d["hybrid"] = merge_findings(d["text"], list(d["redactx+validators"]) + list(d["presidio"]))
                systems["hybrid"] = "hybrid"
        except Exception as e:
            OUT["presidio"] = f"unavailable: {type(e).__name__}: {e}"

    gold_docs = [{"text": d["text"], "gold": d["spans"], **{k: d[k] for k in systems}} for d in docs]
    OUT["chars_all_phi"] = {k: char_metrics(gold_docs, lambda x, k=k: x[k]) for k in systems}
    OUT["hipaa_recall"] = {k: hipaa_recall(docs, k) for k in systems}
    OUT["per_hipaa_category_recall"] = {k: per_category_recall(gold_docs, lambda x, k=k: x[k]) for k in systems}
    OUT["per_n2c2_type_recall"] = {k: per_type_recall(docs, k) for k in systems}
    OUT["windows"] = {k: window_metrics(docs, k, args.max_chars) for k in systems}

    out_path = args.out or os.path.join(args.model_dir if not args.skip_redactx else ".", "n2c2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=2)
    print("Saved:", os.path.abspath(out_path))
    return OUT


def print_tables(OUT: Dict) -> None:
    def f(v):
        return "   -  " if v is None else f"{v:6.3f}"

    systems = list(OUT["chars_all_phi"].keys())
    w = 12
    print("\n" + "=" * 96)
    print(f" n2c2 2014 {OUT['split']} ({OUT['n_docs']} real clinical notes, mean {OUT['mean_chars']} chars)")
    print("=" * 96)
    print(f"{'metric':30s}" + "".join(f"{s:>{w + 6}s}" for s in systems))
    print("-" * 96)
    for k in ("precision", "recall", "f1"):
        print(f"{'char ' + k + ' (all PHI)':30s}" + "".join(f"{f(OUT['chars_all_phi'][s][k]):>{w + 6}s}" for s in systems))
    for k in ("touched_recall", "char_recall"):
        print(f"{'HIPAA ' + k:30s}" + "".join(f"{f(OUT['hipaa_recall'][s][k]):>{w + 6}s}" for s in systems))
    for k in ("recall", "specificity"):
        print(f"{'window ' + k:30s}" + "".join(f"{f(OUT['windows'][s][k]):>{w + 6}s}" for s in systems))
    print("-" * 96)
    cats = OUT["per_hipaa_category_recall"][systems[0]]
    print(f"{'HIPAA category (char recall)':24s}{'n':>6s}" + "".join(f"{s:>{w + 6}s}" for s in systems))
    for c, v in cats.items():
        row = "".join(f"{f(OUT['per_hipaa_category_recall'][s].get(c, {}).get('char_recall')):>{w + 6}s}"
                      for s in systems)
        print(f"{c:24s}{v['n']:6d}{row}")
    print("-" * 96)
    types = OUT["per_n2c2_type_recall"][systems[0]]
    print(f"{'n2c2 type (touched recall)':24s}{'n':>6s}" + "".join(f"{s:>{w + 6}s}" for s in systems))
    for t, v in types.items():
        row = "".join(f"{f(OUT['per_n2c2_type_recall'][s].get(t, {}).get('touched_recall')):>{w + 6}s}"
                      for s in systems)
        print(f"{t:24s}{v['n']:6d}{row}")
    print("-" * 96)
    print(f"seconds: {OUT['seconds']} | thresholds: doc {OUT.get('doc_threshold')} span {OUT.get('span_threshold')} "
          f"({OUT.get('thresholds_calibrated_on') or 'default, not calibrated'})")
    print("HIPAA = Safe Harbor identifiers only (doctor names, hospitals, professions, states excluded; ages > 89).")
    print("=" * 96)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RedactX vs Presidio on n2c2 2014 (per HIPAA category)")
    ap.add_argument("--model-dir", default="./models/RedactX-v3")
    ap.add_argument("--n2c2-dir", default=None, help="folder containing the unpacked n2c2 2014 de-id release (default: auto-discover under data/)")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--part", default="eval", choices=["all", "calib", "eval"],
                    help="n2c2 partition (default: eval, strictly held out from calibration; use 'all' for all notes)")
    ap.add_argument("--limit", type=int, default=None, help="first N notes only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--doc-batch", type=int, default=8, help="notes per detect_many call")
    ap.add_argument("--presidio-score", type=float, default=0.35)
    ap.add_argument("--skip-presidio", action="store_true")
    ap.add_argument("--skip-redactx", action="store_true", help="Presidio + validators only (no GPU needed)")
    ap.add_argument("--out", default=None, help="output JSON (default <model-dir>/n2c2_results.json)")
    args = ap.parse_args(argv)
    print_tables(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
