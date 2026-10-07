"""
Held-out validation for a trained RedactX checkpoint.

Everything here is data the contrastive training recipe does NOT use:
  H.  Handwritten held-out set: 15 PHI + 15 clean sentences (varied domains).
  G.  200-note generator eval set (seed 1337, 140 PHI / 60 clean) with gold spans.
  P.  ai4privacy/pii-masking-openpii-1m *validation* split (English): real PII docs with gold spans,
      plus their generic-replaced clean twins -> doc-level specificity on a real, unseen distribution.
  Q.  PubMedQA pqa_labeled abstracts (training used the pqa_artificial config) -> clean specificity.

Doc metrics: accuracy, recall, specificity, precision, AUROC, Brier, ECE(10 bins).
Span metrics: RedactX span head vs. real Microsoft Presidio (presidio_analyzer + spaCy en_core_web_lg),
scored with the repo's BenchmarkSuite.evaluate_spans (exact-match P/R/F1 + adjusted recall).

Usage:
    python validate_model.py --model-dir ./models/RedactX-v2
    python validate_model.py --model-dir ./models/RedactX-v2 --skip-presidio --n-ai4privacy 100
"""

import argparse
import json
import os
import statistics
import sys
import time

from redactx.data.contrastive_corpus import _as_list, clean_spans, replace_spans_with_generic

HELD_OUT_CLEAN = [
    "Cellular respiration yields ATP through oxidative phosphorylation in the inner mitochondrial membrane.",
    "Routine de-identified laboratory panel with negative infectious markers.",
    "Preheat the oven to 180 C and whisk two eggs with 200 g of flour until smooth.",
    "def add(a, b):\n    return a + b",
    "The quarterly report shows revenue grew 12 percent while operating costs fell slightly.",
    "Beta-blockers reduce mortality in heart failure with reduced ejection fraction according to multiple trials.",
    "Influenza vaccination is recommended annually for adults over 65 years of age.",
    "The patient tolerated the procedure well with no complications.",
    "Sepsis bundles should be initiated within one hour of recognition in the emergency department.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
    "Recommendation Class 1A: Quadruple therapy consisting of ARNI, beta-blocker, MRA, and SGLT2 inhibitor.",
    "Wound healing progressed normally; continue dressing changes twice daily.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Amoxicillin 500 mg three times daily for seven days is first-line for acute otitis media.",
    "The server restarted after the configuration change and all health checks passed.",
]
HELD_OUT_PHI = [
    "Please call Jennifer Okonkwo at 312-555-0198 about her biopsy results.",
    "Pt. Raj Malhotra, DOB 3/14/1962, seen today for knee pain.",
    "My SSN is 512-44-9087 and I live at 88 Birch Lane, Dayton OH.",
    "Email the discharge papers to luis.ferreira@gmail.com before Friday.",
    "Mrs. Agnieszka Wozniak was admitted to St. Luke's on June 3rd.",
    "Account 4417 1234 5678 9113 belongs to Thomas Reed of Boise.",
    "Contact the guardian, Mr. Ahmed Haddad, at (614) 555-2231.",
    "Medical record 00482913 for Hana Suzuki shows a penicillin allergy.",
    "Dr. Priya Raman reviewed the MRI for patient Carlos Mendez on 11/02/2025.",
    "Ship the CPAP machine to Grace Liu, 4120 Pine St Apt 7, Seattle WA 98103.",
    "Kevin O'Brien, age 47, reports chest pain since Tuesday.",
    "Insurance member ID XJ8820417 for Fatima Bello was denied.",
    "Patient Olumide Adebayo's mother can be reached at olu.mom@yahoo.com.",
    "Seen in clinic: Svetlana Morozova, 29 y/o, MRN 77310942.",
    "IP 10.44.18.201 logged into the portal as user mgarcia1987.",
]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(s / (len(pos) * len(neg)), 4)


def ece(probs, labels, bins=10):
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return round(err, 4)


def doc_metrics(probs, labels, threshold=0.5):
    tp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 1)
    fn = sum(1 for p, y in zip(probs, labels) if p < threshold and y == 1)
    fp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 0)
    tn = sum(1 for p, y in zip(probs, labels) if p < threshold and y == 0)
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    n = len(probs)
    return dict(
        n=n, positives=len(pos), negatives=len(neg), TP=tp, FN=fn, FP=fp, TN=tn,
        accuracy=round((tp + tn) / n, 4) if n else None,
        recall=round(tp / (tp + fn), 4) if (tp + fn) else None,
        specificity=round(tn / (tn + fp), 4) if (tn + fp) else None,
        precision=round(tp / (tp + fp), 4) if (tp + fp) else None,
        auroc=auroc(pos, neg),
        brier=round(sum((p - y) ** 2 for p, y in zip(probs, labels)) / n, 4) if n else None,
        ece_10bin=ece(probs, labels) if n else None,
        mean_p_phi_pos=round(statistics.mean(pos), 4) if pos else None,
        mean_p_phi_neg=round(statistics.mean(neg), 4) if neg else None,
        majority_baseline_accuracy=round(max(len(pos), len(neg)) / n, 4) if n else None,
    )


def _span_bounds(s):
    """(start, end) from a DetectedSpan / Finding object, a dict, or a (start, end[, ...]) tuple."""
    if isinstance(s, dict):
        return int(s["start"]), int(s["end"])
    if isinstance(s, (tuple, list)):
        return int(s[0]), int(s[1])
    return int(s.start), int(s.end)


def _char_set(spans):
    out = set()
    for s in spans:
        a, b = _span_bounds(s)
        out.update(range(a, b))
    return out


def char_metrics(docs, predict):
    """
    Character-level precision / recall / F1 of predicted PHI characters vs gold PHI characters, ignoring
    whitespace. Boundary-tolerant (a span found as "Jennifer" + "Okonkwo" still counts), so it is the
    right metric for unions of detectors. docs: [{"text", "gold": [span]}], predict(doc) -> [span].
    """
    tp = fp = fn = 0
    for d in docs:
        text = d["text"]
        g = {i for i in _char_set(d["gold"]) if i < len(text) and not text[i].isspace()}
        p = {i for i in _char_set(predict(d)) if i < len(text) and not text[i].isspace()}
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return dict(gold_chars=tp + fn, precision=round(prec, 4) if prec is not None else None,
                recall=round(rec, 4) if rec is not None else None, f1=round(f1, 4) if f1 is not None else None)


def per_category_recall(docs, predict):
    """
    Recall per HIPAA category (redactx.production.hipaa.to_category of the gold label):
      touched  fraction of gold spans that overlap at least one predicted span
      chars    fraction of gold non-whitespace characters covered by predictions
    """
    from redactx.production.hipaa import to_category
    acc = {}
    for d in docs:
        text = d["text"]
        preds = [_span_bounds(s) for s in predict(d)]
        pchars = _char_set(preds)
        for g in d["gold"]:
            a, b = _span_bounds(g)
            cat = to_category(g.get("category", "") if isinstance(g, dict) else g.category).value
            st = acc.setdefault(cat, {"n": 0, "touched": 0, "chars": 0, "covered": 0})
            st["n"] += 1
            st["touched"] += int(any(pa < b and pb > a for pa, pb in preds))
            idx = [i for i in range(a, b) if i < len(text) and not text[i].isspace()]
            st["chars"] += len(idx)
            st["covered"] += sum(1 for i in idx if i in pchars)
    return {c: dict(n=v["n"], touched_recall=round(v["touched"] / v["n"], 4),
                    char_recall=round(v["covered"] / v["chars"], 4) if v["chars"] else None)
            for c, v in sorted(acc.items(), key=lambda kv: -kv[1]["n"])}


def make_long_documents(docs, per_doc=5, sep="\n\n"):
    """Concatenate consecutive gold-annotated docs into long documents, shifting gold offsets."""
    out = []
    for i in range(0, len(docs) - per_doc + 1, per_doc):
        parts, gold, pos = [], [], 0
        for d in docs[i:i + per_doc]:
            for s, e, lab in d["spans"]:
                gold.append({"start": s + pos, "end": e + pos, "category": lab, "text": d["text"][s:e]})
            parts.append(d["text"])
            pos += len(d["text"]) + len(sep)
        out.append({"text": sep.join(parts), "gold": gold})
    return out


def load_ai4privacy_validation(n, token, max_chars, offset=0):
    """First n English docs (<= max_chars, with gold spans) after skipping `offset` qualifying docs."""
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="validation", token=token, streaming=True)
    docs = []
    seen = 0
    for row in ds:
        lang = (row.get("language") or "en").lower()
        if not lang.startswith("en"):
            continue
        text = row.get("source_text") or ""
        spans = []
        for m in _as_list(row.get("privacy_mask")):
            if isinstance(m, dict) and m.get("start") is not None and m.get("end") is not None:
                spans.append((int(m["start"]), int(m["end"]), str(m.get("label", "pii"))))
        spans = clean_spans(spans, len(text))
        if text and spans and len(text) <= max_chars:
            seen += 1
            if seen > offset:
                docs.append({"text": text, "spans": spans})
        if len(docs) >= n:
            break
    return docs


def load_pubmedqa_labeled(n, token, max_chars, offset=0):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", token=token, streaming=True)
    out = []
    seen = 0
    for row in ds:
        ctx = row.get("context") or {}
        t = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
        if t.strip():
            seen += 1
            if seen > offset:
                out.append(t.strip()[:max_chars])
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Held-out validation of a RedactX checkpoint")
    ap.add_argument("--model-dir", default="./models/RedactX-v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-ai4privacy", type=int, default=200)
    ap.add_argument("--n-pubmed", type=int, default=100)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--n-long", type=int, default=20, help="long documents (5 ai4privacy docs each) for set L")
    ap.add_argument("--skip-presidio", action="store_true")
    ap.add_argument("--out", default=None, help="Output JSON (default: <model-dir>/validation_results.json)")
    args = ap.parse_args()
    out = run_validation(args.model_dir, device=args.device, n_ai4privacy=args.n_ai4privacy,
                         n_pubmed=args.n_pubmed, max_chars=args.max_chars,
                         skip_presidio=args.skip_presidio, out_path=args.out, n_long=args.n_long)
    print_benchmark_table(out)


def run_validation(model_dir, device=None, n_ai4privacy=200, n_pubmed=100, max_chars=600,
                   skip_presidio=False, out_path=None, hf_token=None, n_long=20):
    """Runs all held-out benchmarks, saves <model_dir>/validation_results.json, returns the results dict."""
    args = argparse.Namespace(model_dir=model_dir, device=device, n_ai4privacy=n_ai4privacy, n_pubmed=n_pubmed,
                              max_chars=max_chars, skip_presidio=skip_presidio, out=out_path)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from redactx.models.openjev import OpenJevVaultGemmaEngine
    from redactx.data.generator import generate_redactx_corpus
    from redactx.evaluation.benchmark_suite import BenchmarkSuite
    from redactx.primitives import DetectedSpan

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from train import resolve_hf_token
            token = resolve_hf_token(None)
        except Exception:
            token = None

    engine = OpenJevVaultGemmaEngine.from_pretrained(args.model_dir, device=args.device)
    has_span_head = engine.span_locator is not None
    doc_thr = float(getattr(engine, "doc_threshold", 0.5))
    print(f"Loaded {args.model_dir} | span head: {'yes' if has_span_head else 'NO (spans will be empty)'} "
          f"| doc threshold {doc_thr:.4f} | span threshold {getattr(engine, 'span_threshold', 0.5):.4f}", flush=True)

    latencies = []

    def run(text):
        t0 = time.perf_counter()
        d = engine.evaluate_text(text[:args.max_chars])
        latencies.append((time.perf_counter() - t0) * 1000)
        return d

    OUT = {"model_dir": os.path.abspath(args.model_dir), "has_span_head": has_span_head,
           "doc_threshold": doc_thr, "span_threshold": float(getattr(engine, "span_threshold", 0.5)),
           "thresholds_calibrated_on": getattr(getattr(engine, "thresholds", None), "calibrated_on", None)}
    suite = BenchmarkSuite(redactx_engine=None)

    def span_score(docs, predict):
        etp = atp = fpc = gt = 0
        for text, gold, preds in ((d["text"], d["gold"], predict(d)) for d in docs):
            e, a, f = suite.evaluate_spans(preds, gold)
            etp += e; atp += a; fpc += f; gt += len(gold)
        prec = etp / max(etp + fpc, 1); rec = etp / max(gt, 1)
        return dict(gold_spans=gt, precision=round(prec, 4), recall=round(rec, 4),
                    adjusted_recall=round(atp / max(gt, 1), 4),
                    f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4))

    def add_char_and_category(prefix, docs):
        """Char-level P/R/F1 + per-HIPAA-category recall for RedactX, Presidio and their union.
        Clean docs stay in: predictions on them count as false positives."""
        predictors = {"redactx": lambda d: d["pred"]}
        if docs and "presidio" in docs[0]:
            predictors["presidio"] = lambda d: d["presidio"]
            predictors["hybrid_union"] = lambda d: list(d["pred"]) + list(d["presidio"])
        OUT[f"{prefix}_chars"] = {k: char_metrics(docs, p) for k, p in predictors.items()}
        OUT[f"{prefix}_per_category_recall"] = {k: per_category_recall(docs, p) for k, p in predictors.items()}

    analyzer = None
    if not args.skip_presidio:
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
        except Exception as e:
            OUT["presidio"] = f"unavailable: {type(e).__name__}: {e}"

    def presidio_spans(text):
        return [DetectedSpan(text=text[x.start:x.end], start=x.start, end=x.end,
                             category=x.entity_type, confidence=float(x.score))
                for x in analyzer.analyze(text=text, language="en")]

    # ---------------- H: handwritten held-out ----------------
    texts = HELD_OUT_PHI + HELD_OUT_CLEAN
    labels = [1] * len(HELD_OUT_PHI) + [0] * len(HELD_OUT_CLEAN)
    probs = [run(t).phi_probability for t in texts]
    OUT["H_handwritten_15_15"] = doc_metrics(probs, labels, doc_thr)
    print("H handwritten:", json.dumps(OUT["H_handwritten_15_15"]), flush=True)

    # ---------------- G: generator eval set with gold spans ----------------
    data = generate_redactx_corpus(num_samples=200, phi_ratio=0.70, seed=1337)
    g_docs, g_probs, g_labels = [], [], []
    for r in data:
        d = run(r["context"])
        g_probs.append(d.phi_probability)
        g_labels.append(1 if r["target"] == "Contains_PHI_PII" else 0)
        g_docs.append({"text": r["context"], "gold": r.get("spans", []), "pred": d.spans})
    OUT["G_generator_200_doc"] = doc_metrics(g_probs, g_labels, doc_thr)
    OUT["G_generator_200_spans_redactx"] = span_score(g_docs, lambda d: d["pred"])
    if analyzer:
        for d in g_docs:
            d["presidio"] = presidio_spans(d["text"])
        OUT["G_generator_200_spans_presidio"] = span_score(g_docs, lambda d: d["presidio"])
    add_char_and_category("G_generator_200", g_docs)
    print("G doc:", json.dumps(OUT["G_generator_200_doc"]), flush=True)
    print("G spans RedactX:", OUT["G_generator_200_spans_redactx"], flush=True)
    print("G spans Presidio:", OUT.get("G_generator_200_spans_presidio"), flush=True)
    print("G chars:", json.dumps(OUT.get("G_generator_200_chars")), flush=True)

    # ---------------- P: ai4privacy validation (real, unseen) ----------------
    a_docs = []
    try:
        a_docs = load_ai4privacy_validation(args.n_ai4privacy, token, args.max_chars)
        p_probs, p_labels, p_span_docs = [], [], []
        for doc in a_docs:
            text = doc["text"]
            d = run(text)
            p_probs.append(d.phi_probability); p_labels.append(1)
            gold = [{"start": s, "end": e, "category": lab, "text": text[s:e]} for s, e, lab in doc["spans"]]
            p_span_docs.append({"text": text, "gold": gold, "pred": d.spans})
            twin = replace_spans_with_generic(text, doc["spans"])
            p_probs.append(run(twin).phi_probability); p_labels.append(0)
        OUT["P_ai4privacy_val_doc_with_clean_twins"] = doc_metrics(p_probs, p_labels, doc_thr)
        OUT["P_ai4privacy_val_spans_redactx"] = span_score(p_span_docs, lambda d: d["pred"])
        if analyzer:
            for d in p_span_docs:
                d["presidio"] = presidio_spans(d["text"])
            OUT["P_ai4privacy_val_spans_presidio"] = span_score(p_span_docs, lambda d: d["presidio"])
        add_char_and_category("P_ai4privacy_val", p_span_docs)
    except Exception as e:
        OUT["P_ai4privacy"] = f"could not load: {type(e).__name__}: {e}"
    print("P doc:", json.dumps(OUT.get("P_ai4privacy_val_doc_with_clean_twins", OUT.get("P_ai4privacy"))), flush=True)
    print("P spans RedactX:", OUT.get("P_ai4privacy_val_spans_redactx"), flush=True)
    print("P spans Presidio:", OUT.get("P_ai4privacy_val_spans_presidio"), flush=True)
    print("P chars:", json.dumps(OUT.get("P_ai4privacy_val_chars")), flush=True)

    # ---------------- L: long documents (5 ai4privacy docs concatenated) ----------------
    if a_docs and n_long > 0:
        from redactx.production.detectors import RedactXDetector
        long_docs = make_long_documents(a_docs, per_doc=5)[:n_long]
        if long_docs:
            detector = RedactXDetector(engine, max_chars=args.max_chars, overlap=min(150, args.max_chars // 2))
            for d, det in zip(long_docs, detector.detect_many([d["text"] for d in long_docs])):
                d["pred"] = det.findings
                d["truncated"] = run(d["text"]).spans  # old behaviour: only the first max_chars are seen
            OUT["L_long_docs"] = {
                "n_docs": len(long_docs),
                "mean_chars": round(statistics.mean(len(d["text"]) for d in long_docs), 1),
                "chunked_redactx": char_metrics(long_docs, lambda d: d["pred"]),
                "single_window_truncated": char_metrics(long_docs, lambda d: d["truncated"]),
            }
            if analyzer:
                for d in long_docs:
                    d["presidio"] = presidio_spans(d["text"])
                OUT["L_long_docs"]["presidio"] = char_metrics(long_docs, lambda d: d["presidio"])
                OUT["L_long_docs"]["hybrid_union"] = char_metrics(long_docs, lambda d: list(d["pred"]) + d["presidio"])
            print("L long docs:", json.dumps(OUT["L_long_docs"]), flush=True)

    # ---------------- Q: PubMedQA labeled (clean, unseen config) ----------------
    try:
        q_texts = load_pubmedqa_labeled(args.n_pubmed, token, args.max_chars)
        q_probs = [run(t).phi_probability for t in q_texts]
        flagged = sum(p >= doc_thr for p in q_probs)
        OUT["Q_pubmedqa_labeled_clean"] = dict(n=len(q_probs), flagged_as_phi=flagged,
                                               specificity=round(1 - flagged / max(len(q_probs), 1), 4),
                                               mean_p_phi=round(statistics.mean(q_probs), 4) if q_probs else None)
    except Exception as e:
        OUT["Q_pubmedqa_labeled_clean"] = f"could not load: {type(e).__name__}: {e}"
    print("Q clean PubMed:", OUT["Q_pubmedqa_labeled_clean"], flush=True)

    lat = sorted(latencies[5:]) if len(latencies) > 10 else sorted(latencies)
    OUT["latency_ms_evaluate_text"] = dict(
        n=len(lat), median=round(statistics.median(lat), 1), p95=round(lat[int(0.95 * (len(lat) - 1))], 1),
        note="Wall time of engine.evaluate_text (tokenize + forward + span extraction), first 5 calls excluded.")
    print("Latency:", OUT["latency_ms_evaluate_text"], flush=True)

    out_path = args.out or os.path.join(args.model_dir, "validation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=2)
    print("Saved:", os.path.abspath(out_path))

    del engine
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return OUT


# Measured on 2026-10-06 for the previous checkpoint (models/RedactX, "v1", 60/20/20 recipe).
#   H.*            : handwritten 15+15 set (verify_redactx.py)
#   G doc metrics  : same 200 generator notes, seed 1337 (validate_all.py -> validation_results.json)
#   G span metrics : same 200 notes, from comparative_benchmark_results.json (v1 had an UNTRAINED span head)
# None = not measured on that set.
V1_BASELINE = {
    "H.specificity": 0.0, "H.auroc": 0.562,
    "G.accuracy": 0.775, "G.recall": 1.0, "G.specificity": 0.25, "G.auroc": 0.902,
    "G.brier": 0.199, "G.ece_10bin": 0.177,
    "G.span_precision": 0.784, "G.span_recall": 0.206, "G.span_f1": 0.326,
}


def print_benchmark_table(OUT):
    def get(section, key):
        v = OUT.get(section)
        return v.get(key) if isinstance(v, dict) else None

    def f(v):
        return "   -  " if v is None else f"{v:6.3f}"

    rows = []
    for tag, sec in [("H", "H_handwritten_15_15"), ("G", "G_generator_200_doc"),
                     ("P", "P_ai4privacy_val_doc_with_clean_twins")]:
        for k in ["accuracy", "recall", "specificity", "auroc", "brier", "ece_10bin"]:
            rows.append((f"{tag} doc {k}", get(sec, k), V1_BASELINE.get(f"{tag}.{k}"), None))
    rows.append(("Q PubMed clean specificity", get("Q_pubmedqa_labeled_clean", "specificity"), None, None))
    for tag, red, pre in [("G", "G_generator_200_spans_redactx", "G_generator_200_spans_presidio"),
                          ("P", "P_ai4privacy_val_spans_redactx", "P_ai4privacy_val_spans_presidio")]:
        for k in ["precision", "recall", "f1"]:
            rows.append((f"{tag} span {k}", get(red, k), V1_BASELINE.get(f"{tag}.span_{k}"), get(pre, k)))

    print("\n" + "=" * 78)
    print(" RedactX held-out benchmark (none of these sets are used in contrastive training)")
    print("=" * 78)
    print(f"{'metric':34s} {'this model':>11s} {'old v1':>9s} {'Presidio':>10s}")
    print("-" * 78)
    for name, new, old, pres in rows:
        print(f"{name:34s} {f(new):>11s} {f(old):>9s} {f(pres):>10s}")
    lat = OUT.get("latency_ms_evaluate_text", {})
    print("-" * 78)
    print(f"latency evaluate_text: median {lat.get('median')} ms, p95 {lat.get('p95')} ms")
    print("H = 15 PHI + 15 clean handwritten | G = 200 generator notes (140 PHI/60 clean)")
    print("P = ai4privacy validation (real PII) + generic-replaced clean twins | Q = PubMedQA pqa_labeled")
    print("Presidio column = real presidio_analyzer + spaCy en_core_web_lg, run live on the same docs.")
    print("old v1 column = numbers measured earlier for models/RedactX; '-' = not measured.")
    print(f"doc threshold used: {OUT.get('doc_threshold', 0.5)} "
          f"({OUT.get('thresholds_calibrated_on') or 'default, not calibrated'})")
    print("=" * 78)
    print_production_table(OUT)


def print_production_table(OUT):
    """Character-level detection quality per deployment mode, long documents, and per-HIPAA-category recall."""
    def f(v):
        return "   -  " if v is None else f"{v:6.3f}"

    print("\n" + "=" * 78)
    print(" Deployment modes: character-level PHI detection (whitespace ignored)")
    print("=" * 78)
    print(f"{'set / metric':34s} {'RedactX':>10s} {'Presidio':>10s} {'Hybrid':>10s}")
    print("-" * 78)
    for tag, key in [("G", "G_generator_200_chars"), ("P", "P_ai4privacy_val_chars"), ("L", "L_long_docs")]:
        sec = OUT.get(key)
        if not isinstance(sec, dict):
            continue
        rx = sec.get("redactx") or sec.get("chunked_redactx") or {}
        pr = sec.get("presidio") or {}
        hy = sec.get("hybrid_union") or {}
        for k in ["precision", "recall", "f1"]:
            print(f"{tag + ' char ' + k:34s} {f(rx.get(k)):>10s} {f(pr.get(k)):>10s} {f(hy.get(k)):>10s}")
    L = OUT.get("L_long_docs")
    if isinstance(L, dict):
        tr = L.get("single_window_truncated", {})
        print(f"{'L char recall, no chunking':34s} {f(tr.get('recall')):>10s}   (first window only: old behaviour)")
    print("-" * 78)
    print("G = 200 generator notes | P = ai4privacy validation | L = long docs (5 ai4privacy docs joined,"
          f" n={L.get('n_docs') if isinstance(L, dict) else 0}), RedactX chunked")
    print("Hybrid = union of RedactX and Presidio spans (the recommended deployment mode).")

    cats = OUT.get("P_ai4privacy_val_per_category_recall")
    if isinstance(cats, dict) and cats.get("redactx"):
        print("\n" + "=" * 78)
        print(" Per-category recall on P (ai4privacy validation), char-level coverage of gold spans")
        print("=" * 78)
        print(f"{'category':22s} {'n':>5s} {'RedactX':>10s} {'Presidio':>10s} {'Hybrid':>10s}")
        print("-" * 78)
        for cat, v in cats["redactx"].items():
            p = (cats.get("presidio") or {}).get(cat, {})
            h = (cats.get("hybrid_union") or {}).get(cat, {})
            print(f"{cat:22s} {v['n']:5d} {f(v.get('char_recall')):>10s} {f(p.get('char_recall')):>10s} "
                  f"{f(h.get('char_recall')):>10s}")
        print("-" * 78)
        print("Categories map ai4privacy labels onto HIPAA Safe Harbor identifiers (redactx/production/hipaa.py).")
        print("ai4privacy is general PII, not clinical notes: clinical recall must be measured on n2c2 2014.")
        print("=" * 78)


if __name__ == "__main__":
    main()
