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


def load_ai4privacy_validation(n, token, max_chars):
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="validation", token=token, streaming=True)
    docs = []
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
            docs.append({"text": text, "spans": spans})
        if len(docs) >= n:
            break
    return docs


def load_pubmedqa_labeled(n, token, max_chars):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", token=token, streaming=True)
    out = []
    for row in ds:
        ctx = row.get("context") or {}
        t = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
        if t.strip():
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
    ap.add_argument("--skip-presidio", action="store_true")
    ap.add_argument("--out", default=None, help="Output JSON (default: <model-dir>/validation_results.json)")
    args = ap.parse_args()
    out = run_validation(args.model_dir, device=args.device, n_ai4privacy=args.n_ai4privacy,
                         n_pubmed=args.n_pubmed, max_chars=args.max_chars,
                         skip_presidio=args.skip_presidio, out_path=args.out)
    print_benchmark_table(out)


def run_validation(model_dir, device=None, n_ai4privacy=200, n_pubmed=100, max_chars=600,
                   skip_presidio=False, out_path=None, hf_token=None):
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
    print(f"Loaded {args.model_dir} | span head: {'yes' if has_span_head else 'NO (spans will be empty)'}", flush=True)

    latencies = []

    def run(text):
        t0 = time.perf_counter()
        d = engine.evaluate_text(text[:args.max_chars])
        latencies.append((time.perf_counter() - t0) * 1000)
        return d

    OUT = {"model_dir": os.path.abspath(args.model_dir), "has_span_head": has_span_head}
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
    OUT["H_handwritten_15_15"] = doc_metrics(probs, labels)
    print("H handwritten:", json.dumps(OUT["H_handwritten_15_15"]), flush=True)

    # ---------------- G: generator eval set with gold spans ----------------
    data = generate_redactx_corpus(num_samples=200, phi_ratio=0.70, seed=1337)
    g_docs, g_probs, g_labels = [], [], []
    for r in data:
        d = run(r["context"])
        g_probs.append(d.phi_probability)
        g_labels.append(1 if r["target"] == "Contains_PHI_PII" else 0)
        g_docs.append({"text": r["context"], "gold": r.get("spans", []), "pred": d.spans})
    OUT["G_generator_200_doc"] = doc_metrics(g_probs, g_labels)
    OUT["G_generator_200_spans_redactx"] = span_score(g_docs, lambda d: d["pred"])
    if analyzer:
        OUT["G_generator_200_spans_presidio"] = span_score(g_docs, lambda d: presidio_spans(d["text"]))
    print("G doc:", json.dumps(OUT["G_generator_200_doc"]), flush=True)
    print("G spans RedactX:", OUT["G_generator_200_spans_redactx"], flush=True)
    print("G spans Presidio:", OUT.get("G_generator_200_spans_presidio"), flush=True)

    # ---------------- P: ai4privacy validation (real, unseen) ----------------
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
        OUT["P_ai4privacy_val_doc_with_clean_twins"] = doc_metrics(p_probs, p_labels)
        OUT["P_ai4privacy_val_spans_redactx"] = span_score(p_span_docs, lambda d: d["pred"])
        if analyzer:
            OUT["P_ai4privacy_val_spans_presidio"] = span_score(p_span_docs, lambda d: presidio_spans(d["text"]))
    except Exception as e:
        OUT["P_ai4privacy"] = f"could not load: {type(e).__name__}: {e}"
    print("P doc:", json.dumps(OUT.get("P_ai4privacy_val_doc_with_clean_twins", OUT.get("P_ai4privacy"))), flush=True)
    print("P spans RedactX:", OUT.get("P_ai4privacy_val_spans_redactx"), flush=True)
    print("P spans Presidio:", OUT.get("P_ai4privacy_val_spans_presidio"), flush=True)

    # ---------------- Q: PubMedQA labeled (clean, unseen config) ----------------
    try:
        q_texts = load_pubmedqa_labeled(args.n_pubmed, token, args.max_chars)
        q_probs = [run(t).phi_probability for t in q_texts]
        flagged = sum(p >= 0.5 for p in q_probs)
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
    print("=" * 78)


if __name__ == "__main__":
    main()
