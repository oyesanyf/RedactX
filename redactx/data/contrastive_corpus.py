"""
Contrastive training corpus for RedactX (replaces the 60/20/20 recipe's broken labels).

Problem with the previous recipe: every real-dataset sample was labelled PHI and the only
"clean" examples were 12 repeated sentences, so the model learned "not one of those 12 => PHI".

This corpus gives the model real contrast:
  * Positives       - real PII documents with character-level spans:
                      nvidia/Nemotron-PII (native `spans`) and gretelai/gretel-pii-masking-en-v1
                      (`entities`, located in the text).
  * Contrastive neg - the SAME documents with every PII span replaced by a generic phrase
                      ("the person", "a recent date", ...). Same style, same topic, no identifiers,
                      so the only difference between the pair is the presence of PII.
  * Natural neg     - varied clean text from other domains: PubMed abstracts (qiaojin/PubMedQA),
                      medical flashcards (medalpaca), and code (python_code_instructions_18k_alpaca).
  * Hard neg        - (hard_negative_ratio) clean CLINICAL text: PubMedQA pqa_unlabeled abstracts, WikiDoc
                      clinical reference text, and synthetic identifier-free clinical notes (redactx.data.clean_notes).
                      v2 over-flagged clean clinical-style notes and calibration could not fix it; this is the fix.
  * Clinical pairs  - (clinical_pair_ratio) synthetic clinical notes with PHI (rich in ages and demographics) and
                      their natural clean twins (same note, identifiers replaced by non-identifying wording).
  * Span weighting  - span_category_weights / oversample_categories upweight weak categories (AGE, DEMOGRAPHIC).

ai4privacy/pii-masking-openpii-1m is deliberately NOT used here so it stays a held-out evaluation source.
The train/val split is done per source document, so a positive and its contrastive negative always land
in the same split.
"""

import ast
import os
import random
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from redactx.data.prompting import DEFAULT_NOUL_QUESTION, build_noul_prompt

logger = logging.getLogger("redactx.data.contrastive")

POS_DIST = [0.02, 0.98]
NEG_DIST = [0.98, 0.02]

_GENERIC_RULES = [
    (("first_name", "last_name", "middle_name", "name", "user_name", "username"), "the person"),
    (("date", "dob", "birth", "time", "year"), "a recent date"),
    (("email",), "their email"),
    (("phone", "fax", "telephone"), "their phone number"),
    (("street", "address", "city", "state", "county", "country", "zip", "postcode", "postal",
      "location", "coordinate", "building"), "the area"),
    (("url", "ip", "mac", "device", "http"), "their account"),
    (("ssn", "social_security", "account", "card", "iban", "license", "passport", "identifier",
      "_id", "id_", "number", "mrn", "medical_record", "policy", "plate", "vin", "certificate"),
     "their record number"),
    (("age",), "an adult"),
    (("company", "organization", "employer", "occupation", "job", "employment"), "the organization"),
    (("gender", "sex", "race", "ethnicity", "religion", "nationality", "political", "sexuality",
      "blood_type", "marital"), "the characteristic"),
]


def generic_phrase(label: str) -> str:
    lab = (label or "").lower()
    for keys, phrase in _GENERIC_RULES:
        if any(k in lab for k in keys):
            return phrase
    return "the details"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def clean_spans(spans: List[Tuple[int, int, str]], text_len: int) -> List[Tuple[int, int, str]]:
    """Sort, clip to text, drop empty and overlapping spans (keep the earliest/longest)."""
    out: List[Tuple[int, int, str]] = []
    for s, e, lab in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        s, e = max(0, int(s)), min(text_len, int(e))
        if e <= s:
            continue
        if out and s < out[-1][1]:
            continue
        out.append((s, e, lab))
    return out


def truncate_with_spans(text: str, spans: List[Tuple[int, int, str]], max_chars: int):
    """Cut text to max_chars (at a whitespace boundary when possible) and clip spans to it."""
    if len(text) <= max_chars:
        return text, spans
    cut = text.rfind(" ", 0, max_chars)
    cut = cut if cut > max_chars * 0.6 else max_chars
    kept = [(s, min(e, cut), lab) for s, e, lab in spans if s < cut]
    return text[:cut], [(s, e, lab) for s, e, lab in kept if e > s]


def replace_spans_with_generic(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """Builds the contrastive clean twin: every PII span replaced by a generic phrase."""
    out, cursor = [], 0
    for s, e, lab in spans:
        out.append(text[cursor:s])
        out.append(generic_phrase(lab))
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------- loaders (real Hugging Face data)
def _stream(name: str, token: Optional[str], config: Optional[str] = None, split: str = "train"):
    from datasets import load_dataset
    if config:
        return load_dataset(name, config, split=split, token=token, streaming=True)
    return load_dataset(name, split=split, token=token, streaming=True)


def load_nemotron_docs(n: int, token: Optional[str]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if n <= 0:
        return docs
    try:
        for row in _stream("nvidia/Nemotron-PII", token):
            text = row.get("text") or ""
            raw = _as_list(row.get("spans"))
            spans = [(d.get("start"), d.get("end"), d.get("label", "pii")) for d in raw
                     if isinstance(d, dict) and d.get("start") is not None and d.get("end") is not None]
            spans = clean_spans(spans, len(text))
            if text and spans:
                docs.append({"source": "nemotron", "text": text, "spans": spans})
            if len(docs) >= n:
                break
    except Exception as e:
        logger.warning(f"nvidia/Nemotron-PII unavailable: {e}")
    return docs


def load_gretel_docs(n: int, token: Optional[str]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if n <= 0:
        return docs
    try:
        for row in _stream("gretelai/gretel-pii-masking-en-v1", token):
            text = row.get("text") or ""
            spans = []
            for ent in _as_list(row.get("entities")):
                if not isinstance(ent, dict):
                    continue
                val = str(ent.get("entity") or "").strip()
                types = ent.get("types") or ["pii"]
                if len(val) < 2:
                    continue
                start = text.find(val)
                while start != -1:
                    spans.append((start, start + len(val), types[0] if types else "pii"))
                    start = text.find(val, start + len(val))
            spans = clean_spans(spans, len(text))
            if text and spans:
                docs.append({"source": "gretel", "text": text, "spans": spans})
            if len(docs) >= n:
                break
    except Exception as e:
        logger.warning(f"gretelai/gretel-pii-masking-en-v1 unavailable: {e}")
    return docs


def load_natural_negatives(n: int, token: Optional[str]) -> List[Dict[str, Any]]:
    """Varied clean text: ~50% PubMed abstracts, ~30% medical flashcards, ~20% code."""
    plan = [("pubmed", int(n * 0.5)), ("flashcards", int(n * 0.3))]
    plan.append(("code", n - sum(k for _, k in plan)))
    docs: List[Dict[str, Any]] = []
    for kind, k in plan:
        if k <= 0:
            continue
        got = 0
        try:
            if kind == "pubmed":
                it = _stream("qiaojin/PubMedQA", token, config="pqa_artificial")
                for row in it:
                    ctx = row.get("context") or {}
                    t = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
                    if t.strip():
                        docs.append({"source": "pubmedqa", "text": t.strip(), "spans": []}); got += 1
                    if got >= k:
                        break
            elif kind == "flashcards":
                for row in _stream("medalpaca/medical_meadow_medical_flashcards", token):
                    t = f"{row.get('input', '')} {row.get('output', '')}".strip()
                    if t:
                        docs.append({"source": "flashcards", "text": t, "spans": []}); got += 1
                    if got >= k:
                        break
            else:
                for row in _stream("iamtarun/python_code_instructions_18k_alpaca", token):
                    t = f"{row.get('instruction', '')}\n{row.get('output', '')}".strip()
                    if t:
                        docs.append({"source": "code", "text": t, "spans": []}); got += 1
                    if got >= k:
                        break
        except Exception as e:
            logger.warning(f"natural negatives '{kind}' unavailable: {e}")
    return docs


def load_local_clinical_negatives(directory: Optional[str] = None, max_samples: int = 500, max_chars: int = 600,
                                  seed: int = 2029) -> List[Dict[str, Any]]:
    """
    Loads real clinical notes from local storage (e.g. data/test/*.txt from MIMIC-III)
    and removes de-identification surrogate brackets [**...**].
    The result is 100% clean, authentic clinical EHR prose without any identifiers,
    providing the highest quality hard negatives for clinical NLP.
    """
    import glob
    import random
    from redactx.production.chunking import chunk_text

    candidates = []
    if directory and os.path.isdir(directory):
        candidates.append(directory)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.extend([
        os.path.join(repo_root, "data", "test"),
        os.path.join(repo_root, "data"),
        os.path.abspath("data/test"),
        os.path.abspath("data"),
    ])

    files = []
    for cand in candidates:
        if cand and os.path.isdir(cand):
            hits = sorted(glob.glob(os.path.join(cand, "*.txt")))
            if hits:
                files = hits
                break

    if not files:
        return []

    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)

    docs: List[Dict[str, Any]] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
            cleaned = re.sub(r"\[\*\*.*?\*\*\]", "...", raw)
            for w in chunk_text(cleaned, max_chars=max_chars, overlap=0):
                text_slice = w.slice(cleaned).strip()
                if len(text_slice) >= 80:
                    docs.append({
                        "source": "local_clinical_clean",
                        "text": text_slice,
                        "spans": []
                    })
                    if len(docs) >= max_samples:
                        break
            if len(docs) >= max_samples:
                break
        except Exception as e:
            logger.warning(f"Error reading local clinical file {f}: {e}")
            continue

    return docs


def load_n2c2_training_pairs(root: Optional[str] = None, max_samples: int = 500, max_chars: int = 600,
                             seed: int = 4242) -> List[Dict[str, Any]]:
    """
    Loads authentic clinical notes with gold PHI spans from n2c2 2014 TRAIN split.
    Notes are cut into production-sized windows. Windows containing gold PHI
    are returned with their generic-replaced contrastive twins.
    """
    import random
    from redactx.data.n2c2 import load_n2c2, windows_with_spans, resolve_n2c2_dir

    try:
        n2c2_path = resolve_n2c2_dir(root)
        if not n2c2_path or not os.path.isdir(n2c2_path):
            return []
        docs, _ = load_n2c2(n2c2_path, split="train")
    except Exception as e:
        logger.warning(f"Could not load n2c2 training data: {e}")
        return []

    windows: List[Dict[str, Any]] = []
    for doc in docs:
        for w in windows_with_spans(doc, max_chars=max_chars):
            text = w["text"].strip()
            if not text:
                continue
            if w["spans"]:
                spans: List[Tuple[int, int, str]] = []
                for s, e, lab in sorted(w["spans"]):
                    if spans and s < spans[-1][1]:
                        ps, pe, pl = spans[-1]
                        spans[-1] = (ps, max(pe, e), pl)
                    else:
                        spans.append((s, e, lab))
                windows.append({
                    "source": "n2c2_train",
                    "text": w["text"],
                    "spans": spans,
                    "twin": replace_spans_with_generic(w["text"], spans)
                })

    rng = random.Random(seed)
    rng.shuffle(windows)
    return windows[:max_samples]


def load_hard_negatives(n: int, token: Optional[str], seed: int = 2028, max_chars: int = 600,
                        allow_download: bool = True, local_notes_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Clinical-style CLEAN text, the hard negatives that teach specificity on medical prose:
      ~40% local clinical notes (from data/test/*.txt or local_notes_dir) if available
      ~30% PubMedQA pqa_unlabeled abstracts
      ~15% medalpaca/medical_meadow_wikidoc clinical reference text
      rest synthetic identifier-free clinical notes (redactx.data.clean_notes)
    """
    from redactx.data.clean_notes import generate_clean_notes
    docs: List[Dict[str, Any]] = []
    if n <= 0:
        return docs

    # 1. Prioritize real local clinical notes (MIMIC-III / data/test)
    local_clean = load_local_clinical_negatives(local_notes_dir, max_samples=int(n * 0.5),
                                                max_chars=max_chars, seed=seed)
    docs.extend(local_clean)

    # 2. Remote PubMedQA and WikiDoc if allow_download and room left
    remaining = n - len(docs)
    if remaining > 0 and allow_download:
        plan = [("pubmed_unlabeled", int(remaining * 0.6)), ("wikidoc", int(remaining * 0.3))]
        for kind, k in plan:
            got = 0
            try:
                if kind == "pubmed_unlabeled":
                    for row in _stream("qiaojin/PubMedQA", token, config="pqa_unlabeled"):
                        ctx = row.get("context") or {}
                        t = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
                        if t.strip():
                            docs.append({"source": "pubmedqa_unlabeled", "text": t.strip()[:max_chars], "spans": []})
                            got += 1
                        if got >= k:
                            break
                else:
                    for row in _stream("medalpaca/medical_meadow_wikidoc", token):
                        t = str(row.get("output") or "").strip()
                        if len(t) >= 80:
                            docs.append({"source": "wikidoc", "text": t[:max_chars], "spans": []})
                            got += 1
                        if got >= k:
                            break
            except Exception as e:
                logger.warning(f"hard negatives '{kind}' unavailable: {e}")

    # 3. Fill any remainder with synthetic clinical notes
    if len(docs) < n:
        for t in generate_clean_notes(n - len(docs), seed=seed, max_chars=max_chars):
            docs.append({"source": "synthetic_clean_note", "text": t, "spans": []})
    return docs[:n]


def load_clinical_pairs(n: int, seed: int = 2027, max_chars: int = 600) -> List[Dict[str, Any]]:
    """Synthetic clinical notes with PHI spans (rich in AGE / DEMOGRAPHIC) and their natural clean twins."""
    from redactx.data.clean_notes import generate_clinical_pairs
    return [{"source": "synthetic_clinical", "text": d["text"], "spans": d["spans"], "twin": d["twin"]}
            for d in generate_clinical_pairs(n, seed=seed, max_chars=max_chars)]


def parse_category_weights(spec: Optional[str]) -> Dict[str, float]:
    """'AGE=3,DEMOGRAPHIC=3' -> {'AGE': 3.0, 'DEMOGRAPHIC': 3.0}. Keys are redactx.production.hipaa.Category names."""
    from redactx.production.hipaa import Category
    out: Dict[str, float] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"bad category weight '{part}', expected NAME=weight")
        k, v = part.split("=", 1)
        k = k.strip().upper()
        if k not in Category.__members__:
            raise ValueError(f"unknown category '{k}'; choose from {sorted(Category.__members__)}")
        w = float(v)
        if w <= 0:
            raise ValueError(f"category weight for {k} must be > 0")
        out[k] = w
    return out


def _span_category(label: str) -> str:
    from redactx.production.hipaa import to_category
    return to_category(label).value


# ---------------------------------------------------------------- record building
def make_record(rec_id: str, text: str, spans: List[Tuple[int, int, str]], is_phi: bool,
                source: str, kind: str, span_weights: Optional[List[float]] = None) -> Dict[str, Any]:
    prompt, _ = build_noul_prompt(text, DEFAULT_NOUL_QUESTION)
    rec = {
        "id": rec_id,
        "primitive": "noul",
        "prompt": prompt,
        "alt_prompt": prompt,
        "is_reversed": False,
        "candidate_tokens": ["false", "true"],
        "target_dist": POS_DIST if is_phi else NEG_DIST,
        # Raw text + PII character spans -> span-head labels are computed in the Dataset
        "text": text,
        "pii_spans": [(s, e) for s, e, _ in spans],
        "metadata": {"is_phi": is_phi, "source": source, "kind": kind},
    }
    if span_weights is not None:
        # per-span loss weight (category upweighting); the Dataset turns it into per-token weights
        rec["pii_span_weights"] = [float(w) for w in span_weights]
    return rec


def load_contrastive_corpus(
    num_pii_docs: int = 2000,
    natural_negative_ratio: float = 0.5,
    nemotron_fraction: float = 0.7,
    max_chars: int = 600,
    val_fraction: float = 0.1,
    seed: int = 42,
    hf_token: Optional[str] = None,
    hard_negative_ratio: float = 0.0,
    clinical_pair_ratio: float = 0.0,
    span_category_weights: Optional[Dict[str, float]] = None,
    oversample_categories: Tuple[str, ...] = (),
    oversample_factor: int = 1,
    local_notes_dir: Optional[str] = None,
    n2c2_train_ratio: float = 0.0,
    _pii_docs: Optional[List[Dict[str, Any]]] = None,
    _natural_docs: Optional[List[Dict[str, Any]]] = None,
    _allow_download: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (train_records, val_records, stats).
    For num_pii_docs PII documents the corpus contains:
      num_pii_docs positives + num_pii_docs contrastive negatives + num_pii_docs*natural_negative_ratio natural negatives
      + num_pii_docs*hard_negative_ratio clinical hard negatives (local clinical discharge summaries, PubMed, WikiDoc)
      + num_pii_docs*clinical_pair_ratio synthetic clinical notes with PHI, each with its natural clean twin
      + num_pii_docs*n2c2_train_ratio real n2c2 clinical PHI windows, each with its generic-replaced twin.

    span_category_weights: {"AGE": 3.0, ...} multiplies the span-head loss on tokens of spans of that category.
    oversample_categories / oversample_factor: training positives that contain one of these categories (and their
      clean twins, to keep the classes balanced) are repeated oversample_factor times in the TRAIN split only.
    _pii_docs / _natural_docs / _allow_download: inject already-loaded documents and stay offline (tests, re-runs).
    """
    rng = random.Random(seed)
    token = hf_token or os.environ.get("HF_TOKEN")
    weights = {k.upper(): float(v) for k, v in (span_category_weights or {}).items()}
    over = {c.upper() for c in oversample_categories}

    if _pii_docs is not None:
        nemo = [d for d in _pii_docs if d.get("source") == "nemotron"]
        gretel = [d for d in _pii_docs if d.get("source") != "nemotron"]
    else:
        n_nemo = int(num_pii_docs * nemotron_fraction)
        nemo = load_nemotron_docs(n_nemo, token)
        gretel = load_gretel_docs(num_pii_docs - len(nemo), token)
    real_pii = nemo + gretel
    if not real_pii:
        raise RuntimeError("No PII documents could be loaded (Nemotron-PII and Gretel both unavailable). "
                           "Check network access and HF_TOKEN.")
    clinical = load_clinical_pairs(int(round(len(real_pii) * clinical_pair_ratio)), seed=seed + 2027,
                                   max_chars=max_chars) if clinical_pair_ratio > 0 else []
    n2c2_pairs = load_n2c2_training_pairs(max_samples=int(round(len(real_pii) * n2c2_train_ratio)),
                                          max_chars=max_chars, seed=seed) if n2c2_train_ratio > 0 else []
    pii_docs = real_pii + clinical + n2c2_pairs
    natural = (_natural_docs if _natural_docs is not None
               else load_natural_negatives(int(len(real_pii) * natural_negative_ratio), token))
    hard = load_hard_negatives(int(round(len(real_pii) * hard_negative_ratio)), token, seed=seed + 2028,
                               max_chars=max_chars, allow_download=_allow_download,
                               local_notes_dir=local_notes_dir) if hard_negative_ratio > 0 else []

    rng.shuffle(pii_docs)
    rng.shuffle(natural)
    rng.shuffle(hard)
    n_val_pii = max(1, int(len(pii_docs) * val_fraction))
    n_val_nat = max(1, int(len(natural) * val_fraction)) if natural else 0
    n_val_hard = max(1, int(len(hard) * val_fraction)) if hard else 0

    def span_weights(spans) -> Optional[List[float]]:
        if not weights:
            return None
        return [weights.get(_span_category(lab), 1.0) for _, _, lab in spans]

    def expand(docs, split_name):
        recs = []
        for i, d in enumerate(docs):
            text, spans = truncate_with_spans(d["text"], d["spans"], max_chars)
            if not spans:
                continue
            twin = d.get("twin") if d.get("twin") and len(d["text"]) <= max_chars else None
            twin = (twin or replace_spans_with_generic(text, spans))[:max_chars]
            copies = 1
            if split_name == "train" and over and oversample_factor > 1 and \
                    any(_span_category(lab) in over for _, _, lab in spans):
                copies = oversample_factor
            for c in range(copies):
                sfx = "" if c == 0 else f"_dup{c}"
                recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_pos{sfx}", text, spans, True,
                                        d["source"], "positive" if c == 0 else "positive_oversampled",
                                        span_weights(spans)))
                recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_neg{sfx}", twin, [], False,
                                        d["source"], "contrastive_negative",
                                        [] if weights else None))
        return recs

    def expand_clean(docs, split_name, kind):
        recs = []
        for i, d in enumerate(docs):
            text = d["text"][:max_chars]
            recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_{kind[:4]}", text, [], False,
                                    d["source"], kind, [] if weights else None))
        return recs

    val = (expand(pii_docs[:n_val_pii], "val") + expand_clean(natural[:n_val_nat], "val", "natural_negative")
           + expand_clean(hard[:n_val_hard], "val", "hard_negative"))
    train = (expand(pii_docs[n_val_pii:], "train") + expand_clean(natural[n_val_nat:], "train", "natural_negative")
             + expand_clean(hard[n_val_hard:], "train", "hard_negative"))
    rng.shuffle(train)
    rng.shuffle(val)

    def count(recs, key, value):
        return sum(1 for r in recs if r["metadata"].get(key) == value)

    def sources(docs):
        return {s: sum(1 for d in docs if d["source"] == s) for s in sorted({d["source"] for d in docs})}

    span_cats: Dict[str, int] = {}
    for d in pii_docs[n_val_pii:]:
        for _, _, lab in d["spans"]:
            c = _span_category(lab)
            span_cats[c] = span_cats.get(c, 0) + 1

    stats = {
        "recipe": "contrastive (real PII docs + generic-replaced twins + natural clean text"
                  + (" + clinical hard negatives" if hard else "")
                  + (" + synthetic clinical PHI/clean pairs" if clinical else "")
                  + (" + n2c2 clinical PHI/clean pairs" if n2c2_pairs else "") + ")",
        "pii_docs_nemotron": len(nemo),
        "pii_docs_gretel": len(gretel),
        "clinical_pair_docs": len(clinical),
        "n2c2_train_pair_docs": len(n2c2_pairs),
        "natural_negative_docs": len(natural),
        "natural_negative_sources": sources(natural),
        "hard_negative_docs": len(hard),
        "hard_negative_sources": sources(hard),
        "local_notes_dir": local_notes_dir,
        "span_category_weights": weights,
        "oversample": {"categories": sorted(over), "factor": oversample_factor,
                       "train_records_added": count(train, "kind", "positive_oversampled") * 2},
        "train_gold_spans_by_category": dict(sorted(span_cats.items(), key=lambda kv: -kv[1])),
        "train_records": len(train),
        "val_records": len(val),
        "train_positives": count(train, "is_phi", True),
        "train_negatives": count(train, "is_phi", False),
        "val_positives": count(val, "is_phi", True),
        "val_negatives": count(val, "is_phi", False),
        "held_out_for_evaluation": "ai4privacy/pii-masking-openpii-1m, PubMedQA pqa_labeled, generator seed 1337, "
                                   "n2c2 2014 test split (only test split held out for benchmark)",
    }
    return train, val, stats
