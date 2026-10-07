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

ai4privacy/pii-masking-openpii-1m is deliberately NOT used here so it stays a held-out evaluation source.
The train/val split is done per source document, so a positive and its contrastive negative always land
in the same split.
"""

import ast
import os
import random
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


# ---------------------------------------------------------------- record building
def make_record(rec_id: str, text: str, spans: List[Tuple[int, int, str]], is_phi: bool,
                source: str, kind: str) -> Dict[str, Any]:
    prompt, _ = build_noul_prompt(text, DEFAULT_NOUL_QUESTION)
    return {
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


def load_contrastive_corpus(
    num_pii_docs: int = 2000,
    natural_negative_ratio: float = 0.5,
    nemotron_fraction: float = 0.7,
    max_chars: int = 600,
    val_fraction: float = 0.1,
    seed: int = 42,
    hf_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (train_records, val_records, stats).
    For num_pii_docs PII documents the corpus contains:
      num_pii_docs positives + num_pii_docs contrastive negatives + num_pii_docs*natural_negative_ratio natural negatives.
    """
    rng = random.Random(seed)
    token = hf_token or os.environ.get("HF_TOKEN")

    n_nemo = int(num_pii_docs * nemotron_fraction)
    nemo = load_nemotron_docs(n_nemo, token)
    gretel = load_gretel_docs(num_pii_docs - len(nemo), token)
    pii_docs = nemo + gretel
    if not pii_docs:
        raise RuntimeError("No PII documents could be loaded (Nemotron-PII and Gretel both unavailable). "
                           "Check network access and HF_TOKEN.")
    natural = load_natural_negatives(int(len(pii_docs) * natural_negative_ratio), token)

    rng.shuffle(pii_docs)
    rng.shuffle(natural)
    n_val_pii = max(1, int(len(pii_docs) * val_fraction))
    n_val_nat = max(1, int(len(natural) * val_fraction)) if natural else 0

    def expand(docs, split_name):
        recs = []
        for i, d in enumerate(docs):
            text, spans = truncate_with_spans(d["text"], d["spans"], max_chars)
            if not spans:
                continue
            recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_pos", text, spans, True,
                                    d["source"], "positive"))
            twin = replace_spans_with_generic(text, spans)[:max_chars]
            recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_neg", twin, [], False,
                                    d["source"], "contrastive_negative"))
        return recs

    def expand_natural(docs, split_name):
        recs = []
        for i, d in enumerate(docs):
            text = d["text"][:max_chars]
            recs.append(make_record(f"{split_name}_{d['source']}_{i:05d}_nat", text, [], False,
                                    d["source"], "natural_negative"))
        return recs

    val = expand(pii_docs[:n_val_pii], "val") + expand_natural(natural[:n_val_nat], "val")
    train = expand(pii_docs[n_val_pii:], "train") + expand_natural(natural[n_val_nat:], "train")
    rng.shuffle(train)
    rng.shuffle(val)

    def count(recs, key, value):
        return sum(1 for r in recs if r["metadata"].get(key) == value)

    stats = {
        "recipe": "contrastive (real PII docs + generic-replaced twins + natural clean text)",
        "pii_docs_nemotron": len(nemo),
        "pii_docs_gretel": len(gretel),
        "natural_negative_docs": len(natural),
        "natural_negative_sources": {s: sum(1 for d in natural if d["source"] == s)
                                     for s in sorted({d["source"] for d in natural})},
        "train_records": len(train),
        "val_records": len(val),
        "train_positives": count(train, "is_phi", True),
        "train_negatives": count(train, "is_phi", False),
        "val_positives": count(val, "is_phi", True),
        "val_negatives": count(val, "is_phi", False),
        "held_out_for_evaluation": "ai4privacy/pii-masking-openpii-1m (not used in training)",
    }
    return train, val, stats
