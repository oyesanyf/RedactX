"""
Unified Dataset Loader for RedactX.
Loads clinical datasets from synthetic generators, local files (JSON, JSONL, CSV),
or external benchmarks (HuggingFace datasets, i2b2 format).
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional

from redactx.data.generator import generate_redactx_corpus

logger = logging.getLogger("redactx.data.loader")


def normalize_record(raw_record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalizes arbitrary clinical record into standardized Jev decision format."""
    text = (
        raw_record.get("context") or
        raw_record.get("text") or
        raw_record.get("note") or
        raw_record.get("content") or
        raw_record.get("document") or
        ""
    )

    raw_target = (
        raw_record.get("target") or
        raw_record.get("label") or
        raw_record.get("verdict") or
        raw_record.get("is_phi") or
        "Clean"
    )

    if str(raw_target).lower() in ["1", "true", "phi", "contains_phi_pii", "contains_phi", "positive"]:
        norm_target = "Contains_PHI_PII"
    else:
        norm_target = "Clean"

    spans = raw_record.get("spans") or raw_record.get("entities") or []
    norm_spans = []
    for s in spans:
        if isinstance(s, dict) and "text" in s:
            norm_spans.append({
                "text": s.get("text", ""),
                "category": s.get("category", s.get("type", "PHI")),
                "start": s.get("start", s.get("start_offset", 0)),
                "end": s.get("end", s.get("end_offset", 0)),
                "confidence": s.get("confidence", 1.0)
            })

    return {
        "id": raw_record.get("id", f"record_{idx:06d}"),
        "primitive": "noul",
        "context": text,
        "query": raw_record.get(
            "query",
            "Does this text contain protected health information or personal identifiers?"
        ),
        "options": ["Clean", "Contains_PHI_PII"],
        "target": norm_target,
        "spans": norm_spans
    }


def load_dataset_source(
    source: Optional[str] = None,
    num_samples: int = 500,
    phi_ratio: float = 0.50,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Loads dataset from:
    1. Local file path (.json, .jsonl, .csv)
    2. Hugging Face dataset identifier
    3. High-diversity synthetic HIPAA generator (default if source is None)
    """
    if source is None or source.lower() in ["synthetic", "gen", "default"]:
        logger.info(f"Generating {num_samples} diverse HIPAA Safe Harbor clinical records...")
        return generate_redactx_corpus(num_samples=num_samples, phi_ratio=phi_ratio, seed=seed)

    # 1. Local File Check
    if os.path.exists(source):
        logger.info(f"Loading local dataset from: {source}")
        records = []

        if source.endswith(".jsonl"):
            with open(source, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line:
                        records.append(normalize_record(json.loads(line), idx))

        elif source.endswith(".json"):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("records", data.get("data", []))
                for idx, item in enumerate(items):
                    records.append(normalize_record(item, idx))

        elif source.endswith(".csv"):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    records.append(normalize_record(row, idx))
        else:
            raise ValueError(f"Unsupported file format: {source}. Please use .json, .jsonl, or .csv")

        logger.info(f"Successfully loaded {len(records)} records from {source}")
        return records

    # 2. Try Hugging Face Hub Dataset
    try:
        from datasets import load_dataset
        logger.info(f"Attempting to download Hugging Face dataset: '{source}'...")
        hf_data = load_dataset(source, split="train")
        records = [normalize_record(item, idx) for idx, item in enumerate(hf_data)]
        logger.info(f"Loaded {len(records)} records from Hugging Face dataset: {source}")
        return records
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load dataset from '{source}'. It is neither an existing file path "
            f"nor a valid Hugging Face dataset ({e})."
        )
