"""
RedactX Benchmark and Real-World Dataset Loaders.
Supports standard privacy de-identification benchmarks and real corpora:
1. N2C2 / i2b2 2014 Clinical De-identification Challenge (HIPAA Safe Harbor)
2. MIMIC-III / MIMIC-IV De-identified Clinical Notes (PhysioNet)
3. Gretel AI PII Masking Dataset (gretelai/gretel-pii-masking-en-v1)
4. AI4Privacy OpenPII (ai4privacy/pii-masking-openpii-1m)
5. NVIDIA Nemotron-PII (nvidia/Nemotron-PII)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def format_record_to_openjev(
    text: str,
    has_phi: bool,
    category: str = "PATIENT_NAME",
    severity: int = 4
) -> Dict[str, Any]:
    """
    Maps an arbitrary benchmark text snippet into an OpenJev decision record
    with rigid prompt anchors and calibrated soft target distributions.
    """
    noul_dist = [0.02, 0.98] if has_phi else [0.98, 0.02]

    # Dynamic choice options
    options = ["CLEAN_TEXT", "PATIENT_NAME", "SSN", "MEDICAL_RECORD_NUM"]
    if category not in options:
        options[1] = category
    idx = options.index(category) if has_phi else 0
    choice_dist = [0.05] * len(options)
    choice_dist[idx] = 0.85

    score_val = severity if has_phi else 1
    score_dist = [0.05] * 5
    score_dist[score_val - 1] = 0.80

    return {
        "text": text,
        "has_phi": has_phi,
        "noul": {
            "type": "noul",
            "state": {"raw_text": text},
            "question": "Does this text contain Protected Health Information or Personal Identifiable Information?",
            "target_dist": noul_dist
        },
        "choice": {
            "type": "choice",
            "state": {"raw_text": text},
            "question": "Select the primary sensitive entity class detected in the state payload:",
            "options": options,
            "target_dist": choice_dist
        },
        "score": {
            "type": "score",
            "state": {"raw_text": text},
            "question": "Rate the redaction urgency and privacy risk from 1 (Safe) to 5 (Critical Breach):",
            "options": ["1", "2", "3", "4", "5"],
            "target_dist": score_dist
        }
    }


def load_gretel_pii_benchmark(
    max_samples: int = 200,
    split: str = "train",
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Loads gretelai/gretel-pii-masking-en-v1 from Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    records = []
    try:
        from datasets import load_dataset
        ds = load_dataset("gretelai/gretel-pii-masking-en-v1", split=split, token=token, streaming=True)
        count = 0
        for row in ds:
            text = row.get("unmasked_text") or row.get("text", "")
            if not text:
                continue
            records.append(format_record_to_openjev(text, has_phi=True, category="PATIENT_NAME", severity=4))
            count += 1
            if count >= max_samples:
                break
        logger.info(f"Loaded {len(records)} samples from gretelai/gretel-pii-masking-en-v1")
    except Exception as e:
        logger.warning(f"Could not load gretelai/gretel-pii-masking-en-v1 ({e}). Returning empty list.")
    return records


def load_ai4privacy_openpii(
    max_samples: int = 200,
    split: str = "train",
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Loads ai4privacy/pii-masking-openpii-1m from Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    records = []
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split=split, token=token, streaming=True)
        count = 0
        for row in ds:
            text = row.get("source_text") or row.get("text", "")
            if not text:
                continue
            records.append(format_record_to_openjev(text, has_phi=True, category="PATIENT_NAME", severity=4))
            count += 1
            if count >= max_samples:
                break
        logger.info(f"Loaded {len(records)} samples from ai4privacy/pii-masking-openpii-1m")
    except Exception as e:
        logger.warning(f"Could not load ai4privacy/pii-masking-openpii-1m ({e}). Returning empty list.")
    return records


def load_nemotron_pii_benchmark(
    max_samples: int = 200,
    split: str = "train",
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Loads nvidia/Nemotron-PII from Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    records = []
    try:
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-PII", split=split, token=token, streaming=True)
        count = 0
        for row in ds:
            text = row.get("text") or row.get("content", "")
            if not text:
                continue
            records.append(format_record_to_openjev(text, has_phi=True, category="PATIENT_NAME", severity=4))
            count += 1
            if count >= max_samples:
                break
        logger.info(f"Loaded {len(records)} samples from nvidia/Nemotron-PII")
    except Exception as e:
        logger.warning(f"Could not load nvidia/Nemotron-PII ({e}). Returning empty list.")
    return records


def load_local_clinical_notes(directory_path: str, max_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Parses local clinical text notes (such as N2C2/i2b2 or MIMIC-III/IV notes).
    """
    records = []
    if not os.path.exists(directory_path):
        logger.warning(f"Directory {directory_path} not found.")
        return records

    files = [f for f in os.listdir(directory_path) if f.endswith((".txt", ".xml", ".json"))]
    for filename in files[:max_samples]:
        filepath = os.path.join(directory_path, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if content:
            # Check for surrogate PHI markers common in MIMIC or N2C2
            has_phi = any(marker in content for marker in ["[**", "MRN", "SSN", "DOB", "Dr."])
            records.append(format_record_to_openjev(content[:1000], has_phi=has_phi))

    return records


def benchmark_sample_to_decision_records(
    sample_id: str,
    text: str,
    has_phi: bool,
    category: str = "PATIENT_NAME",
    severity: int = 4
) -> List[Dict[str, Any]]:
    """
    Transforms an actual benchmark text sample into calibrated OpenJev decision records
    across Noul, Choice (with permutation pair), and Score primitives.
    """
    clean_text = text.strip()[:600]
    escaped_json = json.dumps({"raw_text": clean_text})
    records = []

    # 1. Noul Record (Binary hypothesis)
    noul_dist = [0.02, 0.98] if has_phi else [0.98, 0.02]
    noul_prompt = (
        f"[STATE]: {escaped_json}\n"
        f"[DECISION]: Contains HIPAA PHI or PII identifiers.\n"
        f"[VERDICT]:"
    )
    records.append({
        "id": f"{sample_id}_noul",
        "primitive": "noul",
        "prompt": noul_prompt,
        "alt_prompt": noul_prompt,
        "is_reversed": False,
        "candidate_tokens": ["false", "true"],
        "target_dist": noul_dist,
        "metadata": {"is_phi": has_phi, "category": category, "is_actual_benchmark": True}
    })

    # 2. Choice Records (Categorical routing across privacy classes with permutation pairing)
    base_options = ["CLEAN_TEXT", "PATIENT_NAME", "SSN", "MEDICAL_RECORD_NUM"]
    if category not in base_options:
        base_options[1] = category
    target_idx = base_options.index(category) if (has_phi and category in base_options) else 0

    dist = [0.05] * len(base_options)
    dist[target_idx] = 0.85

    opts_str_1 = ", ".join([f"Option {chr(65+k)}: {base_options[k]}" for k in range(len(base_options))])
    choice_prompt_1 = (
        f"[STATE]: {escaped_json}\n"
        f"[QUESTION]: Categorize privacy domain.\n"
        f"[OPTIONS]: [{opts_str_1}]\n"
        f"[SELECTION]:"
    )

    rev_options = list(reversed(base_options))
    rev_dist = list(reversed(dist))
    opts_str_2 = ", ".join([f"Option {chr(65+k)}: {rev_options[k]}" for k in range(len(rev_options))])
    choice_prompt_2 = (
        f"[STATE]: {escaped_json}\n"
        f"[QUESTION]: Categorize privacy domain.\n"
        f"[OPTIONS]: [{opts_str_2}]\n"
        f"[SELECTION]:"
    )

    records.append({
        "id": f"{sample_id}_choice_p1",
        "primitive": "choice",
        "prompt": choice_prompt_1,
        "alt_prompt": choice_prompt_2,
        "is_reversed": True,
        "candidate_tokens": [chr(65+k) for k in range(len(base_options))],
        "target_dist": dist,
        "metadata": {"options": base_options, "is_actual_benchmark": True}
    })

    records.append({
        "id": f"{sample_id}_choice_p2",
        "primitive": "choice",
        "prompt": choice_prompt_2,
        "alt_prompt": choice_prompt_1,
        "is_reversed": True,
        "candidate_tokens": [chr(65+k) for k in range(len(rev_options))],
        "target_dist": rev_dist,
        "metadata": {"options": rev_options, "is_permuted": True, "is_actual_benchmark": True}
    })

    # 3. Score Record (Ordinal risk rating 1 to 5)
    score_val = severity if has_phi else 1
    score_dist = [0.05] * 5
    score_dist[score_val - 1] = 0.80

    score_prompt = (
        f"[STATE]: {escaped_json}\n"
        f"[QUESTION]: Rate the redaction urgency and privacy risk from 1 (Safe) to 5 (Critical Breach):\n"
        f"[OPTIONS]: 1, 2, 3, 4, 5\n"
        f"[SCORE]:"
    )
    records.append({
        "id": f"{sample_id}_score",
        "primitive": "score",
        "prompt": score_prompt,
        "alt_prompt": score_prompt,
        "is_reversed": False,
        "candidate_tokens": ["1", "2", "3", "4", "5"],
        "target_dist": score_dist,
        "metadata": {"is_actual_benchmark": True}
    })

    return records


def load_hybrid_training_corpus(
    synthetic_samples: int = 300,
    real_benchmark_samples: int = 100,
    local_notes_dir: Optional[str] = None,
    seed: int = 42,
    hf_token: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Constructs a hybrid training corpus uniting both:
    1. Verified Real-World Benchmark Datasets (Gretel AI PII, OpenPII, Nemotron-PII, or local N2C2/MIMIC notes)
    2. Dynamic Synthetic Healthcare & Enterprise Corpora (Faker + dynamic context permutations)
    """
    import random
    from redactx.data.openjev_dataset import build_calibrated_decision_corpus

    random.seed(seed)
    combined: List[Dict[str, Any]] = []

    # 1. Generate Synthetic Stream
    synthetic_records = build_calibrated_decision_corpus(num_samples=synthetic_samples, seed=seed)
    combined.extend(synthetic_records)

    # 2. Ingest Actual Benchmark Stream
    benchmark_records: List[Dict[str, Any]] = []
    if real_benchmark_samples > 0:
        actual_raw_samples = []

        # Try local notes (N2C2 / MIMIC) if path provided
        if local_notes_dir and os.path.exists(local_notes_dir):
            local_items = load_local_clinical_notes(local_notes_dir, max_samples=real_benchmark_samples)
            actual_raw_samples.extend(local_items)

        # Ingest Gretel AI PII Masking Dataset
        if len(actual_raw_samples) < real_benchmark_samples:
            needed = real_benchmark_samples - len(actual_raw_samples)
            gretel_items = load_gretel_pii_benchmark(max_samples=needed, hf_token=hf_token)
            actual_raw_samples.extend(gretel_items)

        # Ingest AI4Privacy OpenPII Dataset
        if len(actual_raw_samples) < real_benchmark_samples:
            needed = real_benchmark_samples - len(actual_raw_samples)
            openpii_items = load_ai4privacy_openpii(max_samples=needed, hf_token=hf_token)
            actual_raw_samples.extend(openpii_items)

        # Format actual raw samples into calibrated decision records
        for idx, item in enumerate(actual_raw_samples):
            text = item.get("text", "")
            has_phi = item.get("has_phi", True)
            rec_group = benchmark_sample_to_decision_records(
                sample_id=f"actual_bench_{idx:05d}",
                text=text,
                has_phi=has_phi,
                category="PATIENT_NAME",
                severity=4 if has_phi else 1
            )
            benchmark_records.extend(rec_group)

    combined.extend(benchmark_records)
    random.shuffle(combined)

    stats = {
        "synthetic_records": len(synthetic_records),
        "actual_benchmark_records": len(benchmark_records),
        "total_records": len(combined),
        "uses_actual_data": len(benchmark_records) > 0,
        "uses_synthetic_data": len(synthetic_records) > 0
    }

    return combined, stats


# ==============================================================================
# Hard Negative Corpus (Medical Abstracts, Lab Panels, Telemetry, Zero PHI)
# ==============================================================================
HARD_NEGATIVE_CORPUS = [
    # Medical literature / PubMed style
    "Randomized double-blind trial evaluated Metformin 500mg BID vs placebo in type 2 diabetes with HbA1c 7.8% and eGFR 65 mL/min.",
    "Histopathological examination revealed invasive ductal carcinoma Nottingham Grade II with negative surgical margins and HER2 1+ status.",
    "Electrocardiogram demonstrates sinus rhythm at 72 bpm, PR interval 160 ms, QRS duration 88 ms, QTc 418 ms without ST-T wave abnormalities.",
    "Total cholesterol measured 195 mg/dL, HDL 48 mg/dL, LDL 112 mg/dL, triglycerides 175 mg/dL under Atorvastatin 20mg daily therapy.",
    "Computed tomography of the chest with IV contrast demonstrated no pulmonary embolism, normal cardiac silhouette, and clear lung parenchyma.",
    "Systematic review of 42 randomized clinical trials across PubMed and Embase evaluated mortality outcomes in acute sepsis guidelines.",
    "ICD-10-CM code I10 essential hypertension documented alongside E78.5 hyperlipidemia and K21.9 gastroesophageal reflux disease without esophagitis.",
    # System events, telemetry logs, API headers (zero personal identity markers)
    "2026-10-06T11:24:38.102Z [INFO] kernel: TCP cubic congestion control enabled, socket buffer size set to 262144 bytes.",
    "HTTP/1.1 200 OK Content-Type: application/json; charset=utf-8 X-Request-ID: 7f8a9b2c-e3d1-419b-a01e-6c2e3a89012f Latency: 38ms.",
    "Cluster autoscaler node group us-east-1a: 12 instances active, 0 pending, memory utilization 62.4%, CPU load 34.1%.",
    "PostgreSQL query executed in 14.2ms: SELECT count(*) FROM telemetry_events WHERE event_type = 'HEARTBEAT' AND timestamp >= NOW() - INTERVAL '1 hour';",
    "Network flow monitor: ingress 4.2 Gbps, egress 3.8 Gbps, zero dropped packets recorded across interface eth0 over 24-hour observation window."
]


def load_recipe_60_20_20_corpus(
    total_samples: int = 500,
    local_notes_dir: Optional[str] = None,
    seed: int = 42,
    hf_token: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Implements the Recommended 60/20/20 Hybrid Recipe for RedactX:
    - 60% Synthetic Persona Data (nvidia/Nemotron-PII + Faker structured payloads)
    - 20% Clinical Contexts (Gretel AI / OpenPII / local N2C2 / MIMIC notes)
    - 20% Hard Negative Samples (Medical literature, PubMed abstracts, telemetry logs)
    """
    import random
    from redactx.data.openjev_dataset import build_calibrated_decision_corpus

    random.seed(seed)
    num_persona = int(total_samples * 0.60)
    num_clinical = int(total_samples * 0.20)
    num_negatives = total_samples - num_persona - num_clinical

    # 1. 60% Persona Data (Nemotron-PII / Synthetic Personas)
    persona_records = []
    nemotron_items = load_nemotron_pii_benchmark(max_samples=num_persona, hf_token=hf_token)
    if nemotron_items:
        for idx, item in enumerate(nemotron_items):
            recs = benchmark_sample_to_decision_records(
                sample_id=f"nemotron_{idx:05d}",
                text=item.get("text", ""),
                has_phi=item.get("has_phi", True),
                category="PATIENT_NAME"
            )
            persona_records.extend(recs)

    needed_persona = max(0, num_persona - len(nemotron_items))
    if needed_persona > 0:
        synth_recs = build_calibrated_decision_corpus(num_samples=needed_persona, seed=seed)
        persona_records.extend(synth_recs)

    # 2. 20% Clinical Contexts (Gretel / OpenPII / N2C2 / MIMIC)
    clinical_records = []
    clinical_items = []
    if local_notes_dir and os.path.exists(local_notes_dir):
        clinical_items.extend(load_local_clinical_notes(local_notes_dir, max_samples=num_clinical))
    if len(clinical_items) < num_clinical:
        gretel_items = load_gretel_pii_benchmark(max_samples=num_clinical - len(clinical_items), hf_token=hf_token)
        clinical_items.extend(gretel_items)
    if len(clinical_items) < num_clinical:
        openpii_items = load_ai4privacy_openpii(max_samples=num_clinical - len(clinical_items), hf_token=hf_token)
        clinical_items.extend(openpii_items)

    for idx, item in enumerate(clinical_items):
        recs = benchmark_sample_to_decision_records(
            sample_id=f"clinical_{idx:05d}",
            text=item.get("text", ""),
            has_phi=item.get("has_phi", True),
            category="PATIENT_NAME",
            severity=4
        )
        clinical_records.extend(recs)

    # 3. 20% Hard Negative Samples (Medical abstracts, telemetry, zero PHI)
    negative_records = []
    for idx in range(num_negatives):
        neg_text = HARD_NEGATIVE_CORPUS[idx % len(HARD_NEGATIVE_CORPUS)]
        if idx >= len(HARD_NEGATIVE_CORPUS):
            neg_text = f"{neg_text} Batch sequence index {idx:04d} verified clean."
        recs = benchmark_sample_to_decision_records(
            sample_id=f"hard_neg_{idx:05d}",
            text=neg_text,
            has_phi=False,
            category="CLEAN_TEXT",
            severity=1
        )
        negative_records.extend(recs)

    combined = persona_records + clinical_records + negative_records
    random.shuffle(combined)

    stats = {
        "persona_records": len(persona_records),
        "clinical_records": len(clinical_records),
        "hard_negative_records": len(negative_records),
        "total_records": len(combined),
        "recipe": "60% Persona (Nemotron/Faker) / 20% Clinical (Gretel/OpenPII) / 20% Hard Negatives (PubMed/Telemetry)"
    }

    return combined, stats
