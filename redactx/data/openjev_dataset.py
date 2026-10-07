"""
OpenJev Calibrated Decision Dataset & Collation.
Generates Noul, Choice (with permutation copies), and Score records with soft target distributions.
Handles single-token candidate ID resolution and dynamic batch collation.
"""

import random
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from redactx.data.generator import (
    FIRST_NAMES, LAST_NAMES, PHYSICIAN_TITLES, FACILITIES,
    CITIES_STATES, STREETS, CONDITIONS, MEDICATIONS, LAB_DATA, HARD_NEGATIVE_TEMPLATES
)


def resolve_anchor_token(tokenizer: AutoTokenizer, token_str: str) -> int:
    """Extracts single token ID from vocabulary with space prefix fallback."""
    clean = token_str.strip()
    ids_space = tokenizer.encode(f" {clean}", add_special_tokens=False)
    if ids_space:
        return ids_space[-1]
    ids = tokenizer.encode(clean, add_special_tokens=False)
    return ids[-1]


def build_calibrated_decision_corpus(
    num_samples: int = 600,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Step 1: Construct the Calibrated Decision Dataset.
    Generates balanced records across the three Jev primitives with soft target distributions:
    - Noul: [P(false), P(true)] e.g. [0.05, 0.95]
    - Choice: [P(A), P(B), ...] with shuffled permutation copies to eliminate positional bias
    - Score: Ordinal distribution over discrete buckets (1 through 5)
    """
    random.seed(seed)
    records = []

    # Distribution split: 40% Noul, 40% Choice (with permutations), 20% Score
    noul_count = int(num_samples * 0.40)
    choice_count = int(num_samples * 0.40)
    score_count = num_samples - noul_count - choice_count

    # 1. Noul Records (Binary Hypothesis with Soft Target Distributions)
    for i in range(noul_count):
        is_phi = (i % 2 == 0)
        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        dr_name = f"{random.choice(PHYSICIAN_TITLES)} {random.choice(LAST_NAMES)}"
        facility = random.choice(FACILITIES)
        city, state, _ = random.choice(CITIES_STATES)
        date_str = f"0{random.randint(1, 9)}/{random.randint(10, 28)}/2026"
        mrn = f"MRN-{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        cond = random.choice(CONDITIONS)
        med = random.choice(MEDICATIONS)

        if is_phi:
            state = f"Patient {fname} {lname} presented to {facility} on {date_str} under {mrn} for {cond}. Prescribed {med}."
            target_dist = [0.02, 0.98]
        else:
            if random.random() < 0.5:
                state = random.choice(HARD_NEGATIVE_TEMPLATES)
            else:
                state = f"Clinical follow-up for chronic {cond}. Vital signs stable with {random.choice(LAB_DATA)}. Continue {med}."
            target_dist = [0.98, 0.02]

        prompt = (
            f"[STATE]: {{\"raw_text\": \"{state}\"}}\n"
            f"[DECISION]: Contains HIPAA PHI or PII identifiers.\n"
            f"[VERDICT]:"
        )

        records.append({
            "id": f"noul_{i:05d}",
            "primitive": "noul",
            "prompt": prompt,
            "alt_prompt": prompt,
            "is_reversed": False,
            "candidate_tokens": ["false", "true"],
            "target_dist": target_dist,
            "metadata": {"is_phi": is_phi, "state": state}
        })

    # 2. Choice Records (Categorical Routing with Shuffled Permutations)
    routing_templates = [
        ("Suspected high-risk PHI disclosure requiring automated redaction pipeline.",
         ["Redaction_Pipeline", "Direct_Ingestion", "Compliance_Audit"], [0.85, 0.05, 0.10]),
        ("Routine de-identified laboratory panel with negative infectious markers.",
         ["Direct_Ingestion", "Redaction_Pipeline", "Specialist_Triage"], [0.90, 0.05, 0.05]),
        ("Conflicting medical record transfer with unverified patient authorization.",
         ["Compliance_Audit", "Direct_Ingestion", "Redaction_Pipeline"], [0.80, 0.05, 0.15]),
        ("Acute ST-elevation myocardial infarction requiring emergency cardiac cath.",
         ["Emergency_Cardiology", "Primary_Care", "Outpatient_Rehab"], [0.92, 0.05, 0.03]),
        ("Routine diabetic retinopathy screening requested for outpatient clinic.",
         ["Outpatient_Ophthalmology", "Emergency_Cardiology", "Inpatient_ICU"], [0.88, 0.02, 0.10])
    ]

    for i in range(choice_count):
        ctx_desc, base_options, base_dist = random.choice(routing_templates)

        # Rotate options to ensure perfectly balanced target placement across A, B, and C
        shift = i % len(base_options)
        options = base_options[shift:] + base_options[:shift]
        dist = base_dist[shift:] + base_dist[:shift]

        # Permutation 1
        opts_str_1 = ", ".join([f"Option {chr(65+k)}: {options[k]}" for k in range(len(options))])
        prompt_1 = (
            f"[STATE]: {{\"raw_text\": \"{ctx_desc}\"}}\n"
            f"[QUESTION]: Categorize privacy domain.\n"
            f"[OPTIONS]: [{opts_str_1}]\n"
            f"[SELECTION]:"
        )

        # Permutation 2 (reversed option order to eliminate positional bias)
        rev_options = list(reversed(options))
        rev_dist = list(reversed(dist))
        opts_str_2 = ", ".join([f"Option {chr(65+k)}: {rev_options[k]}" for k in range(len(rev_options))])
        prompt_2 = (
            f"[STATE]: {{\"raw_text\": \"{ctx_desc}\"}}\n"
            f"[QUESTION]: Categorize privacy domain.\n"
            f"[OPTIONS]: [{opts_str_2}]\n"
            f"[SELECTION]:"
        )

        records.append({
            "id": f"choice_{i:05d}_p1",
            "primitive": "choice",
            "prompt": prompt_1,
            "alt_prompt": prompt_2,
            "is_reversed": True,
            "candidate_tokens": [chr(65+k) for k in range(len(options))],
            "target_dist": dist,
            "metadata": {"options": options, "state": ctx_desc}
        })

        records.append({
            "id": f"choice_{i:05d}_p2",
            "primitive": "choice",
            "prompt": prompt_2,
            "alt_prompt": prompt_1,
            "is_reversed": True,
            "candidate_tokens": [chr(65+k) for k in range(len(rev_options))],
            "target_dist": rev_dist,
            "metadata": {"options": rev_options, "is_permuted": True, "state": ctx_desc}
        })

    # 3. Score Records (Ordinal Distributions across Discrete Buckets 1 to 5)
    acuity_cases = [
        ("Cardiac arrest, CPR ongoing, asystole.", [0.01, 0.01, 0.03, 0.10, 0.85]),  # Level 5 critical
        ("Severe respiratory distress, SpO2 84%, gasping.", [0.01, 0.02, 0.07, 0.70, 0.20]),  # Level 4 high
        ("Moderate abdominal pain for 2 days, vital signs stable.", [0.05, 0.20, 0.65, 0.08, 0.02]),  # Level 3 moderate
        ("Mild wrist sprain sustained during basketball, full mobility.", [0.25, 0.65, 0.08, 0.01, 0.01]),  # Level 2 low
        ("Suture removal request, wound completely healed.", [0.90, 0.08, 0.01, 0.01, 0.00])  # Level 1 minimal
    ]

    for i in range(score_count):
        clinical_case, dist = random.choice(acuity_cases)
        # Add slight empirical rater agreement noise
        noise = [random.uniform(-0.01, 0.01) for _ in dist]
        noisy_dist = [max(0.001, p + n) for p, n in zip(dist, noise)]
        sum_p = sum(noisy_dist)
        normalized_dist = [round(p / sum_p, 4) for p in noisy_dist]
        normalized_dist[-1] = round(1.0 - sum(normalized_dist[:-1]), 4)

        prompt = (
            f"[STATE]: {{\"raw_text\": \"{clinical_case}\"}}\n"
            f"[QUESTION]: Rate the redaction urgency and privacy risk from 1 (Safe) to 5 (Critical Breach):\n"
            f"[OPTIONS]: 1, 2, 3, 4, 5\n"
            f"[SCORE]:"
        )
        records.append({
            "id": f"score_{i:05d}",
            "primitive": "score",
            "prompt": prompt,
            "alt_prompt": prompt,
            "is_reversed": False,
            "candidate_tokens": ["1", "2", "3", "4", "5"],
            "target_dist": normalized_dist,
            "metadata": {"case": clinical_case}
        })

    random.shuffle(records)
    return records


class OpenJevCalibratedDataset(Dataset):
    """
    PyTorch Dataset collating OpenJev decision records.
    Pre-resolves candidate anchor token IDs, aligns soft probability distributions,
    and supports permutation consistency pairs.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 256
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        candidate_tokens = item["candidate_tokens"]
        target_dist = item["target_dist"]
        candidate_ids = [resolve_anchor_token(self.tokenizer, tok) for tok in candidate_tokens]

        # Contrastive records carry raw text + PII char spans -> build prompt with offsets and span labels.
        if "text" in item and "pii_spans" in item:
            from redactx.data.prompting import build_noul_prompt, token_span_labels, token_span_weights
            text = item["text"]
            spans = [tuple(s) for s in item["pii_spans"]]
            span_w = list(item.get("pii_span_weights") or [1.0] * len(spans))
            while True:
                prompt, raw_index = build_noul_prompt(text)
                enc = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
                if enc.input_ids.size(1) <= self.max_length or len(text) < 50:
                    break
                # Shorten the text (never truncate the prompt: the last token must stay "[VERDICT]:")
                cut = int(len(text) * 0.8)
                text = text[:cut]
                kept = [((s, min(e, cut)), w) for (s, e), w in zip(spans, span_w) if s < cut]
                spans = [sp for sp, _ in kept]
                span_w = [w for _, w in kept]
            offsets = [tuple(o) for o in enc.offset_mapping[0].tolist()]
            labels = token_span_labels(offsets, raw_index, spans)
            input_ids = enc.input_ids.squeeze(0)
            attention_mask = enc.attention_mask.squeeze(0)
            out = {
                "id": item["id"],
                "primitive": item["primitive"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "alt_input_ids": input_ids,
                "alt_attention_mask": attention_mask,
                "is_reversed": False,
                "candidate_token_ids": torch.tensor(candidate_ids, dtype=torch.long),
                "target_dist": torch.tensor(target_dist, dtype=torch.float32),
                "token_span_labels": torch.tensor(labels, dtype=torch.long),
            }
            if "pii_span_weights" in item:
                out["token_span_weights"] = torch.tensor(
                    token_span_weights(offsets, raw_index, spans, span_w), dtype=torch.float32)
            return out

        prompt = item["prompt"]
        alt_prompt = item.get("alt_prompt", prompt)

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        alt_encoding = self.tokenizer(
            alt_prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "id": item["id"],
            "primitive": item["primitive"],
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "alt_input_ids": alt_encoding.input_ids.squeeze(0),
            "alt_attention_mask": alt_encoding.attention_mask.squeeze(0),
            "is_reversed": item.get("is_reversed", False),
            "candidate_token_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "target_dist": torch.tensor(target_dist, dtype=torch.float32)
        }


def collate_openjev_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads input_ids, alt_input_ids, attention_mask, candidate_token_ids, and target_dist.
    Provides boolean candidate_mask and is_reversed flag for permutation consistency loss.
    """
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_alt_seq_len = max(item["alt_input_ids"].size(0) for item in batch)
    max_candidates = max(item["candidate_token_ids"].size(0) for item in batch)

    batch_size = len(batch)
    padded_input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    padded_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    padded_alt_input_ids = torch.zeros(batch_size, max_alt_seq_len, dtype=torch.long)
    padded_alt_attention_mask = torch.zeros(batch_size, max_alt_seq_len, dtype=torch.long)

    padded_candidate_ids = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    padded_target_dist = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    is_reversed = torch.zeros(batch_size, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        padded_input_ids[i, :seq_len] = item["input_ids"]
        padded_attention_mask[i, :seq_len] = item["attention_mask"]

        alt_seq_len = item["alt_input_ids"].size(0)
        padded_alt_input_ids[i, :alt_seq_len] = item["alt_input_ids"]
        padded_alt_attention_mask[i, :alt_seq_len] = item["alt_attention_mask"]

        num_cands = item["candidate_token_ids"].size(0)
        padded_candidate_ids[i, :num_cands] = item["candidate_token_ids"]
        padded_target_dist[i, :num_cands] = item["target_dist"]
        candidate_mask[i, :num_cands] = True
        is_reversed[i] = bool(item.get("is_reversed", False))

    return_batch = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "alt_input_ids": padded_alt_input_ids,
        "alt_attention_mask": padded_alt_attention_mask,
        "candidate_token_ids": padded_candidate_ids,
        "target_dist": padded_target_dist,
        "candidate_mask": candidate_mask,
        "is_reversed": is_reversed
    }

    # Per-token span labels (contrastive recipe). Missing/padded positions = -100 (ignored).
    if any("token_span_labels" in item for item in batch):
        span_labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            lab = item.get("token_span_labels")
            if lab is not None:
                span_labels[i, :lab.size(0)] = lab
        return_batch["token_span_labels"] = span_labels
    if any("token_span_weights" in item for item in batch):
        span_weights = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        for i, item in enumerate(batch):
            w = item.get("token_span_weights")
            if w is not None:
                span_weights[i, :w.size(0)] = w
            elif item.get("token_span_labels") is not None:
                span_weights[i, :item["token_span_labels"].size(0)] = 1.0
        return_batch["token_span_weights"] = span_weights

    return return_batch
