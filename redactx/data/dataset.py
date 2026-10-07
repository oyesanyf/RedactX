"""
PyTorch Dataset collation and token alignment for RedactX.
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RedactXDataset(Dataset):
    """
    Collation dataset for RedactX bounded decision and span attribution tasks.
    Adheres to Jev prompt conventions and aligns character spans to token offsets.
    """

    LABEL_MAP = {"Clean": 0, "Contains_PHI_PII": 1}

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        include_span_labels: bool = True
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_span_labels = include_span_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        context = item["context"]
        query = item.get("query", "Does this text contain protected health information or personal identifiers?")

        # Jev Prompt Template
        formatted_prompt = f"Context: {context}\nQuery: {query}\nDecision:"

        encoding = self.tokenizer(
            formatted_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True if self.include_span_labels else False
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        label_idx = self.LABEL_MAP.get(item["target"], 0)

        data = {
            "id": item.get("id", f"sample_{idx}"),
            "context": context,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_idx, dtype=torch.long)
        }

        # Token-level span labeling (1 for PHI tokens inside context, 0 otherwise)
        if self.include_span_labels and hasattr(encoding, "offset_mapping"):
            offsets = encoding.offset_mapping.squeeze(0)
            token_span_labels = torch.zeros(self.max_length, dtype=torch.float32)
            
            # Find context offset start within formatted prompt
            context_prefix = "Context: "
            context_start_in_prompt = len(context_prefix)
            context_end_in_prompt = context_start_in_prompt + len(context)

            spans = item.get("spans", [])
            for span in spans:
                s_start = context_start_in_prompt + span["start"]
                s_end = context_start_in_prompt + span["end"]
                for t_idx, (t_start, t_end) in enumerate(offsets):
                    if t_start == 0 and t_end == 0:
                        continue  # Special or padding token
                    if max(t_start, s_start) < min(t_end, s_end):
                        token_span_labels[t_idx] = 1.0

            data["token_span_labels"] = token_span_labels

        return data
