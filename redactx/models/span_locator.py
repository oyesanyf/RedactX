"""
Token-level span attribution head for localizing exact PHI entities within clinical context.
"""

from typing import List, Dict, Any, Tuple
import re
import torch
import torch.nn as nn
from redactx.primitives import DetectedSpan


class TokenSpanLocator(nn.Module):
    """
    Token-level pointer and span attribution head.
    Predicts per-token PHI probabilities from sequence hidden states,
    enabling sub-token entity boundary localization without autoregressive generation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, sequence_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            token_logits: [batch_size, seq_len]
        """
        target_dtype = next(self.classifier.parameters()).dtype
        if sequence_hidden_states.dtype != target_dtype:
            sequence_hidden_states = sequence_hidden_states.to(dtype=target_dtype)
        logits = self.classifier(sequence_hidden_states).squeeze(-1)
        return logits

    @staticmethod
    def infer_category(span_text: str) -> str:
        """Heuristic category tagger for detected PHI spans."""
        text = span_text.strip()
        if re.search(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)?\d{2}\b", text):
            return "DATE"
        if re.search(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", text) or "MRN" in text.upper():
            return "IDENTIFIER"
        if re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text) or "@" in text:
            return "CONTACT"
        if any(w in text.lower() for w in ["hospital", "clinic", "center", "pavilion", "street", "avenue", "terrace", "drive"]):
            return "LOCATION"
        return "NAME"

    def extract_spans(
        self,
        token_probs: torch.Tensor,
        offset_mapping: List[Tuple[int, int]],
        full_prompt: str,
        context_text: str,
        threshold: float = 0.50
    ) -> List[DetectedSpan]:
        """
        Extracts contiguous character spans of detected PHI within the original context.
        """
        context_offset = full_prompt.find(context_text)
        if context_offset == -1:
            context_offset = 0
        context_end = context_offset + len(context_text)

        spans: List[DetectedSpan] = []
        in_span = False
        span_start = 0
        span_end = 0
        confidences = []

        probs_list = token_probs.tolist()

        for idx, (t_start, t_end) in enumerate(offset_mapping):
            if t_start == 0 and t_end == 0:
                continue
            # Restrict to context boundaries
            if t_start < context_offset or t_end > context_end:
                if in_span:
                    # Flush span
                    s_text = full_prompt[span_start:span_end].strip()
                    if s_text and len(s_text) > 1:
                        avg_conf = sum(confidences) / max(len(confidences), 1)
                        spans.append(DetectedSpan(
                            text=s_text,
                            start=max(0, span_start - context_offset),
                            end=max(0, span_end - context_offset),
                            category=self.infer_category(s_text),
                            confidence=round(avg_conf, 4)
                        ))
                    in_span = False
                    confidences = []
                continue

            prob = probs_list[idx]
            if prob >= threshold:
                if not in_span:
                    in_span = True
                    span_start = t_start
                    span_end = t_end
                    confidences = [prob]
                else:
                    span_end = max(span_end, t_end)
                    confidences.append(prob)
            else:
                if in_span:
                    s_text = full_prompt[span_start:span_end].strip()
                    if s_text and len(s_text) > 1:
                        avg_conf = sum(confidences) / max(len(confidences), 1)
                        spans.append(DetectedSpan(
                            text=s_text,
                            start=max(0, span_start - context_offset),
                            end=max(0, span_end - context_offset),
                            category=self.infer_category(s_text),
                            confidence=round(avg_conf, 4)
                        ))
                    in_span = False
                    confidences = []

        if in_span:
            s_text = full_prompt[span_start:span_end].strip()
            if s_text and len(s_text) > 1:
                avg_conf = sum(confidences) / max(len(confidences), 1)
                spans.append(DetectedSpan(
                    text=s_text,
                    start=max(0, span_start - context_offset),
                    end=max(0, span_end - context_offset),
                    category=self.infer_category(s_text),
                    confidence=round(avg_conf, 4)
                ))

        return spans

    def extract_raw_spans(
        self,
        token_probs: torch.Tensor,
        raw_ranges: List[Tuple[int, int]],
        raw_text: str,
        threshold: float = 0.50
    ) -> List[DetectedSpan]:
        """
        Extracts PHI spans in raw-text coordinates.

        raw_ranges[i] is the (raw_start, raw_end) range of token i in the ORIGINAL text, or (-1, -1) for
        template tokens (see redactx.data.prompting.token_raw_ranges). Consecutive tokens whose span-head
        probability is >= threshold are merged into one span. The category is a heuristic
        (`infer_category`); the head itself only predicts PHI vs. not-PHI per token.
        """
        probs_list = token_probs.float().tolist()
        spans: List[DetectedSpan] = []
        cur_start, cur_end, confs = -1, -1, []

        def flush():
            if cur_start < 0:
                return
            s, e = cur_start, cur_end
            while s < e and raw_text[s].isspace():
                s += 1
            while e > s and raw_text[e - 1].isspace():
                e -= 1
            if e - s > 1:
                seg = raw_text[s:e]
                spans.append(DetectedSpan(
                    text=seg, start=s, end=e,
                    category=self.infer_category(seg),
                    confidence=round(sum(confs) / len(confs), 4)
                ))

        for (rs, re_), prob in zip(raw_ranges, probs_list):
            if rs < 0 or prob < threshold:
                flush()
                cur_start, cur_end, confs = -1, -1, []
                continue
            if cur_start < 0:
                cur_start, cur_end, confs = rs, re_, [prob]
            else:
                cur_end = max(cur_end, re_)
                confs.append(prob)
        flush()
        return spans
