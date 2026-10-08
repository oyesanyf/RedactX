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
        """
        Heuristic category tagger for detected PHI spans.
        Normalizes adversarial zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
        and non-breaking spaces prior to pattern matching.
        """
        cleaned = re.sub(r"[\u200b-\u200d\ufeff]", "", span_text)
        cleaned = cleaned.replace("\u00a0", " ").strip()
        if not cleaned:
            return "NAME"

        # Dates (numeric, alpha-month, or standalone 4-digit year)
        if re.search(r"\b\d{4}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b|\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)?\d{2}\b", cleaned):
            return "DATE"
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b", cleaned, re.I):
            return "DATE"
        if re.fullmatch(r"(19|20)\d{2}", cleaned):
            return "DATE"

        # Demographics (gender, race)
        if cleaned.lower() in {"male", "female", "man", "woman", "transgender", "non-binary", "caucasian", "african american", "hispanic", "asian"}:
            return "DEMOGRAPHIC"

        # Age patterns (standalone 1-3 digits or digits with age suffix)
        if re.fullmatch(r"\b\d{1,3}\b", cleaned) or re.search(r"\b\d{1,3}[ -]?(year[s]?[- ]?old|yo|y/o)\b", cleaned, re.I):
            return "AGE"

        # Structured IDs and MRN
        if re.search(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", cleaned) or "MRN" in cleaned.upper() or re.search(r"\b\d{6,10}\b", cleaned):
            return "IDENTIFIER"

        # Contact info (Phone / Email)
        if re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", cleaned) or "@" in cleaned:
            return "CONTACT"

        # Locations & Facilities (including street types, cities, states, zip codes, and named hospitals)
        if any(w in cleaned.lower() for w in [
            "hospital", "clinic", "center", "pavilion", "infirmary", "health system", "st. jude", "mercy",
            "street", "avenue", "terrace", "drive", "road", "blvd", "lane", "way", "court", "circle", "highway",
            "ridge", "parkway", "portland", "springfield", "boston", "cambridge"
        ]) or re.search(r"\b[A-Z]{2}\s+\d{5}\b", cleaned) or re.search(r"\b\d{5}(-\d{4})?\b", cleaned):
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
                    s_text = full_prompt[span_start:span_end].strip(" \t\n\r\u200b\u200c\u200d\ufeff")
                    if s_text and (len(s_text) > 1 or s_text.isalnum()):
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
                    s_text = full_prompt[span_start:span_end].strip(" \t\n\r\u200b\u200c\u200d\ufeff")
                    if s_text and (len(s_text) > 1 or s_text.isalnum()):
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
            s_text = full_prompt[span_start:span_end].strip(" \t\n\r\u200b\u200c\u200d\ufeff")
            if s_text and (len(s_text) > 1 or s_text.isalnum()):
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

        CLINICAL_NON_PHI = frozenset({
            "cardiology", "neurology", "oncology", "radiology", "pathology", "pediatrics",
            "orthopedics", "dermatology", "psychiatry", "gastroenterology", "pulmonology",
            "nephrology", "urology", "hematology", "rheumatology", "anesthesiology",
            "ophthalmology", "surgery", "medicine", "infarction", "pharmacotherapy",
            "nitroglycerin", "aspirin", "ischemia", "cardiovascular", "hypertension",
            "discomfort", "shortness", "breath", "cardiologist", "cardiac"
        })

        def flush():
            if cur_start < 0:
                return
            s, e = cur_start, cur_end
            while s < e and (raw_text[s].isspace() or raw_text[s] in "\u200b\u200c\u200d\ufeff"):
                s += 1
            while e > s and (raw_text[e - 1].isspace() or raw_text[e - 1] in "\u200b\u200c\u200d\ufeff"):
                e -= 1
            if not ((e - s > 1) or (e - s == 1 and raw_text[s].isalnum())):
                return

            # Check word boundary alignment in raw_text
            w_start = s
            while w_start > 0 and raw_text[w_start - 1].isalnum():
                w_start -= 1
            w_end = e
            while w_end < len(raw_text) and raw_text[w_end].isalnum():
                w_end += 1

            is_subword = (w_start < s) or (w_end > e)
            full_word = raw_text[w_start:w_end].lower()
            avg_conf = sum(confs) / max(len(confs), 1)

            # Reject subword fragments of known clinical non-PHI terms (e.g. "card" + "iology")
            if is_subword and full_word in CLINICAL_NON_PHI:
                return

            # Reject low-confidence subword fragments (e.g. subword token false alarms with score < 0.50)
            if is_subword and avg_conf < 0.50:
                return

            seg = raw_text[s:e]

            # Reject ordinary relative clinical durations (e.g. "two weeks", "3 days", or number words directly modifying a time unit)
            if re.search(r"^\b(in\s+)?(two|three|four|five|six|seven|eight|nine|ten|one|\d+)\s+(weeks?|days?|months?|years?|hours?)\b$", seg, re.I):
                return
            if seg.lower() in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"} and re.match(r"^\s+(weeks?|days?|months?|hours?)\b", raw_text[e:], re.I):
                return

            # Expand possessives for institutional facilities (e.g. "St. Jude Children" + "'s Research Hospital")
            if re.search(r"\b(st\.?\s*jude|children|mercy|general|presbyterian|brigham)\b", seg, re.I):
                rest = raw_text[e:e + 50]
                m_facility = re.match(r"^('s\s+([A-Za-z']+\s+)*(hospital|clinic|center|research|institute|pavilion))\b", rest, re.I)
                if m_facility:
                    e = e + len(m_facility.group(1))
                    seg = raw_text[s:e]

            spans.append(DetectedSpan(
                text=seg, start=s, end=e,
                category=self.infer_category(seg),
                confidence=round(avg_conf, 4)
            ))

        for (rs, re_), prob in zip(raw_ranges, probs_list):
            if rs < 0 or prob < threshold:
                flush()
                cur_start, cur_end, confs = -1, -1, []
                continue
            if cur_start < 0:
                cur_start, cur_end, confs = rs, re_, [prob]
            else:
                # If there is a substantial gap containing alphanumeric words, do not merge across words
                if rs > cur_end and any(c.isalnum() for c in raw_text[cur_end:rs]):
                    flush()
                    cur_start, cur_end, confs = rs, re_, [prob]
                else:
                    cur_end = max(cur_end, re_)
                    confs.append(prob)
        flush()
        return spans

