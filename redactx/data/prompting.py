"""
Shared prompt construction for the RedactX Noul (binary PHI) decision.

Training and inference MUST build the prompt identically. The raw text is JSON-escaped
inside the prompt, which shifts character offsets (quotes, newlines, non-ASCII become
escape sequences). `build_noul_prompt` therefore also returns, for every character of the
prompt, the index of the raw-text character it came from (or -1 for template characters).
That map lets token offsets be translated back to raw-text offsets for span labels and
span extraction.
"""

import json
from typing import List, Tuple

DEFAULT_NOUL_QUESTION = "Contains HIPAA PHI or PII identifiers."
_PREFIX = '[STATE]: {"raw_text": "'


def build_noul_prompt(text: str, question: str = DEFAULT_NOUL_QUESTION) -> Tuple[str, List[int]]:
    """
    Returns (prompt, raw_index_of_prompt_char).

    The prompt is byte-identical to:
        f"[STATE]: {json.dumps({'raw_text': text})}\\n[DECISION]: {question}\\n[VERDICT]:"
    """
    parts = [_PREFIX]
    raw_index: List[int] = [-1] * len(_PREFIX)
    for i, ch in enumerate(text):
        esc = json.dumps(ch)[1:-1]  # per-character JSON escaping, same as json.dumps on the whole string
        parts.append(esc)
        raw_index.extend([i] * len(esc))
    suffix = f'"}}\n[DECISION]: {question}\n[VERDICT]:'
    parts.append(suffix)
    raw_index.extend([-1] * len(suffix))
    prompt = "".join(parts)
    return prompt, raw_index


def token_raw_ranges(offsets: List[Tuple[int, int]], raw_index: List[int]) -> List[Tuple[int, int]]:
    """
    Maps each token's (start, end) prompt-character offsets to a (raw_start, raw_end) range in the
    original text. Tokens that cover only template characters get (-1, -1).
    """
    out: List[Tuple[int, int]] = []
    n = len(raw_index)
    for s, e in offsets:
        idxs = [raw_index[k] for k in range(max(0, s), min(e, n)) if raw_index[k] >= 0]
        out.append((min(idxs), max(idxs) + 1) if idxs else (-1, -1))
    return out


def token_span_labels(
    offsets: List[Tuple[int, int]],
    raw_index: List[int],
    pii_spans: List[Tuple[int, int]],
) -> List[int]:
    """
    Per-token labels for the span head:
      1    token overlaps a PII character span of the raw text
      0    token lies in the raw text but outside every PII span
     -100  token is template / special (ignored by the loss)
    """
    pii = set()
    for s, e in pii_spans:
        pii.update(range(s, e))
    labels: List[int] = []
    for rs, re_ in token_raw_ranges(offsets, raw_index):
        if rs < 0:
            labels.append(-100)
        else:
            labels.append(1 if any(k in pii for k in range(rs, re_)) else 0)
    return labels
