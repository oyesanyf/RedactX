"""
Policy-aware text masking and redaction utilities for RedactX.

Implements standard redaction policies:
1. 'safe_harbor': HIPAA Privacy Rule Safe Harbor standard (45 CFR § 164.514(b)(2)).
   Preserves non-identifying clinical variables:
   - Ages <= 89 (only ages > 89 must be aggregated or redacted under § 164.514(b)(2)(i)(C)).
   - Clinical demographic descriptors (e.g. gender, race, marital status).
2. 'strict': Complete PII suppression. All detected entities are replaced with [CATEGORY].
"""

import re
from typing import Any, Sequence


def mask_text(
    text: str,
    findings: Sequence[Any],
    policy: str = "safe_harbor",
) -> str:
    """
    Replaces detected spans in `text` with [CATEGORY] tags from end to start.

    Args:
        text: Original raw input string.
        findings: Sequence of findings (each having start, end, category, text attributes).
        policy: 'safe_harbor' (default) or 'strict'.

    Returns:
        Redacted text string.
    """
    chars = list(text)
    # Sort findings in descending order of start index to preserve character offsets
    for f in sorted(findings, key=lambda x: x.start, reverse=True):
        cat = getattr(f, "category", "PII")
        cat_name = cat.name if hasattr(cat, "name") else str(cat).replace("Category.", "")
        finding_text = getattr(f, "text", text[f.start:f.end])

        if policy == "safe_harbor":
            # Under HIPAA Safe Harbor, ages <= 89 are non-identifying clinical variables
            if cat_name == "AGE":
                digits = re.findall(r"\d+", finding_text)
                if digits and int(digits[0]) <= 89:
                    continue
            # Demographic gender/race/status is not a Safe Harbor prohibited identifier
            if cat_name == "DEMOGRAPHIC":
                continue

        chars[f.start:f.end] = list(f"[{cat_name}]")

    return "".join(chars)
