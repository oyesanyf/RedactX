"""
RedactX Fine-Tuning Privacy & Weight Memorization Audit Engine.

Evaluates post-fine-tuning privacy guarantees:
1. Membership Inference Attack (MIA) resistance (train member vs. test non-member loss gap).
2. Secret Sharer canary memorization and perplexity ratios (Carlini et al.).
3. LoRA adapter weight spectral norms and parameter gradient concentration.
4. Non-generative runtime containment invariants.
"""

from privacy_audit.mia_auditor import run_mia_audit
from privacy_audit.canary_auditor import run_canary_audit
from privacy_audit.weight_auditor import run_weight_audit

__all__ = ["run_mia_audit", "run_canary_audit", "run_weight_audit"]
