"""
Unit tests for clinical note evaluation and policy-aware redaction.
"""

import os
import pytest

from redactx.models.openjev import OpenJevVaultGemmaEngine
from redactx.production.detectors import RedactXDetector
from redactx.production.masking import mask_text


def test_clinical_notes_pipeline():
    model_dir = "./models/RedactX-v3"
    if not os.path.exists(model_dir):
        pytest.skip("Model weights not present. Skipping test.")

    engine = OpenJevVaultGemmaEngine.from_pretrained(model_dir)
    detector = RedactXDetector(engine)

    # 1. Clean clinical guideline hard negative
    clean_guideline = (
        "Acute myocardial infarction occurs when myocardial tissue dies due to prolonged ischemia. "
        "Initial pharmacotherapy includes aspirin 325 mg chewed immediately, followed by high-dose statin "
        "and sublingual nitroglycerin 0.4 mg every 5 minutes up to 3 doses if systolic BP > 90 mmHg."
    )
    res_clean = detector.detect(clean_guideline)
    assert res_clean.doc_score < 0.20 or not res_clean.contains_phi, "Clean clinical protocol flagged with high PHI score!"

    # 2. Real clinical note with identifiers
    note = (
        "DISCHARGE SUMMARY\n"
        "PATIENT NAME: Marcus Kowalski\n"
        "MRN: 948-21-4820 | ADMIT DATE: October 14, 2026\n"
        "ATTENDING PHYSICIAN: Dr. Sarah Vance, MD.\n"
        "HOSPITAL COURSE: 62-year-old male admitted with heart failure.\n"
        "FOLLOW-UP: Follow-up scheduled with cardiology in two weeks."
    )
    res_phi = detector.detect(note)
    assert res_phi.contains_phi, "Expected clinical note with MRN to be flagged as PHI!"

    # Safe Harbor redaction preserves male and cardiology
    redacted = mask_text(note, res_phi.findings, policy="safe_harbor")
    assert "Marcus Kowalski" not in redacted, "Patient name leaked!"
    assert "948-21-4820" not in redacted, "MRN leaked!"
    assert "cardiology" in redacted.lower(), "Cardiology specialty was mutilated!"
    assert "male" in redacted, "Gender descriptor was redacted!"
