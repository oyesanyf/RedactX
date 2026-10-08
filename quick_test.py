"""
RedactX Quick Test Sample Script
Tests RedactX-v3 on sample clinical and general PII text, displaying detected entities and the redacted text.

Usage:
    py -3.12 quick_test.py
"""

import sys
from redactx.models.openjev import OpenJevVaultGemmaEngine
from redactx.production.detectors import RedactXDetector

def mask_text(text: str, findings, policy: str = "strict") -> str:
    """
    Replaces detected spans with [CATEGORY] tags from end to start.
    
    policy:
      - "strict": masks all detected PII entities (including demographics and ages).
      - "safe_harbor": follows 45 CFR § 164.514(b)(2) — preserves ages <= 89 and
        demographic descriptors (e.g. 'male', 'female') as valuable clinical context.
    """
    import re
    chars = list(text)
    for f in sorted(findings, key=lambda x: x.start, reverse=True):
        cat_name = f.category.name if hasattr(f.category, "name") else str(f.category).replace("Category.", "")
        
        if policy == "safe_harbor":
            # Under HIPAA Safe Harbor, ages <= 89 are permitted clinical data
            if cat_name == "AGE":
                digits = re.findall(r"\d+", f.text)
                if digits and int(digits[0]) <= 89:
                    continue
            # Demographic gender/race is not a Safe Harbor prohibited identifier
            if cat_name == "DEMOGRAPHIC":
                continue

        chars[f.start:f.end] = list(f"[{cat_name}]")
    return "".join(chars)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print(" Loading RedactX-v3 Model (VaultGemma-1B + Dual-Head Locator)...")
    print("=" * 70)

    # 1. Load the fine-tuned engine (automatically loads calibrated thresholds)
    engine = OpenJevVaultGemmaEngine.from_pretrained("./models/RedactX-v3")
    
    # 2. Initialize the production detector (includes chunking & structured validators)
    detector = RedactXDetector(engine)
    print(f"[Ready] Model loaded. Doc Threshold: {detector.doc_threshold:.4f}\n")

    # 3. Test Sample 1: Authentic Complex Clinical Discharge Note
    clinical_sample = (
        "Patient Marcus Vance, a 58-year-old male, was admitted to Brigham and Women's Hospital "
        "on 09/24/2024 by Dr. Sarah Jenkins. Medical Record Number: 9812401. "
        "He reports persistent chest discomfort and shortness of breath. "
        "Discharged home to 742 Evergreen Terrace, Springfield, OR 97477. "
        "Primary contact telephone: (541) 555-0199. Follow-up scheduled with cardiology in two weeks."
    )

    # 4. Test Sample 2: Clean Non-PHI Clinical Text (Control)
    clean_sample = (
        "Acute myocardial infarction occurs when myocardial tissue dies due to prolonged ischemia. "
        "Initial pharmacotherapy includes aspirin 325 mg chewed immediately, followed by high-dose statin "
        "and sublingual nitroglycerin 0.4 mg every 5 minutes up to 3 doses if systolic BP > 90 mmHg."
    )

    samples = [
        ("Clinical Note (Contains PHI)", clinical_sample),
        ("Medical Textbook Abstract (Clean Control)", clean_sample),
    ]

    for title, text in samples:
        print("=" * 70)
        print(f" Sample: {title}")
        print("=" * 70)
        print(f"Input Text:\n\"{text}\"\n")

        # Run detection
        result = detector.detect(text)

        print(f"Decision:       {'[CONTAINS PHI]' if result.contains_phi else '[CLEAN / PASS]'}")
        print(f"Risk Score:     {result.doc_score:.4f} (Threshold: {detector.doc_threshold:.4f})")
        print(f"Entities Found: {len(result.findings)}")
        print("-" * 70)

        if result.findings:
            print("Detected Spans:")
            for f in result.findings:
                cat_name = f.category.name if hasattr(f.category, "name") else str(f.category).replace("Category.", "")
                matched_span = text[f.start:f.end]
                print(f"  * {cat_name:<16} | \"{matched_span}\" (chars {f.start:>3}:{f.end:<3}, score: {f.score:.2f})")
            
            print("-" * 70)
            print("Redacted Output (HIPAA Safe Harbor Mode - 45 CFR § 164.514(b)(2)):")
            print(f"\"{mask_text(text, result.findings, policy='safe_harbor')}\"\n")
            print("Redacted Output (Strict All-PII Mode):")
            print(f"\"{mask_text(text, result.findings, policy='strict')}\"")
        else:
            print("No PHI detected. Text preserved in full without alterations.")
        print()

if __name__ == "__main__":
    main()
