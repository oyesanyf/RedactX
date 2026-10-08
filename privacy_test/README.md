# 🛡️ Privacy Leakage & Memorization Audit

This directory contains the privacy audit suite, reproduction scripts, and certified evaluation results for **RedactX (Google VaultGemma-1B)**.

## Audit Focus & Threat Models

1. **Canary & Training Data Memorization Extraction**:
   Evaluating whether adversarial queries can extract confidential pre-training or fine-tuning EHR sequences (Carlini et al. *Secret Sharer* metric).
2. **Adversarial Prompt Injections & System Overrides**:
   Testing whether injected jailbreaks within clinical notes cause the model to ignore boundaries or emit hallucinated PII.
3. **Substring Containment Invariant (Mathematical Zero-Leakage Guarantee)**:
   Proving that all detected spans strictly point to exact substrings within the user's input:
   $$\forall s \in \text{findings}, \quad s.\text{text} \equiv \text{input}[s.\text{start} : s.\text{end}]$$

## Generated Artifacts

- **[`privacy_audit_report.json`](privacy_audit_report.json)**: Machine-readable audit results with exact character slices, confidence scores, and adversarial test outcomes.
- **[`privacy_audit_report.md`](privacy_audit_report.md)**: Markdown audit report suitable for institutional review boards (IRBs) and compliance audits.

## How to Re-Run the Audit

Run the standalone audit runner:
```powershell
py -3.12 privacy_test/run_privacy_audit.py
```

Or execute via pytest:
```powershell
py -3.12 -m pytest tests/test_privacy_leakage.py -v
```
