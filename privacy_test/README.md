# 🛡️ RedactX Runtime Privacy & Output Containment Audit

This directory contains the privacy audit suite, reproduction scripts, and empirical results for **RedactX (Google VaultGemma-1B)**.

## Audit Scope & Verification Goals

The audit tests the runtime boundaries of RedactX to verify that the deployed inference pipeline cannot be induced to emit external or memorized text:

1. **Substring Containment Invariant (`INV-1`)**:
   Verifies that all entity findings are strictly contiguous slices of the submitted note:
   $$\forall s \in \text{findings}, \quad s.\text{text} \equiv \text{input}[s.\text{start} : s.\text{end}]$$
2. **Adversarial Prompt Injection Resistance (`INV-2`)**:
   Tests whether jailbreak prefixes or extraction prompts can hijack the model or induce hallucinated outputs.
3. **Non-Generative Causal Lockdown (`INV-3`)**:
   Verifies that the inference path bypasses autoregressive decoding (`generate()`), querying only decision candidate logits and dense locator representations.

---

## ⚠️ Scope & Methodological Boundaries

* **Base Pre-training DP**: Google's $(\epsilon, \delta)$ differential privacy guarantee applies to the base pre-training of VaultGemma.
* **Fine-Tuning Assessment**: Fine-tuning with LoRA or full parameters does not automatically inherit formal differential privacy guarantees unless DP-SGD (Differential Privacy Stochastic Gradient Descent) is explicitly incorporated into the fine-tuning training loop.
* **Threat Model Coverage**: This audit confirms **runtime output containment** against black-box extraction and prompt injection. It does not evaluate white-box membership inference attacks (such as loss-ratio thresholding or gradient reconstruction against raw weight checkpoints).

---

## Generated Artifacts

- **[`privacy_audit_report.json`](privacy_audit_report.json)**: Machine-readable audit data detailing test documents, character spans, and probe outcomes.
- **[`privacy_audit_report.md`](privacy_audit_report.md)**: Structured Markdown audit report outlining methodology, results, and limitations.

## How to Re-Run the Audit

```powershell
py -3.12 privacy_test/run_privacy_audit.py
```
