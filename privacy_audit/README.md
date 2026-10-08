# 🛡️ RedactX Fine-Tuning Privacy & Weight Memorization Audit Suite

This directory contains the production-grade privacy auditing engine designed to evaluate fine-tuning privacy risks and weight memorization in **RedactX (Google VaultGemma-1B)**.

## Why This Audit is Necessary

1. **Pre-training vs. Fine-Tuning Boundary**:
   Google's $(\epsilon, \delta)$ differential privacy guarantee applies to the base pre-training of VaultGemma. Downstream task adaptation (LoRA fine-tuning on clinical EHR records) does not automatically inherit this guarantee unless DP-SGD is explicitly applied.
2. **Comprehensive Empirical Evaluation**:
   To assess whether the fine-tuned checkpoint leaks training information, this suite evaluates:
   - **Membership Inference Attacks (MIA)**: Measures the cross-entropy loss gap between member (training) notes and held-out non-member notes (Harvard n2c2 2014 test set), computing MIA ROC-AUC and True Positive Rate @ 1% False Positive Rate.
   - **Canary Memorization (Carlini Secret Sharer)**: Evaluates whether rare/unique synthetic identifiers (SSNs, MRNs, phone numbers) exhibit anomalous perplexity spikes relative to counterfactual baselines.
   - **Weight Spectral Norms**: Audits parameter distributions to verify absence of weight explosion or memorization energy hotspots.
   - **Runtime Output Containment**: Verifies the non-generative causal gate and exact substring containment invariants.

---

## Architecture & Modules

```
privacy_audit/
├── __init__.py           # Package exports
├── mia_auditor.py        # Membership Inference Attack (loss & confidence gap)
├── canary_auditor.py     # Carlini Secret Sharer canary exposure probe
├── weight_auditor.py     # Parameter Frobenius norms & weight distribution analysis
├── audit_runner.py       # Master audit execution orchestrator
├── reports/              # Auto-generated audit reports
│   ├── privacy_audit_summary.json
│   └── privacy_audit_report.md
└── README.md             # This guide
```

---

## How to Run the Audit

Run the full automated privacy audit on GPU/CPU:

```powershell
py -3.12 privacy_audit/audit_runner.py
```

### Optional Arguments:

```powershell
py -3.12 privacy_audit/audit_runner.py `
  --model-dir ./models/RedactX-v3 `
  --output-dir privacy_audit/reports `
  --samples 50
```

---

## Interpreting Audit Metrics

| Metric | Ideal / Secure Value | Risk Threshold | Explanation |
| :--- | :---: | :---: | :--- |
| **MIA ROC-AUC** | ~0.50 (Random Guess) | > 0.65 (Moderate) / > 0.75 (High) | Measures how accurately an attacker can infer if a patient's note was in the training set based on model loss. |
| **TPR @ 1% FPR** | $\le 0.01 - 0.05$ | > 0.10 | True Positive Rate at a strict 1% False Positive Rate. Measures high-confidence leakage. |
| **Max Canary Exposure Z** | < 3.0 | > 5.0 | Log-likelihood difference between secret canaries and random counterfactual strings. |
| **Substring Containment** | 100% Slice | < 100% | Guarantees all reported spans are exact slices of the user's input, preventing text generation. |
