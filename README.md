# RedactX: Differentially Private OpenJev Decision Engine for VaultGemma

RedactX implements the complete **OpenJev architectural blueprint** wrapped around Google's differentially private **VaultGemma 1B** autoregressive transformer.

Unlike generative LLMs that run autoregressive token generation with latency and hallucination risks, RedactX operates as a **Jev-like "System One" decision model**:
- **Prefix Conditioning with Terminal Logit Projection**: Bypasses the autoregressive generation loop by evaluating target token projections at the final sequence token in a single pass.
- **The Three Jev Primitives**:
  - **Noul**: Binary hypothesis evaluation returning calibrated $P(\text{true}) \in [0, 1]$ and epistemic confidence ($1 - \text{entropy}$).
  - **Choice**: Dynamic categorical routing with masked Softmax strictly normalized over runtime candidate option tokens.
  - **Score**: Ordinal Likert/rating scale evaluation computing the mathematical expectation $\mathbb{E}[\text{score}] = \sum m \cdot P(\text{level}_m)$.
- **Active Re-Reads & Evidential Calibration**: Measures normalized Shannon entropy $H(P)$; if uncertainty exceeds threshold ($H > 0.1$), the engine automatically executes $K$ active re-reads with Monte Carlo perturbation and averages the probability vectors.
- **RLCD Calibration Loss**: Fine-tunes VaultGemma with LoRA to eliminate categorical prior skews using a combined calibration loss:
  $$\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_1 \mathcal{L}_{\text{Brier}} + \lambda_2 \mathcal{L}_{\text{ECE}}$$
- **OpenJev Wire-Compatible REST API**: Serves the official `POST /v1/decision` endpoint matching TypeSafe Jev client SDKs.
- **Automated Clean Directory Policy**: The target `models/` directory is automatically wiped before every training run so no stale weights or artifacts linger.
- **Environment Token Detection**: Automatically picks up `HF_TOKEN` from your OS environment variables.

---

## Architecture Flow

```
                      ┌────────────────────────────────────────┐
                      │   State Payload (JSON, Text, Metrics)  │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │  VaultGemma Representation (Cached State) │
                    └─────────────────────┬─────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
    [Noul Question]                [Choice Question]               [Score Question]
"Does text contain PHI?"     "Route to: [A, B, C, D]"        "Risk Level: 1 to 5"
          │                               │                               │
          ▼                               ▼                               ▼
Single-token Sigmoid/Softmax    Dynamic Target Token Mask       Ordinal Softmax Projection
   P(true) in [0.0, 1.0]         P(C_i) normalized over C        Expected value + distribution
```

---

## 1. Quickstart: Training Your Own Model

Set your `HF_TOKEN` in your OS environment (PowerShell):
```powershell
$env:HF_TOKEN = "your_actual_hf_token"
```

Train VaultGemma 1B with RLCD calibration loss and automatic directory cleanup:
```powershell
python train.py --model google/vaultgemma-1b --force-vaultgemma --samples 1000 --epochs 3 --batch-size 8 --output-dir ./models/redactx
```

---

## 2. Using OpenJev in Python Code

```python
from redactx import OpenJev, DecisionRequest, QuestionPayload

# 1. Load your trained model directly from the models folder
engine = OpenJev.from_pretrained("models/redactx")

# 2. Define state payload
state_payload = {
    "patient_id": "MRN-482-19-8041",
    "name": "Marcus Kowalski",
    "intake_date": "2026-04-12",
    "facility": "Bethesda Memorial Hospital",
    "note": "Patient presented with severe chest pain. Administered Lisinopril 10mg."
}

# 3. Construct multi-slot OpenJev decision request
request = DecisionRequest(
    model="vaultgemma-openjev",
    state=state_payload,
    questions=[
        QuestionPayload(
            key="is_phi",
            type="noul",
            question="Does this record contain personal identifiers or PHI?"
        ),
        QuestionPayload(
            key="routing",
            type="choice",
            question="Route payload to appropriate downstream service:",
            options=["Redaction_Pipeline", "Clinical_Audit", "Direct_Ingestion"]
        ),
        QuestionPayload(
            key="acuity_score",
            type="score",
            question="Rate clinical acuity from 1 (Non-urgent) to 5 (Immediate):",
            scale_min=1,
            scale_max=5
        )
    ],
    samples=3,
    entropy_threshold=0.10  # Triggers automatic active re-reads if uncertain
)

# 4. Single forward-pass evaluation (no autoregressive generation)
response = engine.evaluate_decision(request)

print("Latency:", response.latency_ms, "ms")
for key, ans in response.answers.items():
    print(f"{key}: {ans}")
```

---

## 3. Serving Wire-Compatible REST API

Launch the FastAPI gateway:
```powershell
redactx serve --port 8080
```

### Official Wire Protocol: `POST /v1/decision`
Request:
```json
{
  "model": "vaultgemma-openjev",
  "state": "Patient Marcus Kowalski admitted on 04/12/2026 with acute chest pain.",
  "questions": [
    {
      "key": "has_phi",
      "type": "noul",
      "question": "Does this context contain protected health information?"
    },
    {
      "key": "category",
      "type": "choice",
      "question": "Primary clinical category:",
      "options": ["Cardiology", "Neurology", "Oncology"]
    },
    {
      "key": "urgency",
      "type": "score",
      "question": "Urgency rating",
      "scale_min": 1,
      "scale_max": 5
    }
  ],
  "samples": 3,
  "entropy_threshold": 0.10
}
```

Response:
```json
{
  "model": "vaultgemma-openjev (google/vaultgemma-1b)",
  "latency_ms": 28.4,
  "answers": {
    "has_phi": {
      "type": "noul",
      "probability": 0.9842,
      "confidence": 0.965,
      "value": true,
      "re_reads_executed": 0
    },
    "category": {
      "type": "choice",
      "selection": "Cardiology",
      "distribution": {
        "Cardiology": 0.9412,
        "Neurology": 0.0381,
        "Oncology": 0.0207
      },
      "confidence": 0.923,
      "re_reads_executed": 0
    },
    "urgency": {
      "type": "score",
      "expected_value": 4.12,
      "distribution": {"1": 0.02, "2": 0.05, "3": 0.15, "4": 0.35, "5": 0.43},
      "confidence": 0.88,
      "re_reads_executed": 0
    }
  }
}
```
