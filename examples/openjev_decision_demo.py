"""
OpenJev Decision Client Demo:
Demonstrates Noul, Choice, and Score primitives with active re-reads on VaultGemma.

Run:
    python examples/openjev_decision_demo.py
"""

from redactx import OpenJev, DecisionRequest, QuestionPayload

print("Initializing OpenJev VaultGemma Decision Engine...")
engine = OpenJev()

state_payload = {
    "patient_id": "MRN-482-19-8041",
    "name": "Marcus Kowalski",
    "intake_date": "2026-04-12",
    "facility": "Bethesda Memorial Hospital",
    "vitals": {"bp": "128/82", "hr": 74, "spo2": 98},
    "note": "Patient admitted on 04/12/2026 with acute chest discomfort. Initiated Lisinopril 10mg."
}

# Construct multi-primitive OpenJev Decision Request
request = DecisionRequest(
    model="vaultgemma-openjev",
    state=state_payload,
    questions=[
        # 1. Noul Primitive (Binary Hypothesis)
        QuestionPayload(
            key="contains_phi",
            type="noul",
            question="Does this record contain personal identifiers or protected health information?"
        ),
        # 2. Choice Primitive (Constrained Categorical Routing)
        QuestionPayload(
            key="routing_service",
            type="choice",
            question="Route payload to appropriate downstream service:",
            options=["Redaction_Pipeline", "Clinical_Audit", "Direct_Ingestion"]
        ),
        # 3. Score Primitive (Ordinal Mathematical Expectation)
        QuestionPayload(
            key="privacy_risk",
            type="score",
            question="Rate privacy sensitivity risk from 1 (Low) to 5 (Critical):",
            scale_min=1,
            scale_max=5
        )
    ],
    samples=3,
    entropy_threshold=0.10  # Automatically triggers active re-reads if uncertain
)

print("\n--- Evaluating OpenJev Decisions (Single-Pass Terminal Logit Extraction) ---")
response = engine.evaluate_decision(request)

print(f"Model: {response.model}")
print(f"Total Execution Latency: {response.latency_ms:.2f} ms")

print("\nAnswers:")
for key, ans in response.answers.items():
    print(f"\n[{key.upper()}] (Type: {ans.type})")
    if ans.type == "noul":
        print(f"  P(true):        {ans.probability:.4f}")
        print(f"  Confidence:     {ans.confidence:.4f}")
        print(f"  Value:          {ans.value}")
        print(f"  Active Re-reads: {ans.re_reads_executed}")
    elif ans.type == "choice":
        print(f"  Selection:      {ans.selection}")
        print(f"  Distribution:   {ans.distribution}")
        print(f"  Confidence:     {ans.confidence:.4f}")
        print(f"  Active Re-reads: {ans.re_reads_executed}")
    elif ans.type == "score":
        print(f"  Expected Value: {ans.expected_value:.3f} / 5.0")
        print(f"  Distribution:   {ans.distribution}")
        print(f"  Confidence:     {ans.confidence:.4f}")
        print(f"  Active Re-reads: {ans.re_reads_executed}")

if response.spans:
    print(f"\nDetected Entity Spans ({len(response.spans)}):")
    for s in response.spans:
        print(f"  - [{s.category}] \"{s.text}\" (chars {s.start}:{s.end}, conf {s.confidence:.2f})")
