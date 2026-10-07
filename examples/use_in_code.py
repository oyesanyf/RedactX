"""
Example: Loading and using your trained RedactX decision model directly in code.

Run:
    python examples/use_in_code.py
"""

from redactx import RedactX

# 1. Load your trained model directly from the models folder
print("Loading trained RedactX model from models/redactx...")
model = RedactX.from_pretrained("models/redactx")

# 2. Inspect the model's trained benchmark scores and calibration metrics
print("\n--- Model Benchmark Scores & Training History ---")
if model.scores:
    metrics = model.scores.get("final_metrics", {})
    print(f"Validation Accuracy: {metrics.get('accuracy', 0):.2f}%")
    print(f"Validation F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"Validation Recall:   {metrics.get('recall', 0):.4f}")
    print(f"Validation ROC-AUC:  {metrics.get('auc', 0):.4f}")
    print(f"Calibrated Temp (T): {model.scores.get('calibrated_temperature', 1.0):.4f}")
else:
    print("No previous scores.json found in model directory.")

# 3. Use the model in code to evaluate clinical notes
clinical_samples = [
    # Positive PHI sample
    "Patient Marcus Kowalski presented on 04/12/2026 at Bethesda Memorial under MRN-482-19-8041.",
    # Clean negative sample (with dense clinical numbers / lab results)
    "Vital signs stable. Blood pressure 124/80 mmHg, heart rate 68 bpm. Commenced Metformin 500mg BID for secondary prevention."
]

print("\n--- Evaluating Payloads in Code ---")
for note in clinical_samples:
    decision = model.evaluate(note, decision_threshold=0.40)
    print(f"\nText: \"{note}\"")
    print(f"  Decision Verdict:     {decision.verdict}")
    print(f"  Calibrated PHI Score: {decision.phi_probability:.4f}")
    print(f"  Clean Score:          {decision.clean_probability:.4f}")
    print(f"  Latency:              {decision.latency_ms:.2f} ms")
    print(f"  Gate Passed:          {decision.passed_gate}")
    if decision.spans:
        print("  Detected PHI Spans:")
        for s in decision.spans:
            print(f"    * [{s.category}] \"{s.text}\" (chars {s.start}:{s.end})")
