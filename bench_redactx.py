import time
import torch
from redactx.models.openjev import OpenJevVaultGemmaEngine
from redactx.primitives import DecisionRequest, QuestionPayload

print("Loading merged RedactX VaultGemma from ./models/RedactX onto CUDA...")
t0 = time.perf_counter()
engine = OpenJevVaultGemmaEngine.from_pretrained("./models/RedactX", device="cuda")
load_time = time.perf_counter() - t0
print(f"Model successfully loaded onto CUDA in {load_time:.2f}s!")

sample_phi = "Patient Jane Doe presented to Mercy Hospital on 04/12/2026 under MRN-482-91-3829 for Hypertension. Prescribed Lisinopril."
sample_clean = "Cellular respiration yields ATP through oxidative phosphorylation in the inner mitochondrial membrane."

req_phi = DecisionRequest(
    state=sample_phi,
    questions=[QuestionPayload(key="phi_check", type="noul", question="Contains HIPAA PHI or PII identifiers.")]
)

req_clean = DecisionRequest(
    state=sample_clean,
    questions=[QuestionPayload(key="clean_check", type="noul", question="Contains HIPAA PHI or PII identifiers.")]
)

# Warmup pass
_ = engine.evaluate_decision(req_phi)

# Benchmark PHI sample
t1 = time.perf_counter()
resp_phi = engine.evaluate_decision(req_phi)
lat_phi = (time.perf_counter() - t1) * 1000.0

# Benchmark Clean sample
t2 = time.perf_counter()
resp_clean = engine.evaluate_decision(req_clean)
lat_clean = (time.perf_counter() - t2) * 1000.0

ans_phi = resp_phi.answers["phi_check"]
ans_clean = resp_clean.answers["clean_check"]

print("\n" + "=" * 75)
print(" REDACTX OPENJEV LINE-RATE DECISION BENCHMARK (CUDA)")
print("=" * 75)
print(f"PHI Sample:   \"{sample_phi}\"")
print(f"  Decision:   {'CONTAINS PHI' if ans_phi.value else 'CLEAN'} (P(True) = {ans_phi.probability:.4f})")
print(f"  Confidence: {ans_phi.confidence:.4f}")
print(f"  Latency:    {lat_phi:.2f} ms")
print(f"  Spans ({len(resp_phi.spans)}):")
for s in resp_phi.spans:
    print(f"    - [{s.category}] \"{s.text}\" (conf: {s.confidence:.2f}, offset {s.start}:{s.end})")
print("-" * 75)
print(f"Clean Sample: \"{sample_clean}\"")
print(f"  Decision:   {'CONTAINS PHI' if ans_clean.value else 'CLEAN'} (P(True) = {ans_clean.probability:.4f})")
print(f"  Confidence: {ans_clean.confidence:.4f}")
print(f"  Latency:    {lat_clean:.2f} ms")
print(f"  Spans ({len(resp_clean.spans)}): {resp_clean.spans}")
print("=" * 75)
