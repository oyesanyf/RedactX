import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/RedactX")
model = AutoModelForCausalLM.from_pretrained("./models/RedactX", dtype=torch.float16, device_map="cuda")

def test_noul(text):
    escaped_json = json.dumps({"raw_text": text})
    prompt = (
        f"[STATE]: {escaped_json}\n"
        f"[DECISION]: Does this text contain Protected Health Information or Personal Identifiable Information?\n"
        f"[VERDICT]:"
    )
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0, -1, :]
    
    p_false = logits[1566].item()
    p_true = logits[1382].item()
    probs = F.softmax(torch.tensor([p_false, p_true]), dim=-1)
    
    print(f"\nText: {text}")
    print(f"  P(Clean/False): {probs[0]:.4f} | P(PHI/True): {probs[1]:.4f}")
    print(f"  Decision: {'CONTAINS PHI' if probs[1] > probs[0] else 'CLEAN'}")

# Test real PHI sample
test_noul("Admitted to General Hospital, record number 94851724.")
test_noul("Patient Elena Patel seen at Mercy Health on 06/15/2026.")
test_noul("Patient Marcus Kowalski admitted with MRN 123-45-6789.")

# Test hard negative clean samples
test_noul("Query latency measured at 42ms across cluster us-east-1.")
test_noul("Cellular respiration yields ATP through oxidative phosphorylation in the inner mitochondrial membrane.")
test_noul("Routine de-identified laboratory panel with negative infectious markers.")
