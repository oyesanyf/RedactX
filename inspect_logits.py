import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/RedactX")
model = AutoModelForCausalLM.from_pretrained("./models/RedactX", dtype=torch.float16, device_map="cuda")

def check_prompt(raw_text):
    prompt = f'[STATE]: {{"raw_text": "{raw_text}"}}\n[DECISION]: Contains HIPAA PHI or PII identifiers.\n[VERDICT]:'
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0, -1, :]
    
    # Check false (1566) vs true (1382)
    p_false = logits[1566].item()
    p_true = logits[1382].item()
    probs = F.softmax(torch.tensor([p_false, p_true]), dim=-1)
    
    # Top 5 tokens overall
    topk = torch.topk(logits, 5)
    top_toks = [(tokenizer.decode([idx.item()]), prob.item()) for idx, prob in zip(topk.indices, F.softmax(topk.values, dim=-1))]
    print(f"\nText: {raw_text[:65]}...")
    print(f"  ' false' (1566) logit: {p_false:.2f}, ' true' (1382) logit: {p_true:.2f}")
    print(f"  Candidate Softmax: P(false)={probs[0]:.4f}, P(true)={probs[1]:.4f}")
    print(f"  Top 5 overall tokens: {top_toks}")

check_prompt("Patient Jane Doe presented to Mercy Hospital on 04/12/2026 under MRN-482-91-3829 for Hypertension. Prescribed Lisinopril.")
check_prompt("Cellular respiration yields ATP through oxidative phosphorylation in the inner mitochondrial membrane.")
check_prompt("Routine de-identified laboratory panel with negative infectious markers.")
