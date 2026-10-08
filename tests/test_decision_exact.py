"""
Unit test for OpenJev raw causal decision logit extraction.
"""

import os
import json
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_openjev_raw_decision():
    model_dir = "./models/RedactX-v3"
    if not os.path.exists(model_dir):
        pytest.skip(f"Model directory '{model_dir}' not found. Skipping test.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for raw FP16 decision logit test.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cuda")

    phi_text = "Admitted to General Hospital, record number 94851724."
    escaped_json = json.dumps({"raw_text": phi_text})
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

    assert probs[1].item() > probs[0].item(), "Expected PHI detection probability to exceed clean probability."
