"""Shared fixtures."""

import os

import pytest


@pytest.fixture(scope="session")
def gpt2_checkpoint(tmp_path_factory):
    """
    A real, exported RedactX-format checkpoint built from the gpt2 backbone with a freshly initialised span head.
    Used to exercise the engine / detector / server / calibration code paths on CPU. It is NOT a trained
    PHI detector, so tests using it check behaviour and invariants, never detection quality.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from redactx.models.span_locator import TokenSpanLocator
    d = tmp_path_factory.mktemp("gpt2_redactx")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    torch.manual_seed(0)
    head = TokenSpanLocator(hidden_dim=model.config.hidden_size)
    torch.save(head.state_dict(), os.path.join(d, "redactx_span_locator.pt"))
    return str(d)
