"""
Backbone loader supporting Google VaultGemma-1B with LoRA and lightweight local fallbacks.
"""

import os
import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from redactx.config import RedactXConfig

logger = logging.getLogger("redactx.backbone")


def is_vaultgemma_accessible(model_id: str) -> bool:
    """Verifies whether the target model can be accessed without auth failure."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)
        # If gated and no token set, likely inaccessible
        if getattr(info, "gated", False):
            token = os.getenv("HF_TOKEN")
            if not token:
                from huggingface_hub import get_token
                token = get_token()
            return bool(token)
        return True
    except Exception:
        return False


def load_backbone_and_tokenizer(
    config: RedactXConfig,
    device: torch.device
) -> Tuple[nn.Module, AutoTokenizer, int, str]:
    """
    Loads backbone transformer, detaches any generative head, and binds LoRA adapters.
    Returns:
        (backbone_model, tokenizer, hidden_dim, resolved_model_id)
    """
    target_id = config.base_model_id

    # Determine whether to use fallback
    if config.use_fallback_if_gated and "vaultgemma" in target_id.lower():
        if not is_vaultgemma_accessible(target_id):
            logger.warning(
                f"Model '{target_id}' is gated and no active HF_TOKEN was detected. "
                f"Switching to local development fallback: '{config.fallback_model_id}'."
            )
            target_id = config.fallback_model_id

    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None

    logger.info(f"Loading backbone tokenizer and model from '{target_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(target_id, token=token)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load Base Architecture
    is_causal = False
    try:
        model_config = AutoConfig.from_pretrained(target_id, token=token)
        architectures = getattr(model_config, "architectures", [])
        if any("CausalLM" in a for a in architectures) or "gemma" in target_id.lower():
            is_causal = True
    except Exception:
        pass

    if is_causal:
        raw_model = AutoModelForCausalLM.from_pretrained(
            target_id,
            token=token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        # Detach causal LM head to expose the transformer trunk
        if hasattr(raw_model, "model"):
            trunk = raw_model.model
        else:
            trunk = raw_model
        hidden_dim = getattr(raw_model.config, "hidden_size", getattr(raw_model.config, "dim", 768))
        target_modules = config.lora_target_modules
    else:
        raw_model = AutoModel.from_pretrained(target_id, token=token)
        trunk = raw_model
        hidden_dim = getattr(raw_model.config, "hidden_size", getattr(raw_model.config, "dim", 768))
        # Default transformer target modules for encoder architectures
        target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"] if "distilbert" in target_id.lower() else ["query", "key", "value", "dense"]

    # Bind LoRA parameter-efficient adaptation
    try:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        peft_trunk = get_peft_model(trunk, lora_config)
    except Exception as e:
        logger.warning(f"Could not apply standard LoRA target modules ({e}); using trunk parameters directly.")
        peft_trunk = trunk

    peft_trunk.to(device)
    return peft_trunk, tokenizer, hidden_dim, target_id
