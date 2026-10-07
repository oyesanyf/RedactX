"""
Configuration management for RedactX models, training, and deployment.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RedactXConfig:
    """
    Configuration parameters for RedactX decision engine.
    """
    base_model_id: str = "google/vaultgemma-1b"
    fallback_model_id: str = "distilbert/distilbert-base-uncased"
    use_fallback_if_gated: bool = True
    num_classes: int = 2
    hidden_dim: Optional[int] = None  # Inferred dynamically from backbone if None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    max_length: int = 512
    decision_threshold: float = 0.50
    temperature: float = 1.0  # Platt calibration temperature
    enable_span_locator: bool = True
    device: Optional[str] = None  # 'cuda' | 'cpu' or auto-detect


@dataclass
class TrainingConfig:
    """
    Hyperparameters and runtime settings for fine-tuning.
    """
    epochs: int = 3
    batch_size: int = 4
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    val_split: float = 0.20
    output_dir: str = "./models/redactx"
    lambda_brier: float = 0.50
    lambda_ece: float = 0.25
    seed: int = 42


@dataclass
class GatewayConfig:
    """
    Configuration for line-rate FastAPI serving gateway.
    """
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    enable_cors: bool = True


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """
    Robust multi-source token resolution across:
    1. Explicit CLI argument (--hf-token)
    2. OS Environment variables (HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN)
    3. Windows User & System Registry (HKCU\\Environment & HKLM Environment)
    4. Hugging Face CLI login cache (~/.cache/huggingface/token)
    5. Local project token file (token.txt or .env)
    """
    if cli_token and cli_token.strip():
        return cli_token.strip()

    # 1. Process OS Environment variables
    for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_AUTH_TOKEN"]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()

    # 2. Windows Registry (User & Machine)
    try:
        import winreg
        for hkey, subkey in [
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        ]:
            try:
                with winreg.OpenKey(hkey, subkey) as key:
                    num_values = winreg.QueryInfoKey(key)[1]
                    for i in range(num_values):
                        name, val, _ = winreg.EnumValue(key, i)
                        if name.upper() in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_AUTH_TOKEN"] and val:
                            return str(val).strip()
            except Exception:
                pass
    except Exception:
        pass

    # 3. Hugging Face hub cached login token
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok and tok.strip():
            return tok.strip()
    except Exception:
        pass

    # 4. Local files (.env or token.txt)
    for p in ["token.txt", ".env", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
                        elif line.startswith("hf_"):
                            return line
            except Exception:
                pass

    return None

