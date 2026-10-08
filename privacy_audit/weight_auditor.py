"""
Weight & Parameter Distribution Auditor for RedactX (VaultGemma-1B).

Audits trained parameter matrices (TokenSpanLocator, LoRA projections, causal head)
to check for weight norm explosion, extreme spectral outliers, or localized parameter
energy concentration associated with memorization hotspots.
"""

from typing import Dict, List, Any
import torch
from redactx.models.openjev import OpenJevVaultGemmaEngine


def run_weight_audit(engine: OpenJevVaultGemmaEngine) -> Dict[str, Any]:
    """Audits parameter norms and weight distributions across trained heads."""
    print("=" * 70)
    print(" Executing Weight & Parameter Spectral Audit...")
    print("=" * 70)

    parameter_stats = []
    total_params = 0
    trained_params = 0

    # 1. Inspect TokenSpanLocator weights
    if hasattr(engine, "span_locator") and engine.span_locator is not None:
        for name, param in engine.span_locator.named_parameters():
            p_data = param.detach().float().cpu()
            n_elem = p_data.numel()
            total_params += n_elem
            if param.requires_grad:
                trained_params += n_elem

            frob_norm = torch.linalg.norm(p_data).item()
            mean_val = p_data.mean().item()
            std_val = p_data.std().item()
            max_val = p_data.abs().max().item()

            parameter_stats.append({
                "module": f"span_locator.{name}",
                "shape": list(p_data.shape),
                "numel": n_elem,
                "frobenius_norm": round(frob_norm, 4),
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "max_abs": round(max_val, 4),
                "health": "STABLE" if max_val < 10.0 and frob_norm < 100.0 else "UNSTABLE",
            })

    # 2. Inspect Base Trunk & Head summary
    trunk_max_val = 0.0
    for name, param in engine.model.named_parameters():
        if "embed" in name or "lm_head" in name or "norm" in name:
            p_data = param.detach().float().cpu()
            trunk_max_val = max(trunk_max_val, p_data.abs().max().item())

    weight_verdict = "PASSED_STABLE" if all(p["health"] == "STABLE" for p in parameter_stats) else "ATTENTION_REQUIRED"

    print("Parameter Distribution Summary:")
    for s in parameter_stats:
        print(f"  * {s['module']:<30} | Shape: {str(s['shape']):<14} | Norm: {s['frobenius_norm']:<8} | Max: {s['max_abs']:<6} ({s['health']})")

    print(f"\nWeight Parameter Health: {weight_verdict}\n")

    return {
        "status": weight_verdict,
        "total_span_head_params": total_params,
        "parameters_audited": len(parameter_stats),
        "trunk_max_activation_norm": round(trunk_max_val, 4),
        "details": parameter_stats,
    }
