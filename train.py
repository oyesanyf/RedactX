"""
RedactX & OpenJev Fine-Tuning Script for VaultGemma.
Executes the four sequential engineering phases:
1. Construct Calibrated Decision Dataset (Noul, Choice with Permutations, Score).
2. Initialize VaultGemma with Targeted LoRA Adapters.
3. Multi-Objective Calibration Loss (KL Divergence + Brier Score + Consistency).
4. Single-Token Forward Training Loop with Candidate Logit Gathering.
5. Permutation Stability & ECE Validation, followed by merge_and_unload() Export.

Usage:
    # Train VaultGemma with OpenJev calibration & merge:
    python train.py --model google/vaultgemma-1b --force-vaultgemma --samples 500 --epochs 3 --output-dir ./models/openjev_vaultgemma
"""

import os
import sys
import json
import shutil
import argparse
import time
import torch

from redactx.data.openjev_dataset import build_calibrated_decision_corpus
from redactx.data.generator import export_synthetic_train_jsonl
from redactx.data.benchmark_loaders import (
    load_gretel_pii_benchmark, load_ai4privacy_openpii, load_nemotron_pii_benchmark,
    load_hybrid_training_corpus, load_recipe_60_20_20_corpus
)
from redactx.training.openjev_trainer import OpenJevFineTuningPipeline


def resolve_hf_token(cli_token: str = None) -> str:
    r"""
    Robust multi-source token resolution across:
    1. Explicit CLI argument (--hf-token)
    2. OS Environment variables (HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN)
    3. Windows User & System Registry (HKCU\Environment & HKLM Environment)
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


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune VaultGemma into an OpenJev Decision Engine")
    parser.add_argument("--model", type=str, default="google/vaultgemma-1b",
                        help="Base model ID (default: google/vaultgemma-1b)")
    env_token = os.environ.get("HF_TOKEN")
    parser.add_argument("--hf-token", type=str, default=env_token,
                        help="Hugging Face User Access Token (defaults to OS env variable HF_TOKEN)")
    parser.add_argument("--samples", type=int, default=300,
                        help="Number of synthetic calibrated records to construct (default: 300)")
    parser.add_argument("--benchmark-samples", type=int, default=50,
                        help="Number of actual benchmark samples to ingest from Gretel AI / OpenPII (default: 50)")
    parser.add_argument("--local-notes-dir", type=str, default=None,
                        help="Optional directory containing local N2C2 / MIMIC text notes")
    parser.add_argument("--recipe-60-20-20", action="store_true", default=False,
                        help="Use the recommended 60/20/20 hybrid recipe (60%% Persona, 20%% Clinical, 20%% Hard Negatives)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="LoRA learning rate (default: 2e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--lambda-brier", type=float, default=1.5,
                        help="Weight for Brier score loss penalty (default: 1.5)")
    parser.add_argument("--lambda-consistency", type=float, default=0.5,
                        help="Weight for permutation consistency loss (default: 0.5)")
    parser.add_argument("--model-name", type=str, default="RedactX",
                        help="Model name identifier (default: RedactX)")
    parser.add_argument("--output-dir", type=str, default="./models/RedactX",
                        help="Directory to save the merged fine-tuned model (default: ./models/RedactX)")
    parser.add_argument("--force-vaultgemma", action="store_true", default=False,
                        help="Strictly enforce loading google/vaultgemma-1b without fallback")
    parser.add_argument("--no-merge", action="store_true", default=False,
                        help="Keep LoRA adapter unmerged instead of calling merge_and_unload()")
    parser.add_argument("--device", type=str, default=None,
                        help="Target device ('cuda' or 'cpu'). Defaults to auto-detect.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Automatically read and verify HF_TOKEN from OS environment, Windows Registry, or CLI argument
    active_token = resolve_hf_token(args.hf_token)
    if active_token:
        os.environ["HF_TOKEN"] = active_token
        masked = active_token[:4] + "..." + active_token[-4:] if len(active_token) > 8 else "***"
        print(f"[Auth] HF_TOKEN detected and active from OS environment / Registry: {masked}")
    else:
        print("[Auth] Notice: No HF_TOKEN detected in OS environment, Windows Registry, or CLI arguments.")

    # Requirement: Models directory must be deleted on each run
    if os.path.exists(args.output_dir):
        print(f"\n--> Deleting previous model checkpoint in '{args.output_dir}' as requested...")
        shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--> Initialized clean model destination: {os.path.abspath(args.output_dir)}")

    # ==============================================================================
    # Step 1: Ingesting Hybrid Actual Benchmarks & Synthetic Dataset
    # ==============================================================================
    if args.recipe_60_20_20:
        print("\n=======================================================")
        print(" Step 1: Ingesting Recommended 60/20/20 Hybrid Dataset Recipe")
        print("   (60% Persona [Nemotron/Faker] / 20% Clinical [Gretel/OpenPII] / 20% Hard Negatives [PubMed/Logs])")
        print("=======================================================")
        corpus, stats = load_recipe_60_20_20_corpus(
            total_samples=args.samples,
            local_notes_dir=args.local_notes_dir,
            seed=args.seed,
            hf_token=active_token
        )
        print("\n[Corpus Composition - 60/20/20 Recipe]")
        print(f"  - Persona Records (Nemotron-PII / Faker):     {stats['persona_records']}")
        print(f"  - Clinical Records (Gretel / OpenPII / Notes): {stats['clinical_records']}")
        print(f"  - Hard Negative Records (PubMed / Telemetry):  {stats['hard_negative_records']}")
        print(f"  - Total Hybrid Decision Records:               {len(corpus)}")
    else:
        print("\n=======================================================")
        print(" Step 1: Ingesting Hybrid Actual Benchmarks & Synthetic Dataset")
        print("=======================================================")
        print(f"Loading {args.samples} synthetic samples + {args.benchmark_samples} actual benchmark samples...")
        corpus, stats = load_hybrid_training_corpus(
            synthetic_samples=args.samples,
            real_benchmark_samples=args.benchmark_samples,
            local_notes_dir=args.local_notes_dir,
            seed=args.seed,
            hf_token=active_token
        )
        print("\n[Corpus Composition]")
        print(f"  - Actual Benchmark Records (Gretel / OpenPII): {stats['actual_benchmark_records']}")
        print(f"  - Synthetic Healthcare Records (Faker):       {stats['synthetic_records']}")
        print(f"  - Total Hybrid Decision Records:              {stats['total_records']}")

    split_idx = int(len(corpus) * 0.8)
    train_records = corpus[:split_idx]
    val_records = corpus[split_idx:]
    print(f"  - Training Split:                             {len(train_records)} records")
    print(f"  - Validation Split:                           {len(val_records)} records")

    # ==============================================================================
    # Step 2: Initialize VaultGemma with Targeted Adapters
    # ==============================================================================
    print("\n=======================================================")
    print(" Step 2: Initializing VaultGemma with Targeted Adapters")
    print("=======================================================")
    pipeline = OpenJevFineTuningPipeline(
        model_id=args.model,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_brier=args.lambda_brier,
        lambda_consistency=args.lambda_consistency,
        use_fallback_if_gated=not args.force_vaultgemma,
        hf_token=active_token
    )

    # ==============================================================================
    # Step 3, 4 & 5: Calibration Loss, Single-Token Training, Validation & Merge
    # ==============================================================================
    print("\n=======================================================")
    print(" Step 3 & 4: Executing Multi-Objective Single-Token Training")
    print("=======================================================")
    scores = pipeline.train(
        train_records=train_records,
        val_records=val_records,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        merge_and_unload=not args.no_merge,
        dataset_stats=stats,
        model_name=args.model_name
    )

    # Export synthetic jsonl artifact into final model directory
    synthetic_jsonl_path = os.path.join(args.output_dir, "redactx_synthetic_train.jsonl")
    export_synthetic_train_jsonl(synthetic_jsonl_path, num_samples=min(args.samples, 1000))
    print(f"  - Synthetic JSONL exported:                   {synthetic_jsonl_path}")

    print("\n=======================================================")
    print(" Step 5: Final Validation & Merged Model Export Summary")
    print("=======================================================")
    print(f"Model Name:                   {scores.get('model_name', args.model_name)}")
    print(f"Base Backbone:                {scores.get('base_model', scores.get('model_id'))}")
    print(f"Expected Calibration Error:  {scores['expected_calibration_error_ece']:.4f}")
    print(f"Permutation Stability:        {scores['permutation_stability_rate']:.2f}%")
    print(f"Mean Brier Score:             {scores['brier_score']:.4f}")
    print(f"\n[Success] Your fine-tuned model '{args.model_name}' is saved and ready in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
