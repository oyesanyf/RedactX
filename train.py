"""
RedactX & OpenJev Fine-Tuning Script for VaultGemma.
Executes the four sequential engineering phases:
1. Construct Calibrated Decision Dataset (Noul, Choice with Permutations, Score).
2. Initialize VaultGemma with Targeted LoRA Adapters.
3. Multi-Objective Calibration Loss (KL Divergence + Brier Score + Consistency).
4. Single-Token Forward Training Loop with Candidate Logit Gathering.
5. Permutation Stability & ECE Validation, followed by merge_and_unload() Export.

Usage:
    # Recommended (v3): contrastive recipe (real PII vs. same-text-without-PII vs. natural clean text) + span head,
    # plus clinical hard negatives, synthetic clinical PHI/clean pairs and AGE/DEMOGRAPHIC upweighting (all default):
    python train.py --model google/vaultgemma-1b --force-vaultgemma --recipe-contrastive --samples 2000 --epochs 3 --output-dir ./models/RedactX-v3

    # The v2 recipe (no hard negatives / pairs / upweighting):
    python train.py --model google/vaultgemma-1b --force-vaultgemma --recipe-contrastive --samples 2000 --epochs 3
        --hard-negative-ratio 0 --clinical-pair-ratio 0 --span-category-weights "" --oversample-factor 1 --output-dir ./models/RedactX-v2b

    # Legacy recipe:
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
from redactx.data.contrastive_corpus import load_contrastive_corpus
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
    parser.add_argument("--recipe-contrastive", action="store_true", default=False,
                        help="Contrastive recipe: --samples real PII docs (Nemotron-PII/Gretel) + the same docs with PII "
                             "replaced by generic phrases + natural clean text (PubMedQA/flashcards/code). "
                             "Noul-only records with token-level span labels. Recommended.")
    parser.add_argument("--natural-negative-ratio", type=float, default=0.5,
                        help="Contrastive recipe: natural clean docs per PII doc (default: 0.5)")
    parser.add_argument("--hard-negative-ratio", type=float, default=0.75,
                        help="Contrastive recipe: clean CLINICAL docs per PII doc (PubMedQA pqa_unlabeled, WikiDoc, "
                             "synthetic identifier-free notes). Fixes over-flagging of clean clinical text. "
                             "0 = off (v2 recipe). Default: 0.75")
    parser.add_argument("--clinical-pair-ratio", type=float, default=0.25,
                        help="Contrastive recipe: synthetic clinical PHI notes (each with its natural clean twin) "
                             "per PII doc. 0 = off. Default: 0.25")
    parser.add_argument("--n2c2-train-ratio", type=float, default=0.25,
                        help="Contrastive recipe: real n2c2 2014 clinical PHI windows (each with its generic-replaced twin) "
                             "per PII doc. Auto-discovered from data/ if omitted. 0 = off. Default: 0.25")
    parser.add_argument("--span-category-weights", type=str, default="AGE=3,DEMOGRAPHIC=3",
                        help="Span-head loss weight per HIPAA category, e.g. 'AGE=3,DEMOGRAPHIC=3'. '' = off")
    parser.add_argument("--oversample-categories", type=str, default="AGE,DEMOGRAPHIC",
                        help="Repeat training positives containing these categories (with their twins). '' = off")
    parser.add_argument("--oversample-factor", type=int, default=2,
                        help="Copies of each oversampled positive/twin pair (default: 2)")
    parser.add_argument("--max-chars", type=int, default=600,
                        help="Contrastive recipe: max characters of text per record (default: 600)")
    parser.add_argument("--max-length", type=int, default=384,
                        help="Max prompt tokens (default: 384)")
    parser.add_argument("--lambda-span", type=float, default=None,
                        help="Weight of the token span-head loss. Default: 1.0 with --recipe-contrastive, else 0 (off)")
    parser.add_argument("--span-pos-weight", type=float, default=3.0,
                        help="BCE positive-class weight for PII tokens in the span loss (default: 3.0)")
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
    parser.add_argument("--no-benchmark", action="store_true", default=False,
                        help="Skip the held-out benchmark (validate_model.py) that runs after training")
    parser.add_argument("--benchmark-n-ai4privacy", type=int, default=200,
                        help="ai4privacy validation docs for the post-training benchmark (default: 200)")
    parser.add_argument("--benchmark-skip-presidio", action="store_true", default=False,
                        help="Do not run real Presidio as the comparison baseline in the benchmark")

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
    if args.recipe_contrastive:
        print("\n=======================================================")
        print(" Step 1: Building Contrastive Corpus")
        print("   (real PII docs + generic-replaced twins + natural clean text; ai4privacy held out)")
        print("=======================================================")
        from redactx.data.contrastive_corpus import parse_category_weights
        train_records, val_records, stats = load_contrastive_corpus(
            num_pii_docs=args.samples,
            natural_negative_ratio=args.natural_negative_ratio,
            max_chars=args.max_chars,
            seed=args.seed,
            hf_token=active_token,
            hard_negative_ratio=args.hard_negative_ratio,
            clinical_pair_ratio=args.clinical_pair_ratio,
            span_category_weights=parse_category_weights(args.span_category_weights),
            oversample_categories=tuple(c.strip() for c in args.oversample_categories.split(",") if c.strip()),
            oversample_factor=args.oversample_factor,
            local_notes_dir=args.local_notes_dir,
            n2c2_train_ratio=args.n2c2_train_ratio,
        )
        print("\n[Corpus Composition - Contrastive Recipe]")
        print(f"  - PII docs (Nemotron-PII / Gretel):           {stats['pii_docs_nemotron']} / {stats['pii_docs_gretel']}")
        print(f"  - Synthetic clinical PHI/clean pairs:         {stats['clinical_pair_docs']}")
        print(f"  - Real n2c2 clinical PHI/clean pairs:         {stats.get('n2c2_train_pair_docs', 0)}")
        print(f"  - Natural clean docs:                         {stats['natural_negative_docs']} {stats['natural_negative_sources']}")
        print(f"  - Clinical hard negatives:                    {stats['hard_negative_docs']} {stats['hard_negative_sources']}")
        print(f"  - Span category weights / oversampling:       {stats['span_category_weights']} / {stats['oversample']}")
        print(f"  - Train gold spans by category:               {stats['train_gold_spans_by_category']}")
        print(f"  - Train records (PHI / clean):                {stats['train_positives']} / {stats['train_negatives']}")
        print(f"  - Val records (PHI / clean):                  {stats['val_positives']} / {stats['val_negatives']}")
    else:
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

    lambda_span = args.lambda_span if args.lambda_span is not None else (1.0 if args.recipe_contrastive else 0.0)

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
        hf_token=active_token,
        lambda_span=lambda_span,
        span_pos_weight=args.span_pos_weight,
        max_length=args.max_length
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

    if not args.recipe_contrastive:
        # Export synthetic jsonl artifact into final model directory (legacy recipes only)
        synthetic_jsonl_path = os.path.join(args.output_dir, "redactx_synthetic_train.jsonl")
        export_synthetic_train_jsonl(synthetic_jsonl_path, num_samples=min(args.samples, 1000))
        print(f"  - Synthetic JSONL exported:                   {synthetic_jsonl_path}")

    def fmt(v, spec=".4f"):
        return "N/A" if v is None else format(v, spec)

    print("\n=======================================================")
    print(" Step 5: Final Validation & Merged Model Export Summary")
    print("=======================================================")
    print(f"Model Name:                   {scores.get('model_name', args.model_name)}")
    print(f"Base Backbone:                {scores.get('base_model', scores.get('model_id'))}")
    print(f"Expected Calibration Error:   {fmt(scores['expected_calibration_error_ece'])}")
    print(f"Permutation Stability:        {fmt(scores['permutation_stability_rate'], '.2f')}")
    print(f"Mean Brier Score:             {fmt(scores['brier_score'])}")
    print(f"Val doc acc / recall / spec:  {fmt(scores.get('val_doc_accuracy'))} / "
          f"{fmt(scores.get('val_doc_recall'))} / {fmt(scores.get('val_doc_specificity'))}")
    print(f"Val span P / R / F1 (tokens): {fmt(scores.get('val_span_precision'))} / "
          f"{fmt(scores.get('val_span_recall'))} / {fmt(scores.get('val_span_f1'))}")
    print("NOTE: these are in-distribution validation numbers. Run validate_model.py for held-out evaluation.")
    print(f"\n[Success] Your fine-tuned model '{args.model_name}' is saved and ready in: {os.path.abspath(args.output_dir)}")

    # ==============================================================================
    # Step 6: Held-out benchmark (same evaluation as validate_model.py)
    # ==============================================================================
    if args.no_benchmark:
        return
    if not os.path.exists(os.path.join(args.output_dir, "config.json")):
        print("\n[Benchmark] Skipped: output is an unmerged adapter (--no-merge). Run validate_model.py on a merged model.")
        return
    print("\n=======================================================")
    print(" Step 6: Held-out Benchmark (H / G / P / Q sets, vs. old v1 and real Presidio)")
    print("=======================================================")
    import gc
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from validate_model import run_validation, print_benchmark_table
        results = run_validation(
            args.output_dir,
            device=args.device,
            n_ai4privacy=args.benchmark_n_ai4privacy,
            max_chars=args.max_chars,
            skip_presidio=args.benchmark_skip_presidio,
            hf_token=active_token
        )
        print_benchmark_table(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Benchmark] Failed ({type(e).__name__}: {e}). The trained model is saved; re-run with:\n"
              f"  python validate_model.py --model-dir {args.output_dir}")


if __name__ == "__main__":
    main()
