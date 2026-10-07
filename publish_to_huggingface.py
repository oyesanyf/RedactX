"""
RedactX Hugging Face Publisher.
Publishes fine-tuned RedactX (VaultGemma 1B) weights, tokenizers, model card,
and calibration scores to the Hugging Face Hub under your account.

Usage:
  python publish_to_huggingface.py [--repo-id your_username/RedactX] [--private]
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, whoami
from redactx.config import resolve_hf_token


def main():
    parser = argparse.ArgumentParser(description="Publish RedactX Model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Target repository ID (e.g. 'username/RedactX'). Defaults to 'your_username/RedactX'.")
    parser.add_argument("--model-dir", type=str, default="./models/RedactX",
                        help="Path to the model directory (default: ./models/RedactX)")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Create repository as private (default: public)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token (defaults to active environment token)")

    args = parser.parse_args()

    # Resolve token
    token = resolve_hf_token(args.token)
    if not token:
        print("\nError: No active Hugging Face token detected in OS environment, registry, or CLI arguments.")
        print("Please provide --token or set $env:HF_TOKEN.")
        sys.exit(1)

    print("=" * 65)
    print(" REDACTX HUGGING FACE MODEL PUBLISHER")
    print("=" * 65)

    api = HfApi(token=token)
    user_info = whoami(token=token)
    username = user_info.get("name")
    print(f"Authenticated as Hugging Face user: @{username}")

    repo_id = args.repo_id or f"{username}/RedactX"
    print(f"Target Repository: {repo_id} ({'Private' if args.private else 'Public'})")

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        sys.exit(1)

    # Verify key artifacts
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for rf in required_files:
        p = os.path.join(args.model_dir, rf)
        if not os.path.exists(p):
            print(f"Error: Required model artifact '{rf}' missing in {args.model_dir}")
            sys.exit(1)

    print(f"\nStep 1: Creating repository '{repo_id}' on Hugging Face Hub...")
    repo_url = api.create_repo(
        repo_id=repo_id,
        token=token,
        private=args.private,
        exist_ok=True,
        repo_type="model"
    )
    print(f"  Repo created/verified: {repo_url}")

    print(f"\nStep 2: Uploading model folder from {os.path.abspath(args.model_dir)}...")
    print("  (Uploading model.safetensors, tokenizers, config, and benchmark model card)")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message="Initial release of RedactX (VaultGemma 1B fine-tune)"
    )

    print("\n" + "=" * 65)
    print(" SUCCESS! RedactX is now published on Hugging Face:")
    print(f" https://huggingface.co/{repo_id}")
    print("=" * 65)


if __name__ == "__main__":
    main()
