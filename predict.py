"""
Interactive Inference Script: Run scans with your trained RedactX decision model.

Usage:
  # Scan a text string:
  python predict.py "Patient Elena Patel seen at Mercy Health on 06/15/2026."

  # Scan using your custom trained checkpoint:
  python predict.py --checkpoint ./my_model_checkpoint "Patient Marcus Kowalski admitted with MRN 123-45-6789."

  # Scan a file of clinical notes line by line:
  python predict.py --file notes.txt --output results.json
"""

import os
import sys
import json
import argparse
from redactx.config import RedactXConfig
from redactx.models.engine import RedactXDecisionEngine


def main():
    parser = argparse.ArgumentParser(description="Run Scans with Your Trained RedactX Model")
    parser.add_argument("text", nargs="?", type=str, default=None,
                        help="Clinical text to evaluate")
    parser.add_argument("--checkpoint", type=str, default="./models/RedactX",
                        help="Path to trained checkpoint directory (default: ./models/RedactX)")
    parser.add_argument("--model", type=str, default="google/vaultgemma-1b",
                        help="Base model architecture ID")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Decision threshold for PHI gate (default: 0.40)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to text or JSON file with clinical notes to scan")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional file path to save scan results")

    args = parser.parse_args()

    if not args.text and not args.file:
        parser.print_help()
        print("\nError: Please provide a text string or --file to scan.")
        sys.exit(1)

    # Checkpoint resolution
    ckpt = args.checkpoint
    if not os.path.exists(ckpt) and os.path.exists("./models/redactx"):
        ckpt = "./models/redactx"

    config = RedactXConfig(base_model_id=args.model, decision_threshold=args.threshold)
    engine = RedactXDecisionEngine(config=config)

    if os.path.exists(ckpt):
        print(f"Loading trained weights from: {ckpt}")
        engine.load_checkpoint(ckpt)
    else:
        print(f"Warning: Checkpoint '{ckpt}' not found. Running with base initialized weights.")

    if args.text:
        decision = engine.evaluate(args.text, decision_threshold=args.threshold)
        print("\n" + "=" * 60)
        print(f"Decision Verdict:  {decision.verdict}")
        print(f"PHI Probability:   {decision.phi_probability:.4f} (Clean: {decision.clean_probability:.4f})")
        print(f"Execution Latency: {decision.latency_ms:.2f} ms")
        print(f"Gateway Action:    {'[PASSED] No PHI detected' if decision.passed_gate else '[BLOCKED] Sensitive PHI detected'}")
        if decision.spans:
            print(f"Detected PHI Spans ({len(decision.spans)}):")
            for s in decision.spans:
                print(f"  - [{s.category}] \"{s.text}\" (offset {s.start}:{s.end}, conf {s.confidence:.2f})")
        print("=" * 60)

    elif args.file:
        print(f"Scanning file: {args.file}...")
        texts = []
        if args.file.endswith(".json"):
            with open(args.file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                texts = [r["context"] if isinstance(r, dict) else str(r) for r in raw]
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

        decisions = engine.evaluate_batch(texts, decision_threshold=args.threshold)
        blocked = sum(1 for d in decisions if not d.passed_gate)
        print(f"Scanned {len(decisions)} notes: {len(decisions) - blocked} Clean, {blocked} Flagged as PHI.")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([d.model_dump() for d in decisions], f, indent=2)
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
