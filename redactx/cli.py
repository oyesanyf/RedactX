"""
Command-line interface for RedactX.
"""

import os
import json
import argparse
import sys
from typing import List

from redactx.config import RedactXConfig, TrainingConfig
from redactx.data.generator import generate_redactx_corpus
from redactx.models.engine import RedactXDecisionEngine
from redactx.training.trainer import RedactXTrainer


def cmd_scan(args):
    config = RedactXConfig(
        base_model_id=args.model,
        decision_threshold=args.threshold
    )
    engine = RedactXDecisionEngine(config=config)
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)

    print(f"\n--- Scanning Payload with RedactX ({engine.resolved_model_id}) ---")
    decision = engine.evaluate(args.text, decision_threshold=args.threshold)

    print(f"Context: \"{decision.context}\"")
    print(f"Verdict: {decision.verdict}")
    print(f"PHI Probability: {decision.phi_probability:.4f} | Clean Probability: {decision.clean_probability:.4f}")
    print(f"Execution Latency: {decision.latency_ms:.2f} ms")
    print(f"Gateway Action: {'[PASSED] Clean payload' if decision.passed_gate else '[BLOCKED] Redaction required'}")
    if decision.spans:
        print("\nDetected PHI Spans:")
        for s in decision.spans:
            print(f"  [{s.category}] \"{s.text}\" (chars {s.start}:{s.end}, conf {s.confidence:.2f})")


def cmd_generate(args):
    print(f"Generating {args.samples} HIPAA Safe Harbor synthetic records...")
    records = generate_redactx_corpus(num_samples=args.samples, phi_ratio=args.phi_ratio)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Dataset successfully exported to {args.out}")


def cmd_train(args):
    print("Initializing RedactX synthetic dataset...")
    data = generate_redactx_corpus(num_samples=args.samples)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    config = RedactXConfig(
        base_model_id=args.model,
        max_length=args.max_length
    )
    engine = RedactXDecisionEngine(config=config)

    t_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.lr,
        output_dir=args.out
    )
    trainer = RedactXTrainer(engine=engine, config=t_cfg)
    results = trainer.train(train_data, val_data)
    print(f"\nTraining complete. Final validation accuracy: {results['final_metrics'].get('accuracy', 0):.2f}%")


def cmd_serve(args):
    import uvicorn
    from redactx.api.server import create_app

    config = RedactXConfig(base_model_id=args.model)
    engine = RedactXDecisionEngine(config=config)
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)

    app = create_app(engine=engine)
    print(f"Starting RedactX Decision Gateway on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="RedactX: Jev-Like PHI Decision Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan a text payload for PHI/PII")
    p_scan.add_argument("text", type=str, help="Text to evaluate")
    p_scan.add_argument("--model", type=str, default="google/vaultgemma-1b")
    p_scan.add_argument("--checkpoint", type=str, default=None)
    p_scan.add_argument("--threshold", type=float, default=0.50)
    p_scan.set_defaults(func=cmd_scan)

    # generate
    p_gen = subparsers.add_parser("generate", help="Synthesize HIPAA dataset")
    p_gen.add_argument("--samples", type=int, default=200)
    p_gen.add_argument("--phi-ratio", type=float, default=0.50)
    p_gen.add_argument("--out", type=str, default="redactx_corpus.json")
    p_gen.set_defaults(func=cmd_generate)

    # train
    p_train = subparsers.add_parser("train", help="Train LoRA adapters and Hobson head")
    p_train.add_argument("--samples", type=int, default=160)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--max-length", type=int, default=256)
    p_train.add_argument("--model", type=str, default="google/vaultgemma-1b")
    p_train.add_argument("--out", type=str, default="./redactx_checkpoint")
    p_train.set_defaults(func=cmd_train)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run FastAPI line-rate gateway")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--model", type=str, default="google/vaultgemma-1b")
    p_serve.add_argument("--checkpoint", type=str, default=None)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
