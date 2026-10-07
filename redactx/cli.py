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


def cmd_serve_legacy(args):
    import uvicorn
    from redactx.api.server import create_app

    config = RedactXConfig(base_model_id=args.model)
    engine = RedactXDecisionEngine(config=config)
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)

    app = create_app(engine=engine)
    print(f"Starting legacy RedactX gateway on {args.host}:{args.port} (not hardened; use `redactx serve`)...")
    uvicorn.run(app, host=args.host, port=args.port)


def _production_settings(args):
    """Settings from REDACTX_* environment / .env, with CLI flags taking precedence (process-local only)."""
    from redactx.production.settings import Settings
    overrides = {}
    for flag, key in [("model_dir", "REDACTX_MODEL_DIR"), ("mode", "REDACTX_MODE"), ("device", "REDACTX_DEVICE"),
                      ("strategy", "REDACTX_STRATEGY"), ("unlocalized_policy", "REDACTX_UNLOCALIZED_POLICY")]:
        v = getattr(args, flag, None)
        if v:
            overrides[key] = v
    if getattr(args, "allow_no_auth", False):
        overrides["REDACTX_ALLOW_NO_AUTH"] = "1"
    if getattr(args, "hipaa_only", False):
        overrides["REDACTX_HIPAA_ONLY"] = "1"
    env = dict(os.environ)
    env.update(overrides)
    return Settings.from_env(env=env, dotenv_path=getattr(args, "env_file", ".env"))


def cmd_serve(args):
    import uvicorn
    from redactx.production.server import create_production_app

    try:
        settings = _production_settings(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    app = create_production_app(settings)
    print(f"RedactX production server: mode={settings.mode} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info",
                timeout_keep_alive=5, limit_concurrency=max(64, settings.max_queue * 4))


def _iter_inputs(paths: List[str]):
    for p in paths:
        if p == "-":
            yield "<stdin>", sys.stdin.read()
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for name in sorted(files):
                    if name.lower().endswith(".txt") and not name.endswith(".redacted.txt"):
                        fp = os.path.join(root, name)
                        with open(fp, "r", encoding="utf-8", errors="replace") as f:
                            yield fp, f.read()
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                yield p, f.read()


def cmd_redact(args):
    from redactx.production.server import Service

    args.allow_no_auth = True  # local batch tool: no HTTP auth involved
    try:
        settings = _production_settings(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    service = Service.from_settings(settings)
    redactor = service.redactor
    report = open(args.report, "w", encoding="utf-8") if args.report else None
    n_docs = n_flagged = 0
    try:
        for src, text in _iter_inputs(args.inputs):
            result = redactor.redact(text)
            n_docs += 1
            n_flagged += int(result.action != "PASS")
            if src == "<stdin>" or not args.out_dir:
                if src == "<stdin>" or len(args.inputs) == 1 and not os.path.isdir(args.inputs[0]):
                    sys.stdout.write(result.redacted_text)
                    if not result.redacted_text.endswith("\n"):
                        sys.stdout.write("\n")
                out_path = None
            else:
                rel = os.path.basename(src)
                out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".redacted.txt")
                os.makedirs(args.out_dir, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result.redacted_text)
            if report:
                row = {"source": src, "output": out_path, "model_version": service.model_version}
                row.update(result.to_dict(include_finding_text=False))
                row.pop("redacted_text")
                report.write(json.dumps(row) + "\n")
    finally:
        if report:
            report.close()
    print(f"[redactx] {n_docs} document(s), {n_flagged} redacted/flagged, mode={settings.mode}, "
          f"model={service.model_version}", file=sys.stderr)


def cmd_hash_key(args):
    import secrets
    from redactx.production.settings import hash_api_key
    key = args.key or secrets.token_urlsafe(32)
    if not args.key:
        print(f"API key (give to the client, store securely): {key}")
    print(f"SHA-256 hash (put in REDACTX_API_KEY_HASHES):    {hash_api_key(key)}")


def _add_production_flags(p):
    p.add_argument("--model-dir", type=str, default=None, help="Trained RedactX model directory (REDACTX_MODEL_DIR)")
    p.add_argument("--mode", type=str, default=None, choices=["hybrid", "redactx", "presidio"],
                   help="Detector: hybrid (RedactX + Presidio union, default), redactx, presidio")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--strategy", type=str, default=None, choices=["tag", "mask", "char", "pseudonym"])
    p.add_argument("--unlocalized-policy", type=str, default=None, choices=["redact_all", "review"])
    p.add_argument("--hipaa-only", action="store_true", help="Redact only HIPAA Safe Harbor identifier categories")
    p.add_argument("--env-file", type=str, default=".env", help="Optional .env file to read settings from")


def main():
    parser = argparse.ArgumentParser(description="RedactX: PHI/PII detection and redaction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan a text payload for PHI/PII (legacy engine)")
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
    p_train = subparsers.add_parser("train", help="Train LoRA adapters and Hobson head (legacy; use train.py)")
    p_train.add_argument("--samples", type=int, default=160)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--max-length", type=int, default=256)
    p_train.add_argument("--model", type=str, default="google/vaultgemma-1b")
    p_train.add_argument("--out", type=str, default="./redactx_checkpoint")
    p_train.set_defaults(func=cmd_train)

    # serve (production)
    p_serve = subparsers.add_parser("serve", help="Run the hardened production API server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--allow-no-auth", action="store_true", help="Disable API-key auth (development only)")
    _add_production_flags(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    # redact (batch, local)
    p_red = subparsers.add_parser("redact", help="Redact .txt files, directories, or stdin ('-')")
    p_red.add_argument("inputs", nargs="+", help="Files, directories (*.txt, recursive) or '-' for stdin")
    p_red.add_argument("--out-dir", type=str, default=None, help="Write <name>.redacted.txt files here")
    p_red.add_argument("--report", type=str, default=None,
                       help="JSONL report (actions, categories, offsets; never the original values)")
    _add_production_flags(p_red)
    p_red.set_defaults(func=cmd_redact)

    # hash-key
    p_key = subparsers.add_parser("hash-key", help="Generate an API key and its SHA-256 hash for the server")
    p_key.add_argument("--key", type=str, default=None, help="Hash this key instead of generating one")
    p_key.set_defaults(func=cmd_hash_key)

    # serve-legacy
    p_legacy = subparsers.add_parser("serve-legacy", help="Run the old, non-hardened gateway")
    p_legacy.add_argument("--host", type=str, default="127.0.0.1")
    p_legacy.add_argument("--port", type=int, default=8080)
    p_legacy.add_argument("--model", type=str, default="google/vaultgemma-1b")
    p_legacy.add_argument("--checkpoint", type=str, default=None)
    p_legacy.set_defaults(func=cmd_serve_legacy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
