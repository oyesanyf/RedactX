"""
Load test for a running RedactX production server.

Sends real documents (the held-out handwritten set from validate_model.py, optionally a JSONL/text corpus
you supply) to /v2/redact with N concurrent clients and reports throughput, latency percentiles and the
status-code mix (429 = rate limited, 503 = overloaded / not ready, 504 = timeout).

Usage:
    redactx serve --model-dir ./models/RedactX-v2 &            # in another terminal
    python loadtest.py --url http://127.0.0.1:8080 --api-key $KEY --concurrency 8 --requests 400
    python loadtest.py --url ... --api-key ... --corpus notes.jsonl --field text --endpoint /v2/redact/batch --batch 8
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    k = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[k]


def load_corpus(path: Optional[str], field: str) -> List[str]:
    if not path:
        from validate_model import HELD_OUT_CLEAN, HELD_OUT_PHI
        return HELD_OUT_PHI + HELD_OUT_CLEAN
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    texts.append(str(json.loads(line)[field]))
        else:
            texts = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    if not texts:
        raise SystemExit(f"no documents in {path}")
    return texts


async def run_load(url: str, api_key: Optional[str], texts: Sequence[str], concurrency: int, total: int,
                   endpoint: str, batch: int, timeout: float) -> Dict:
    import httpx
    headers = {"X-API-Key": api_key} if api_key else {}
    latencies: List[float] = []
    statuses: Counter = Counter()
    docs_ok = 0
    counter = {"next": 0}
    lock = asyncio.Lock()

    async def worker(client):
        nonlocal docs_ok
        while True:
            async with lock:
                i = counter["next"]
                if i >= total:
                    return
                counter["next"] += 1
            if endpoint.endswith("/batch"):
                body = {"texts": [texts[(i * batch + j) % len(texts)] for j in range(batch)]}
                n_docs = batch
            else:
                body = {"text": texts[i % len(texts)]}
                n_docs = 1
            t0 = time.perf_counter()
            try:
                r = await client.post(endpoint, json=body, headers=headers)
                statuses[str(r.status_code)] += 1
                if r.status_code == 200:
                    latencies.append((time.perf_counter() - t0) * 1000)
                    docs_ok += n_docs
            except httpx.HTTPError as e:
                statuses[type(e).__name__] += 1

    async with httpx.AsyncClient(base_url=url, timeout=timeout) as client:
        ready = await client.get("/readyz")
        if ready.status_code != 200:
            raise SystemExit(f"server not ready: {ready.status_code} {ready.text}")
        t_start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t_start

    lat = sorted(latencies)
    return {
        "endpoint": endpoint, "concurrency": concurrency, "requests": total,
        "batch": batch if endpoint.endswith("/batch") else 1,
        "wall_s": round(wall, 2),
        "requests_per_s": round(len(lat) / wall, 2) if wall else None,
        "documents_per_s": round(docs_ok / wall, 2) if wall else None,
        "latency_ms": {"p50": percentile(lat, 0.50), "p95": percentile(lat, 0.95), "p99": percentile(lat, 0.99),
                       "mean": round(statistics.mean(lat), 1) if lat else None, "max": lat[-1] if lat else None},
        "status_counts": dict(statuses),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Load test a running RedactX server")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=os.environ.get("REDACTX_API_KEY"),
                    help="plaintext API key (default: $REDACTX_API_KEY)")
    ap.add_argument("--endpoint", default="/v2/redact", choices=["/v2/redact", "/v2/detect", "/v2/redact/batch"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8, help="documents per request for /v2/redact/batch")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--corpus", default=None, help=".jsonl (one object per line) or .txt (blank-line separated)")
    ap.add_argument("--field", default="text", help="JSONL field holding the document")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    texts = load_corpus(args.corpus, args.field)
    report = asyncio.run(run_load(args.url, args.api_key, texts, args.concurrency, args.requests,
                                  args.endpoint, args.batch, args.timeout))
    for k in ("p50", "p95", "p99", "max"):
        v = report["latency_ms"][k]
        report["latency_ms"][k] = round(v, 1) if v is not None else None
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    ok = report["status_counts"].get("200", 0)
    if ok == 0:
        print("no successful requests", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
