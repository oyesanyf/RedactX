"""
Tests for the operational tools: threshold calibration, benchmark metrics, long-document construction, and
the load tester against a live uvicorn server (real Presidio, real HTTP).
"""

import asyncio
import shutil
import socket
import threading
import time

import pytest

from calibrate_thresholds import calibrate, gold_span_scores, load_generator
from validate_model import HELD_OUT_CLEAN, char_metrics, make_long_documents, per_category_recall


# ============================================================================ metrics
def test_char_metrics_counts_non_whitespace_characters():
    text = "Call Jane Roe at 555-0101 today."
    gold = [{"start": 5, "end": 13, "category": "NAME"}, {"start": 17, "end": 25, "category": "PHONE"}]
    docs = [{"text": text, "gold": gold}]
    # predicted: the full name, and half of the phone number, plus a false positive on "today"
    pred = [{"start": 5, "end": 13}, {"start": 17, "end": 21}, {"start": 26, "end": 31}]
    m = char_metrics(docs, lambda d: pred)
    # gold non-ws chars: "JaneRoe" (7) + "555-0101" (8) = 15; tp = 7 + 4 = 11; fp = 5 ("today")
    assert m["gold_chars"] == 15
    assert m["recall"] == round(11 / 15, 4) and m["precision"] == round(11 / 16, 4)

    cats = per_category_recall(docs, lambda d: pred)
    assert cats["NAME"] == {"n": 1, "touched_recall": 1.0, "char_recall": 1.0}
    assert cats["PHONE"]["touched_recall"] == 1.0 and cats["PHONE"]["char_recall"] == 0.5

    clean = [{"text": "no identifiers here", "gold": []}]
    assert char_metrics(clean, lambda d: [{"start": 0, "end": 2}])["precision"] == 0.0


def test_make_long_documents_preserves_gold_offsets():
    docs = []
    for i in range(10):
        t = f"Patient number {i} is Ann Lee{i}."
        s = t.index("Ann")
        docs.append({"text": t, "spans": [(s, s + len(f"Ann Lee{i}"), "NAME")]})
    long_docs = make_long_documents(docs, per_doc=5)
    assert len(long_docs) == 2
    for ld in long_docs:
        assert len(ld["gold"]) == 5
        for g in ld["gold"]:
            assert ld["text"][g["start"]:g["end"]] == g["text"] and g["text"].startswith("Ann Lee")


def test_gold_span_scores():
    probs = [0.1, 0.9, 0.2, 0.7, 0.05]
    ranges = [(-1, -1), (0, 4), (5, 9), (10, 14), (15, 19)]
    pos, neg = gold_span_scores(probs, ranges, [(0, 9), (15, 19)])
    assert pos == [0.9, 0.05]                 # max over overlapping tokens per gold span
    assert neg == [0.7]                       # real-text token outside every gold span; template ignored


# ============================================================================ calibration (real engine)
@pytest.fixture(scope="module")
def engine(gpt2_checkpoint):
    from redactx.models.openjev import OpenJevVaultGemmaEngine
    return OpenJevVaultGemmaEngine.from_pretrained(gpt2_checkpoint, device="cpu")


def test_calibrate_meets_target_recall_and_engine_loads_it(engine, gpt2_checkpoint, tmp_path):
    from redactx.models.openjev import OpenJevVaultGemmaEngine
    from redactx.production.thresholds import ThresholdConfig

    positives, negatives = load_generator(40, seed=4242)
    negatives = negatives + HELD_OUT_CLEAN
    target = 0.9
    cfg, diag = calibrate(engine, positives, negatives, target, batch_size=8, sources="test")

    # the chosen thresholds really achieve the target recall on the calibration data
    pos_res = engine.score_texts([d["text"] for d in positives], return_token_probs=True)
    doc_recall = sum(r["p_phi"] >= cfg.doc_threshold for r in pos_res) / len(pos_res)
    assert doc_recall >= target and cfg.doc_operating_point["recall"] >= target
    span_scores = []
    for d, r in zip(positives, pos_res):
        span_scores += gold_span_scores(r["token_probs"], r["raw_ranges"], [(s, e) for s, e, _ in d["spans"]])[0]
    assert sum(s >= cfg.span_threshold for s in span_scores) / len(span_scores) >= target
    assert diag["n_pos_docs"] == len(positives) and diag["n_neg_docs"] == len(negatives)

    # written thresholds are picked up by a freshly loaded engine
    ckpt = tmp_path / "ckpt"
    shutil.copytree(gpt2_checkpoint, ckpt)
    cfg.save(str(ckpt))
    assert ThresholdConfig.load(str(ckpt)).doc_threshold == cfg.doc_threshold
    e2 = OpenJevVaultGemmaEngine.from_pretrained(str(ckpt), device="cpu")
    assert e2.doc_threshold == pytest.approx(cfg.doc_threshold)
    assert e2.span_threshold == pytest.approx(cfg.span_threshold)
    verdict = e2.evaluate_text(positives[0]["text"])
    assert verdict.verdict == ("Contains_PHI_PII" if verdict.phi_probability >= cfg.doc_threshold else "Clean")


def test_calibrate_rejects_empty_inputs(engine):
    with pytest.raises(ValueError):
        calibrate(engine, [], ["clean text"], 0.95)


def test_stratified_threshold_protects_the_hard_source():
    import random
    from calibrate_thresholds import _choose_stratified
    from redactx.production.thresholds import threshold_for_recall
    rng = random.Random(7)
    easy = [0.99 + 0.01 * rng.random() for _ in range(600)]       # e.g. generator notes
    hard = [rng.random() for _ in range(300)]                       # e.g. a harder real-text source
    target = 0.9
    t, method, per = _choose_stratified({"easy": easy, "hard": hard}, target, 0.95)
    assert method["binding_source"] == "hard" and method["target_certified"] is True
    assert set(per) == {"easy", "hard"} and t == pytest.approx(per["hard"]["threshold"])
    # each source meets the target on its own at the chosen threshold
    assert sum(s >= t for s in hard) / len(hard) >= target
    assert sum(s >= t for s in easy) / len(easy) >= target
    # pooling would have chosen a higher threshold that fails the hard source
    pooled = threshold_for_recall(easy + hard, target)
    assert pooled > t and sum(s >= pooled for s in hard) / len(hard) < target


def test_calibrate_reports_per_source_recall(engine):
    positives, negatives = load_generator(40, seed=4242)
    tagged = [{**d, "source": "a" if i % 2 else "b"} for i, d in enumerate(positives)]
    target = 0.9
    cfg, _ = calibrate(engine, tagged, negatives + HELD_OUT_CLEAN, target, batch_size=8, sources="test")
    per = cfg.doc_operating_point["per_source"]
    assert set(per) == {"a", "b"}
    assert all(v["recall_at_chosen"] >= target for v in per.values())
    assert cfg.doc_operating_point["binding_source"] in per


# ============================================================================ load test vs live server
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_loadtest_against_live_server(tmp_path):
    pytest.importorskip("presidio_analyzer")
    import uvicorn
    from loadtest import run_load
    from redactx.production.server import create_production_app
    from redactx.production.settings import Settings, hash_api_key

    key = "loadtest-key-0123456789"
    settings = Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_API_KEY_HASHES": hash_api_key(key),
                                      "REDACTX_AUDIT_LOG": str(tmp_path / "audit.jsonl"),
                                      "REDACTX_MAX_CONCURRENCY": "2",
                                      # throughput test, not a rate-limit test (batches cost 1 token per document)
                                      "REDACTX_RATE_LIMIT_PER_MIN": "6000", "REDACTX_RATE_BURST": "1000"},
                                 dotenv_path=None)
    app = create_production_app(settings)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started
    try:
        texts = ["Please call Jennifer Okonkwo at 312-555-0198.", "The pump pressure stayed within range."]
        report = asyncio.run(run_load(f"http://127.0.0.1:{port}", key, texts, concurrency=4, total=24,
                                      endpoint="/v2/redact", batch=1, timeout=60))
        assert report["status_counts"] == {"200": 24}
        assert report["latency_ms"]["p50"] is not None and report["requests_per_s"] > 0
        batch = asyncio.run(run_load(f"http://127.0.0.1:{port}", key, texts, concurrency=2, total=4,
                                     endpoint="/v2/redact/batch", batch=3, timeout=60))
        assert batch["status_counts"] == {"200": 4} and batch["documents_per_s"] > 0
        bad = asyncio.run(run_load(f"http://127.0.0.1:{port}", "wrong-key", texts, concurrency=1, total=2,
                                   endpoint="/v2/redact", batch=1, timeout=60))
        assert bad["status_counts"] == {"401": 2}
    finally:
        server.should_exit = True
        t.join(timeout=10)
