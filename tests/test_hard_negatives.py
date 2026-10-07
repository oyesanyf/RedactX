"""
Hard-negative retraining data (redactx/data/clean_notes.py, contrastive_corpus.py) and AGE / DEMOGRAPHIC span-loss
upweighting (prompting.token_span_weights, the Dataset and the collate function). Offline: no Hub access.
"""

import random
import re

import pytest
import torch

from redactx.data import clean_notes as cn
from redactx.data.clean_notes import (generate_clean_note, generate_clean_notes, generate_clinical_pair,
                                      generate_clinical_pairs)
from redactx.data.contrastive_corpus import (load_contrastive_corpus, load_hard_negatives, make_record,
                                             parse_category_weights)
from redactx.data.prompting import build_noul_prompt, token_raw_ranges, token_span_labels, token_span_weights

_WORD = lambda w: re.compile(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])")  # noqa: E731


# ---------------------------------------------------------------- synthetic clinical notes
def test_clinical_pairs_spans_are_exact_and_in_bounds():
    for d in generate_clinical_pairs(300, seed=11, max_chars=600):
        text = d["text"]
        assert 0 < len(text) <= 600 and d["spans"]
        prev_end = -1
        for s, e, lab in d["spans"]:
            assert 0 <= s < e <= len(text) and s >= prev_end, (s, e, lab)
            assert text[s:e] == text[s:e].strip() and text[s:e]
            prev_end = e
        # AGE spans are the number only
        for s, e, lab in d["spans"]:
            if lab == "age":
                assert text[s:e].isdigit()


def test_twins_contain_no_identifier_from_the_positive():
    for d in generate_clinical_pairs(300, seed=12):
        twin = d["twin"]
        for s, e, lab in d["spans"]:
            value = d["text"][s:e]
            if lab == "age":
                assert not re.search(r"\b" + value + r"[- ]?(?:year|y/o|yo)", twin)
            else:
                assert value not in twin, (lab, value, twin)


def test_clean_notes_have_no_identifiers_or_demographics():
    banned = [_WORD(w) for w in cn.SEX + cn.ETHNICITY + cn.RELIGION + cn.MARITAL + cn.MONTHS]
    banned += [_WORD(w) for w in cn.FIRST + cn.LAST + [c for c, _, _ in cn.CITIES]]
    for t in generate_clean_notes(300, seed=13):
        assert t and len(t) <= 600
        assert not re.search(r"\d+[- ]?(?:year|y/o|yo\b)", t), t
        # blood pressures ("BP 98/64") are deliberate date-like hard negatives; nothing else may look like a date
        assert not re.search(r"(?<!BP )\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b", t), t
        assert "@" not in t
        for rx in banned:
            assert not rx.search(t), (rx.pattern, t)


def test_vocabulary_is_disjoint_from_held_out_sets():
    """Training on these notes must not leak the generator benchmark (G) or the handwritten set (H)."""
    from redactx.data import generator as g
    from validate_model import HELD_OUT_CLEAN, HELD_OUT_PHI
    gen_names = set(g.FIRST_NAMES) | set(g.LAST_NAMES)
    assert not (set(cn.FIRST) | set(cn.LAST)) & gen_names
    assert not set(cn.FACILITIES) & set(g.FACILITIES)
    assert not set(cn.CONDITIONS) & set(g.CONDITIONS)
    h_text = " ".join(str(x if isinstance(x, str) else x[0]) for x in list(HELD_OUT_PHI) + list(HELD_OUT_CLEAN))
    for name in cn.FIRST + cn.LAST + cn.FACILITIES:
        assert not _WORD(name).search(h_text), name


def test_generators_are_deterministic():
    assert generate_clinical_pairs(5, seed=3) == generate_clinical_pairs(5, seed=3)
    assert generate_clean_notes(5, seed=3) == generate_clean_notes(5, seed=3)
    assert generate_clinical_pair(random.Random(1)) != generate_clinical_pair(random.Random(2))
    assert generate_clean_note(random.Random(1)) != generate_clean_note(random.Random(2))


# ---------------------------------------------------------------- category weights
def test_parse_category_weights():
    assert parse_category_weights("AGE=3, demographic=2.5") == {"AGE": 3.0, "DEMOGRAPHIC": 2.5}
    assert parse_category_weights("") == {}
    for bad in ["AGE", "SHOESIZE=2", "AGE=0", "AGE=-1"]:
        with pytest.raises(ValueError):
            parse_category_weights(bad)


def test_token_span_weights_alignment():
    text = "Patient is 67 years old, Hispanic, lives in Tulsa."
    a, d = text.index("67"), text.index("Hispanic")
    spans = [(a, a + 2), (d, d + len("Hispanic"))]
    prompt, raw_index = build_noul_prompt(text)
    # character-level offsets (one "token" per prompt character) make the expectation exact
    offsets = [(i, i + 1) for i in range(len(prompt))]
    w = token_span_weights(offsets, raw_index, spans, [3.0, 2.0])
    labels = token_span_labels(offsets, raw_index, spans)
    ranges = token_raw_ranges(offsets, raw_index)
    for wt, lab, (rs, _) in zip(w, labels, ranges):
        if rs < 0:
            assert wt == 0.0 and lab == -100
        elif a <= rs < a + 2:
            assert wt == 3.0 and lab == 1
        elif d <= rs < d + len("Hispanic"):
            assert wt == 2.0 and lab == 1
        else:
            assert wt == 1.0 and lab == 0


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def test_dataset_emits_weights_and_collate_pads(tokenizer):
    from redactx.data.openjev_dataset import OpenJevCalibratedDataset, collate_openjev_batch
    text = "Patient is 67 years old and a retired machinist from Tulsa. " * 3
    a = text.index("67")
    o = text.index("retired machinist")
    pos = make_record("p", text, [(a, a + 2, "age"), (o, o + 17, "occupation")], True, "test", "positive",
                      span_weights=[3.0, 2.0])
    neg = make_record("n", "Vitals stable, plan unchanged.", [], False, "test", "contrastive_negative")
    assert "pii_span_weights" in pos and "pii_span_weights" not in neg
    ds = OpenJevCalibratedDataset([pos, neg], tokenizer, max_length=256)
    p, n = ds[0], ds[1]
    assert "token_span_weights" in p and "token_span_weights" not in n
    w, lab = p["token_span_weights"], p["token_span_labels"]
    assert w.shape == lab.shape
    assert torch.all(w[lab == -100] == 0.0)
    assert torch.all(w[lab == 0] == 1.0)
    assert set(w[lab == 1].tolist()) <= {3.0, 2.0} and 3.0 in w[lab == 1].tolist()
    batch = collate_openjev_batch([p, n])
    bw = batch["token_span_weights"]
    assert bw.shape == batch["token_span_labels"].shape
    n_len = n["token_span_labels"].size(0)
    assert torch.all(bw[1, :n_len] == 1.0) and torch.all(bw[1, n_len:] == 0.0)


def test_dataset_truncation_keeps_weights_aligned(tokenizer):
    from redactx.data.openjev_dataset import OpenJevCalibratedDataset
    filler = "Labs notable for creatinine 1.2 mg/dL and hemoglobin 11.4 g/dL. " * 12
    text = "Patient is 91 years old. " + filler + "Ethnicity: Haitian."
    a = text.index("91")
    h = text.index("Haitian")
    rec = make_record("p", text, [(a, a + 2, "age"), (h, h + 7, "race_ethnicity")], True, "t", "positive",
                      span_weights=[3.0, 2.0])
    item = OpenJevCalibratedDataset([rec], tokenizer, max_length=96)[0]
    assert item["input_ids"].size(0) <= 96
    w, lab = item["token_span_weights"], item["token_span_labels"]
    assert w.shape == lab.shape
    assert 3.0 in w[lab == 1].tolist()          # the age survived the cut
    assert 2.0 not in w.tolist()                # the cut-off ethnicity span left no weight behind


def test_trainer_span_loss_uses_weights():
    """span_bce (the trainer's span loss): ignores -100, applies per-token weights, and weight 1 = unweighted."""
    import torch.nn.functional as F
    from redactx.training.openjev_trainer import span_bce
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, 9.0]])
    labels = torch.tensor([[1, 0, 1, 0, -100]])
    weights = torch.tensor([[3.0, 1.0, 1.0, 1.0, 0.0]])
    got = span_bce(logits, labels, weights, pos_weight=2.0)
    per = F.binary_cross_entropy_with_logits(logits[0, :4], labels[0, :4].float(), reduction="none",
                                             pos_weight=torch.tensor(2.0))
    assert torch.isclose(got, (per * weights[0, :4]).mean())
    ones = span_bce(logits, labels, torch.ones_like(weights), pos_weight=2.0)
    assert torch.isclose(ones, span_bce(logits, labels, None, pos_weight=2.0))
    assert got > ones          # upweighting a misclassified-ish PHI token raises the loss


# ---------------------------------------------------------------- corpus assembly (offline)
def _pii_docs():
    docs = []
    for i in range(20):
        name = f"{cn.FIRST[i]} {cn.LAST[i]}"
        age = str(30 + i)
        text = f"Contact {name}, age {age}, at {name.split()[0].lower()}@example.org about invoice {1000 + i}."
        s = text.index(name)
        a = text.index(f"age {age}") + 4
        e = text.index("@") - len(name.split()[0])
        spans = [(s, s + len(name), "name"), (a, a + len(age), "age"),
                 (e, e + len(name.split()[0]) + len("@example.org"), "email")]
        docs.append({"source": "nemotron" if i % 2 else "gretel", "text": text, "spans": spans})
    return docs


def test_hard_negatives_offline_are_synthetic_and_clean():
    docs = load_hard_negatives(25, token=None, seed=5, allow_download=False)
    assert len(docs) == 25 and {d["source"] for d in docs} == {"synthetic_clean_note"}
    assert all(d["spans"] == [] and d["text"] for d in docs)


def test_contrastive_corpus_with_hard_negatives_pairs_and_weights():
    train, val, stats = load_contrastive_corpus(
        num_pii_docs=20, natural_negative_ratio=0.0, max_chars=600, val_fraction=0.1, seed=7,
        hard_negative_ratio=0.75, clinical_pair_ratio=0.5,
        span_category_weights=parse_category_weights("AGE=3,DEMOGRAPHIC=3"),
        oversample_categories=("AGE", "DEMOGRAPHIC"), oversample_factor=2,
        _pii_docs=_pii_docs(), _natural_docs=[], _allow_download=False)
    assert stats["hard_negative_docs"] == 15 and stats["clinical_pair_docs"] == 10
    assert stats["hard_negative_sources"] == {"synthetic_clean_note": 15}
    assert stats["span_category_weights"] == {"AGE": 3.0, "DEMOGRAPHIC": 3.0}
    assert stats["oversample"]["train_records_added"] > 0
    assert stats["train_gold_spans_by_category"].get("AGE", 0) > 0
    recs = train + val
    kinds = {r["metadata"]["kind"] for r in recs}
    assert {"positive", "contrastive_negative", "hard_negative"} <= kinds
    assert "positive_oversampled" in {r["metadata"]["kind"] for r in train}
    assert "positive_oversampled" not in {r["metadata"]["kind"] for r in val}    # never in validation
    for r in recs:
        assert "pii_span_weights" in r and len(r["pii_span_weights"]) == len(r["pii_spans"])
        if not r["metadata"]["is_phi"]:
            assert r["pii_spans"] == []
    # an AGE span carries weight 3, a name span weight 1
    pos = [r for r in recs if r["metadata"]["is_phi"] and r["metadata"]["source"] in ("nemotron", "gretel")]
    r = pos[0]
    by_text = {r["text"][s:e]: w for (s, e), w in zip(r["pii_spans"], r["pii_span_weights"])}
    assert any(v == 3.0 and k.isdigit() for k, v in by_text.items())
    assert any(v == 1.0 and " " in k for k, v in by_text.items())
    # synthetic clinical positives use their natural twin, not a generic replacement
    clin_negs = [r["text"] for r in recs if r["metadata"]["source"] == "synthetic_clinical"
                 and not r["metadata"]["is_phi"]]
    assert clin_negs and not any("the details" in t for t in clin_negs)


def test_contrastive_corpus_defaults_unchanged():
    """Without the new options the corpus has no hard negatives, pairs, weights or duplicates (v2 behaviour)."""
    train, val, stats = load_contrastive_corpus(num_pii_docs=20, natural_negative_ratio=0.0, seed=7,
                                                _pii_docs=_pii_docs(), _natural_docs=[], _allow_download=False)
    assert stats["hard_negative_docs"] == 0 and stats["clinical_pair_docs"] == 0
    assert all("pii_span_weights" not in r for r in train + val)
    assert len(train) + len(val) == 40
