"""
n2c2 2014 loader (redactx/data/n2c2.py), benchmark metrics (benchmark_n2c2.py) and the n2c2 calibration source
(calibrate_thresholds.py). The corpus itself is DUA-restricted, so these tests write records in the real
deIdi2b2 XML schema to a temporary corpus folder.
"""

import os

import pytest

from redactx.data.n2c2 import TYPE_MAP, find_split_dirs, hipaa_flag, load_n2c2, parse_record, windows_with_spans
from redactx.production.hipaa import Category

NOTE = ("Record date: 2067-05-03\n\nPatient Wilma Hargrove, 93 year old, seen at Fairview Hospital by Dr. Lee.\n"
        "MRN: 4417290. Phone (617) 555-0123. Works as a carpenter in Boston, MA.\n"
        + "Assessment: stable chronic kidney disease, continue lisinopril and recheck labs. " * 12)


def _tag(tag, typ, value, i, start=None, text=NOTE):
    s = text.index(value) if start is None else start
    return f'<{tag} id="P{i}" start="{s}" end="{s + len(value)}" text="{value}" TYPE="{typ}" comment="" />'


def _xml(note=NOTE, tags=()):
    return ('<?xml version="1.0" encoding="UTF-8" ?>\n<deIdi2b2>\n<TEXT><![CDATA[' + note + ']]></TEXT>\n<TAGS>\n'
            + "\n".join(tags) + '\n</TAGS>\n</deIdi2b2>\n')


TAGS = [
    _tag("DATE", "DATE", "2067-05-03", 0),
    _tag("NAME", "PATIENT", "Wilma Hargrove", 1),
    _tag("AGE", "AGE", "93", 2),
    _tag("LOCATION", "HOSPITAL", "Fairview Hospital", 3),
    _tag("NAME", "DOCTOR", "Lee", 4),
    _tag("ID", "MEDICALRECORD", "4417290", 5),
    _tag("CONTACT", "PHONE", "(617) 555-0123", 6),
    _tag("PROFESSION", "PROFESSION", "carpenter", 7),
    _tag("LOCATION", "CITY", "Boston", 8),
    _tag("LOCATION", "STATE", "MA", 9, start=NOTE.index("Boston, MA") + 8),
]


def test_type_map_and_hipaa_flags():
    assert TYPE_MAP["PATIENT"] == (Category.NAME, True)
    assert TYPE_MAP["DOCTOR"] == (Category.NAME, False)
    assert TYPE_MAP["MEDICALRECORD"] == (Category.MRN, True)
    assert hipaa_flag("AGE", "93") and not hipaa_flag("AGE", "67") and not hipaa_flag("AGE", "89")
    assert not hipaa_flag("STATE", "MA") and hipaa_flag("CITY", "Boston") and not hipaa_flag("HOSPITAL", "x")


def test_parse_record_exact_relocated_dropped():
    shifted = _tag("NAME", "PATIENT", "Wilma Hargrove", 20, start=NOTE.index("Wilma Hargrove") + 7)  # offsets off by 7
    missing = '<NAME id="P21" start="5" end="12" text="Nobody Here" TYPE="PATIENT" comment="" />'
    broken = '<DATE id="P22" start="x" end="y" text="2067" TYPE="DATE" comment="" />'
    doc, rep = parse_record(_xml(tags=TAGS + [shifted, missing, broken]), from_string=True)
    assert rep == {"tags": 13, "exact": 10, "relocated": 1, "dropped": 2}
    assert doc["text"] == NOTE
    for sp in doc["spans"]:
        assert NOTE[sp["start"]:sp["end"]] == sp["text"]
    by_type = {sp["type"]: sp for sp in doc["spans"]}
    assert by_type["AGE"]["hipaa"] is True and by_type["AGE"]["category"] == "AGE"
    assert by_type["DOCTOR"]["hipaa"] is False and by_type["STATE"]["category"] == "COARSE_LOCATION"
    assert [sp["start"] for sp in doc["spans"]] == sorted(sp["start"] for sp in doc["spans"])


def test_parse_record_requires_text():
    with pytest.raises(ValueError):
        parse_record("<deIdi2b2><TAGS/></deIdi2b2>", from_string=True)


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "n2c2-2014"
    for split_dir, n in [("training-PHI-Gold-Set1", 3), ("training-PHI-Gold-Set2", 2), ("testing-PHI-Gold-fixed", 2),
                         ("testing-PHI-Gold", 1)]:
        d = root / "unpacked" / split_dir
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{100 + i}-0{i}.xml").write_text(_xml(tags=TAGS), encoding="utf-8")
    return str(root)


def test_split_discovery_and_loading(corpus, tmp_path):
    train_dirs = find_split_dirs(corpus, "train")
    assert [os.path.basename(d) for d in train_dirs] == ["training-PHI-Gold-Set1", "training-PHI-Gold-Set2"]
    assert [os.path.basename(d) for d in find_split_dirs(corpus, "test")] == ["testing-PHI-Gold-fixed"]
    docs, rep = load_n2c2(corpus, "train")
    assert rep["files"] == 5 and rep["tags"] == 5 * len(TAGS) and rep["dropped"] == 0
    docs, rep = load_n2c2(corpus, "test", limit=1)
    assert len(docs) == 1
    with pytest.raises(FileNotFoundError, match="Data Use Agreement"):
        load_n2c2(str(tmp_path / "empty"), "test")
    with pytest.raises(ValueError):
        find_split_dirs(corpus, "dev")


def test_windows_with_spans_rebase_and_clip():
    doc, _ = parse_record(_xml(tags=TAGS), from_string=True)
    wins = windows_with_spans(doc, max_chars=200)
    assert len(wins) > 1
    covered = set()
    for w in wins:
        assert NOTE[w["offset"]:w["offset"] + len(w["text"])] == w["text"]
        for s, e, cat in w["spans"]:
            assert 0 <= s < e <= len(w["text"])
            covered.update(range(w["offset"] + s, w["offset"] + e))
    gold = {i for sp in doc["spans"] for i in range(sp["start"], sp["end"])}
    assert gold <= covered


# ---------------------------------------------------------------- benchmark metrics
def test_benchmark_metrics_on_parsed_note():
    from benchmark_n2c2 import hipaa_recall, per_type_recall, window_metrics
    from redactx.production.detectors import validator_findings
    from redactx.production.validators import StructuredIdValidator
    doc, _ = parse_record(_xml(tags=TAGS), from_string=True)
    doc["validators"] = validator_findings(doc["text"], StructuredIdValidator())
    doc["all"] = [(sp["start"], sp["end"]) for sp in doc["spans"]]
    doc["none"] = []

    t = per_type_recall([doc], "validators")
    assert t["MEDICALRECORD"]["touched_recall"] == 1.0 and t["PHONE"]["touched_recall"] == 1.0
    assert t["PATIENT"]["touched_recall"] == 0.0          # validators do not find names
    assert per_type_recall([doc], "all")["PATIENT"]["char_recall"] == 1.0

    h_all, h_val = hipaa_recall([doc], "all"), hipaa_recall([doc], "validators")
    n_hipaa = sum(1 for sp in doc["spans"] if sp["hipaa"])   # DATE, PATIENT, AGE>89, MRN, PHONE, CITY
    assert h_all == {"n_spans": n_hipaa, "touched_recall": 1.0, "char_recall": 1.0}
    assert h_val["touched_recall"] == round(2 / n_hipaa, 4)

    w_all, w_none = window_metrics([doc], "all", 200), window_metrics([doc], "none", 200)
    assert w_all["recall"] == 1.0 and w_all["specificity"] == 1.0
    assert w_none["recall"] == 0.0 and w_none["specificity"] == 1.0
    assert w_all["phi_windows"] >= 1 and w_all["clean_windows"] >= 1


def test_benchmark_runs_validators_only(corpus, tmp_path):
    from benchmark_n2c2 import main
    out = tmp_path / "n2c2_results.json"
    main(["--n2c2-dir", corpus, "--split", "test", "--skip-redactx", "--skip-presidio", "--out", str(out)])
    import json
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_docs"] == 2 and set(res["chars_all_phi"]) == {"validators"}
    assert res["per_n2c2_type_recall"]["validators"]["MEDICALRECORD"]["touched_recall"] == 1.0
    assert res["windows"]["validators"]["specificity"] == 1.0


# ---------------------------------------------------------------- calibration source
def test_n2c2_calibration_source_uses_train_split_only(corpus):
    from calibrate_thresholds import load_n2c2_calibration
    pos, neg, rep = load_n2c2_calibration(corpus, n_pos=50, max_chars=200)
    assert rep["files"] == 5 and all(os.path.basename(d).startswith("training") for d in rep["dirs"])
    assert pos and all(p["source"] == "n2c2" for p in pos)
    for p in pos:
        for s, e, _ in p["spans"]:
            assert 0 <= s < e <= len(p["text"])
    # first len(pos) negatives are the twins: no gold PHI value survives in them
    for p, twin in zip(pos, neg):
        for s, e, _ in p["spans"]:
            value = p["text"][s:e]
            if len(value) > 3:
                assert value not in twin
    assert len(neg) == len(pos) + min(rep["windows_without_phi"], 50 // 2)
    # deterministic for a seed
    assert load_n2c2_calibration(corpus, 50, 200)[0] == pos


def test_calibrate_cli_requires_n2c2_dir():
    from calibrate_thresholds import main
    with pytest.raises(SystemExit):
        main(["--model-dir", ".", "--sources", "n2c2"])
    with pytest.raises(SystemExit):
        main(["--model-dir", ".", "--sources", "generator,i2b2"])
