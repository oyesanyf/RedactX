"""
Tests for the production layer (CPU only).

Real components only: real Presidio (spaCy en_core_web_lg), real tokenizers, and a real gpt2 backbone saved
as a RedactX checkpoint. The gpt2 checkpoint's span head is freshly initialised (untrained), so engine tests
assert structural invariants (offsets, batching parity, stitching, API behaviour) - never detection quality.
"""

import json
import os
import random

import pytest

from redactx.production.chunking import Window, chunk_text, merge_intervals
from redactx.production.detectors import Detection, Finding, merge_findings
from redactx.production.hipaa import Category, HIPAA_IDENTIFIERS, is_hipaa_identifier, to_category
from redactx.production.redactor import REDACTED_DOCUMENT, Redactor
from redactx.production.settings import Settings, hash_api_key, read_dotenv
from redactx.production.thresholds import (ThresholdConfig, conservative_threshold_for_recall, operating_point,
                                           threshold_for_recall, wilson_lower_bound)

presidio_available = pytest.importorskip  # alias for readability below


# ============================================================================ chunking
def _long_text(n_notes=30, seed=7):
    from redactx.data.generator import generate_redactx_corpus
    notes = generate_redactx_corpus(num_samples=n_notes, phi_ratio=0.7, seed=seed)
    return "\n".join(r["context"] for r in notes)


@pytest.mark.parametrize("max_chars,overlap", [(600, 150), (300, 100), (200, 0), (128, 64)])
def test_chunk_windows_cover_text_and_respect_limits(max_chars, overlap):
    text = _long_text()
    wins = chunk_text(text, max_chars, overlap)
    assert wins[0].start == 0 and wins[-1].end == len(text)
    for a, b in zip(wins, wins[1:]):
        assert b.start <= a.end - overlap, "overlap guarantee violated"
        assert b.start > a.start, "no progress"
    assert all(0 < w.end - w.start <= max_chars for w in wins)


def test_chunk_overlap_guarantee_every_short_span_fits_some_window():
    text = _long_text(12, seed=3)
    max_chars, overlap = 250, 80
    wins = chunk_text(text, max_chars, overlap)
    for a in range(0, len(text) - overlap):
        b = a + overlap
        assert any(w.start <= a and b <= w.end for w in wins), (a, b)


def test_chunk_short_text_single_window_and_validation():
    assert chunk_text("short", 600, 150) == [Window(0, 5)]
    assert chunk_text("", 600, 150) == [Window(0, 0)]
    with pytest.raises(ValueError):
        chunk_text("x" * 1000, 100, 60)
    with pytest.raises(ValueError):
        chunk_text("x", 0, 0)


def test_chunk_no_whitespace_text():
    text = "A" * 2000
    wins = chunk_text(text, 600, 150)
    assert wins[-1].end == 2000 and all(w.end - w.start <= 600 for w in wins)


def test_merge_intervals():
    assert merge_intervals([(5, 8), (0, 3), (2, 4), (8, 9), (20, 20)]) == [(0, 4), (5, 9)]
    assert merge_intervals([(0, 3), (4, 6)], join_gap=1) == [(0, 6)]


# ============================================================================ thresholds
def test_threshold_for_recall_matches_definition():
    rng = random.Random(11)
    for _ in range(200):
        pos = [rng.random() for _ in range(rng.randint(1, 60))]
        target = rng.choice([0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
        t = threshold_for_recall(pos, target)
        recall = sum(s >= t for s in pos) / len(pos)
        assert recall >= target
        higher = [s for s in pos if s > t]
        if higher:  # the next larger candidate threshold would miss the target
            t2 = min(higher)
            assert sum(s >= t2 for s in pos) / len(pos) < target


def test_wilson_lower_bound_values():
    z2 = 1.6448536269514722 ** 2
    # all successes: closed form 1 / (1 + z^2 / n)
    assert wilson_lower_bound(450, 450, 0.95) == pytest.approx(1 / (1 + z2 / 450), abs=1e-12)
    assert wilson_lower_bound(446, 450, 0.95) == pytest.approx(0.98035, abs=2e-5)
    assert wilson_lower_bound(444, 450, 0.95) < 0.98 < wilson_lower_bound(446, 450, 0.95)
    assert wilson_lower_bound(7, 10, 0.5) == pytest.approx(0.7)          # 50% confidence = point estimate
    assert wilson_lower_bound(0, 10, 0.95) == 0.0
    with pytest.raises(ValueError):
        wilson_lower_bound(1, 0)


def test_conservative_threshold_is_lower_and_covers_true_recall():
    """
    Scores ~ Uniform(0, 1), so the TRUE recall at threshold t is exactly 1 - t. A 95% lower-bound threshold must
    reach the target on the true distribution in about 95% of samples; the point estimate only in about half.
    """
    rng = random.Random(7)
    target, n, trials = 0.9, 300, 400
    cover_cons = cover_point = 0
    for _ in range(trials):
        pos = [rng.random() for _ in range(n)]
        t_point = threshold_for_recall(pos, target)
        t_cons, lb, ok = conservative_threshold_for_recall(pos, target, 0.95)
        assert ok and t_cons <= t_point and lb >= target
        cover_cons += (1 - t_cons) >= target
        cover_point += (1 - t_point) >= target
    assert cover_cons / trials >= 0.92
    assert cover_point / trials <= 0.70
    # too few positives to certify a high target -> flags every positive and says so
    t, lb, ok = conservative_threshold_for_recall([0.9, 0.8, 0.7], 0.99, 0.95)
    assert not ok and t == 0.7 and lb < 0.99
    # ties: the threshold is always one of the scores and counts every tied positive
    t, lb, ok = conservative_threshold_for_recall([0.9] * 50 + [0.1] * 2, 0.5, 0.95)
    assert t == 0.9 and ok


def test_operating_point_and_threshold_file(tmp_path):
    op = operating_point([0.9, 0.8, 0.2], [0.1, 0.85], 0.5)
    assert op["recall"] == round(2 / 3, 4) and op["specificity"] == 0.5 and op["precision"] == round(2 / 3, 4)
    cfg = ThresholdConfig(doc_threshold=0.31, span_threshold=0.27, target_recall=0.98, calibrated_on="unit")
    cfg.save(str(tmp_path))
    back = ThresholdConfig.load(str(tmp_path))
    assert back.doc_threshold == 0.31 and back.span_threshold == 0.27 and back.target_recall == 0.98
    assert ThresholdConfig.load(str(tmp_path / "missing")).doc_threshold == 0.5
    (tmp_path / "redactx_thresholds.json").write_text(json.dumps({"doc_threshold": 3}))
    with pytest.raises(ValueError):
        ThresholdConfig.load(str(tmp_path))


# ============================================================================ HIPAA mapping
OBSERVED_LABELS = {
    "nemotron": ['company_name', 'date', 'first_name', 'last_name', 'url', 'email', 'occupation', 'phone_number',
                 'time', 'country', 'state', 'street_address', 'city', 'customer_id', 'date_time',
                 'biometric_identifier', 'employee_id', 'education_level', 'county', 'certificate_license_number',
                 'user_name', 'account_number', 'date_of_birth', 'coordinate', 'vehicle_identifier', 'ipv4',
                 'postcode', 'employment_status', 'credit_debit_card', 'password', 'fax_number', 'license_plate',
                 'medical_record_number', 'pin', 'bank_routing_number', 'ssn', 'race_ethnicity',
                 'device_identifier', 'mac_address', 'health_plan_beneficiary_number', 'swift_bic', 'api_key',
                 'language', 'http_cookie', 'religious_belief', 'gender', 'ipv6', 'age', 'blood_type', 'cvv',
                 'political_view', 'tax_id', 'unique_id', 'sexuality'],
    "gretel": ['name', 'address', 'credit_card_number', 'unique_identifier', 'national_id'],
    "ai4privacy": ['GIVENNAME', 'DATE', 'SURNAME', 'EMAIL', 'CITY', 'TITLE', 'TELEPHONENUM', 'AGE', 'BUILDINGNUM',
                   'STREET', 'ZIPCODE', 'IDCARDNUM', 'SEX', 'DRIVERLICENSENUM', 'CREDITCARDNUMBER', 'TAXNUM',
                   'GENDER', 'PASSPORTNUM', 'SOCIALNUM'],
    "presidio": ['PERSON', 'LOCATION', 'DATE_TIME', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'US_SSN', 'IP_ADDRESS', 'URL',
                 'CREDIT_CARD', 'IBAN_CODE', 'US_BANK_NUMBER', 'US_DRIVER_LICENSE', 'US_PASSPORT', 'US_ITIN',
                 'MEDICAL_LICENSE', 'NRP', 'CRYPTO', 'UK_NHS'],
    "generator": ['NAME', 'LOCATION', 'DATE', 'IDENTIFIER', 'CONTACT'],
}


def test_every_observed_label_maps_to_a_known_category():
    for source, labels in OBSERVED_LABELS.items():
        for lab in labels:
            assert to_category(lab) != Category.UNKNOWN, (source, lab)


def test_specific_hipaa_mappings():
    assert to_category("medical_record_number") == Category.MRN
    assert to_category("health_plan_beneficiary_number") == Category.HEALTH_PLAN_ID
    assert to_category("SOCIALNUM") == Category.SSN and to_category("US_SSN") == Category.SSN
    assert to_category("fax_number") == Category.FAX
    assert to_category("license_plate") == Category.VEHICLE_ID
    assert to_category("PERSON") == Category.NAME and to_category("GIVENNAME") == Category.NAME
    assert to_category("state") == Category.COARSE_LOCATION and not is_hipaa_identifier("state")
    assert is_hipaa_identifier("date_of_birth") and not is_hipaa_identifier("gender")
    assert to_category("") == Category.UNKNOWN
    assert len(HIPAA_IDENTIFIERS) == 19


# ============================================================================ merging + redaction
TEXT = "Please call Jennifer Okonkwo at 312-555-0198 about her biopsy results."


def _f(text, s, e, cat, src, score=0.9):
    return Finding(s, e, text[s:e], cat, score, (src,))


def test_merge_findings_prefers_presidio_category_and_unions_bounds():
    a = TEXT.index("Jennifer")
    merged = merge_findings(TEXT, [
        _f(TEXT, a, a + 8, Category.OTHER_ID, "redactx", 0.7),
        _f(TEXT, a, a + len("Jennifer Okonkwo"), Category.NAME, "presidio", 0.85),
    ])
    assert len(merged) == 1
    m = merged[0]
    assert (m.start, m.end, m.text) == (a, a + 16, "Jennifer Okonkwo")
    assert m.category == Category.NAME and m.sources == ("presidio", "redactx") and m.score == 0.85


@pytest.fixture(scope="module")
def presidio_detector():
    pytest.importorskip("presidio_analyzer")
    from redactx.production.detectors import PresidioDetector
    return PresidioDetector()


def test_presidio_detector_real_findings(presidio_detector):
    d = presidio_detector.detect(TEXT)
    cats = {f.category for f in d.findings}
    assert d.contains_phi and Category.NAME in cats and Category.PHONE in cats
    for f in d.findings:
        assert TEXT[f.start:f.end] == f.text


def test_redaction_strategies(presidio_detector):
    key = b"k" * 32
    r = Redactor(presidio_detector, strategy="tag").redact(TEXT)
    assert r.action == "REDACTED" and "Jennifer" not in r.redacted_text and "312-555-0198" not in r.redacted_text
    assert "[NAME]" in r.redacted_text and "[PHONE]" in r.redacted_text
    assert r.category_counts.get("NAME") == 1

    c = Redactor(presidio_detector, strategy="char").redact(TEXT)
    assert len(c.redacted_text) == len(TEXT) and "Jennifer" not in c.redacted_text

    m = Redactor(presidio_detector, strategy="mask").redact(TEXT)
    assert "[REDACTED]" in m.redacted_text

    p1 = Redactor(presidio_detector, strategy="pseudonym", hmac_key=key)
    a = p1.redact(TEXT).redacted_text
    b = p1.redact("Jennifer Okonkwo called back on 312-555-0198.").redacted_text
    tok = [t for t in a.split() if t.startswith("[NAME_")][0]
    assert tok in b, "same value must map to the same pseudonym under one key"
    other = Redactor(presidio_detector, strategy="pseudonym", hmac_key=b"z" * 32).redact(TEXT).redacted_text
    assert tok not in other

    # NB: Presidio's spaCy NER tags words like "quarterly" as DATE_TIME, so the clean text avoids time words.
    clean = Redactor(presidio_detector).redact("The pump pressure stayed within the expected range.")
    assert clean.action == "PASS", [(f.text, f.category) for f in clean.findings]

    with pytest.raises(ValueError):
        Redactor(presidio_detector, strategy="pseudonym")
    with pytest.raises(ValueError):
        Redactor(presidio_detector, strategy="bogus")


def test_unlocalized_policy_fails_closed(presidio_detector):
    det = Detection(contains_phi=True, findings=[], doc_score=0.97, unlocalized_phi=True, detectors=["redactx"])
    r = Redactor(presidio_detector, unlocalized_policy="redact_all").apply("Some text with PHI", det)
    assert r.action == "REDACTED_DOCUMENT" and r.redacted_text == REDACTED_DOCUMENT
    r2 = Redactor(presidio_detector, unlocalized_policy="review").apply("Some text with PHI", det)
    assert r2.action == "REVIEW"
    # review never returns a localized identifier in clear text (hybrid: Presidio found the phone number)
    text = "Call 312-555-0198 about the results."
    det2 = Detection(contains_phi=True, findings=[_f(text, 5, 17, Category.PHONE, "presidio")], doc_score=0.97,
                     unlocalized_phi=True, detectors=["redactx", "presidio"])
    r3 = Redactor(presidio_detector, unlocalized_policy="review").apply(text, det2)
    assert r3.action == "REVIEW" and "312-555-0198" not in r3.redacted_text and "[PHONE]" in r3.redacted_text


def test_hipaa_only_skips_non_identifiers(presidio_detector):
    text = "Jane Roe works at Contoso as an engineer."
    det = Detection(contains_phi=True, findings=[
        _f(text, 0, 8, Category.NAME, "presidio"), _f(text, 18, 25, Category.ORGANIZATION, "presidio")])
    r = Redactor(presidio_detector, hipaa_only=True).apply(text, det)
    assert r.redacted_text == "[NAME] works at Contoso as an engineer."


# ============================================================================ settings
def test_settings_validation(tmp_path):
    good_hash = hash_api_key("secret-key")
    s = Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_API_KEY_HASHES": good_hash}, dotenv_path=None)
    assert s.mode == "presidio" and s.api_key_hashes == (good_hash,)
    with pytest.raises(ValueError, match="No API keys"):
        Settings.from_env(env={"REDACTX_MODE": "presidio"}, dotenv_path=None)
    with pytest.raises(ValueError, match="SHA-256"):
        Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_API_KEY_HASHES": "plaintext"}, dotenv_path=None)
    with pytest.raises(ValueError, match="CORS"):
        Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_ALLOW_NO_AUTH": "1",
                               "REDACTX_CORS_ORIGINS": "*"}, dotenv_path=None)
    with pytest.raises(ValueError, match="MODEL_DIR"):
        Settings.from_env(env={"REDACTX_MODE": "hybrid", "REDACTX_ALLOW_NO_AUTH": "1"}, dotenv_path=None)
    # an existing directory that is not an exported checkpoint is refused (no silent fallback model)
    with pytest.raises(ValueError, match="not an exported RedactX checkpoint"):
        Settings.from_env(env={"REDACTX_MODE": "hybrid", "REDACTX_ALLOW_NO_AUTH": "1",
                               "REDACTX_MODEL_DIR": str(tmp_path)}, dotenv_path=None)
    with pytest.raises(ValueError, match="pseudonym"):
        Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_ALLOW_NO_AUTH": "1",
                               "REDACTX_STRATEGY": "pseudonym"}, dotenv_path=None)
    dev = Settings.from_env(env={"REDACTX_MODE": "presidio", "REDACTX_ALLOW_NO_AUTH": "true"}, dotenv_path=None)
    assert dev.allow_no_auth and dev.warnings
    env_file = tmp_path / ".env"
    env_file.write_text(f"REDACTX_MODE=presidio\nREDACTX_API_KEY_HASHES={good_hash}\n# comment\n")
    assert read_dotenv(str(env_file))["REDACTX_MODE"] == "presidio"
    s2 = Settings.from_env(env={}, dotenv_path=str(env_file))
    assert s2.mode == "presidio"
    # process env wins over .env
    s3 = Settings.from_env(env={"REDACTX_MAX_BATCH": "5"}, dotenv_path=str(env_file))
    assert s3.max_batch == 5
