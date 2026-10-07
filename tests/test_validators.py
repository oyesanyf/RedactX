"""
Structured-identifier validators (redactx/production/validators.py): checksums, format rules, keyword requirements,
the clinical look-alikes they must NOT match, their merge into RedactXDetector, and the settings that configure them.
"""

import pytest

from redactx.production.hipaa import Category
from redactx.production.validators import (KINDS, StructuredIdValidator, email_valid, luhn_valid, mod11_valid,
                                           nanp_valid, npi_valid, parse_kinds, ssn_valid)


def found(v, text):
    return [(text[m.start:m.end], m.category, m.kind) for m in v.find(text)]


@pytest.fixture(scope="module", params=[True, False], ids=["libphonenumber", "nanp-rules"])
def validator(request):
    if request.param:
        pytest.importorskip("phonenumbers")
    return StructuredIdValidator(use_phonenumbers=request.param)


# ---------------------------------------------------------------- checksums / rules
def test_luhn_and_mod11():
    assert luhn_valid("79927398713") and not luhn_valid("79927398710")
    assert not luhn_valid("") and not luhn_valid("12a4")
    # mod-11 (weights 2..7 from the right): body 12345 -> 5*2+4*3+3*4+2*5+1*6 = 50, (11 - 50 % 11) % 11 = 5
    assert mod11_valid("123455") and not mod11_valid("123456")


def test_npi_cms_check_digit():
    assert npi_valid("1234567893")          # CMS published example
    assert not npi_valid("1234567890")      # wrong check digit
    assert not npi_valid("3234567893")      # must start with 1 or 2
    assert not npi_valid("123456789")


def test_ssn_issuance_rules():
    assert ssn_valid("123", "45", "6789")
    for a, g, s in [("000", "12", "3456"), ("666", "12", "3456"), ("912", "12", "3456"), ("123", "00", "4567"),
                    ("123", "45", "0000"), ("078", "05", "1120")]:
        assert not ssn_valid(a, g, s), (a, g, s)


def test_email_and_nanp_rules():
    assert email_valid("rosalind.abernathy@northmail.net")
    for bad in ["a..b@x.com", ".a@x.com", "a@x", "a@-x.com", "a@x.c0m", "a@@x.com"]:
        assert not email_valid(bad), bad
    assert nanp_valid("617", "555", "0123")
    assert not nanp_valid("117", "555", "0123") and not nanp_valid("617", "155", "0123")
    assert not nanp_valid("611", "555", "0123") and not nanp_valid("617", "411", "0123")


def test_parse_kinds():
    assert parse_kinds(None) == KINDS and parse_kinds("all") == KINDS
    assert parse_kinds("none") == ()
    assert parse_kinds("ssn, email") == ("ssn", "email")
    with pytest.raises(ValueError):
        parse_kinds("ssn,passport")
    with pytest.raises(ValueError):
        StructuredIdValidator(kinds=("ssn", "passport"))
    with pytest.raises(ValueError):
        StructuredIdValidator(mrn_checksum="crc32")


# ---------------------------------------------------------------- detection
def test_finds_verified_identifiers(validator):
    text = ("Pt seen 3/4. SSN 123-45-6789. Email rosalind.abernathy@northmail.net. Cell (617) 555-0123. "
            "MRN: 00482913. Referring NPI 1234567893.")
    got = found(validator, text)
    assert ("123-45-6789", Category.SSN, "ssn") in got
    assert ("rosalind.abernathy@northmail.net", Category.EMAIL, "email") in got
    assert ("(617) 555-0123", Category.PHONE, "phone") in got
    assert ("00482913", Category.MRN, "mrn") in got
    assert ("1234567893", Category.OTHER_ID, "npi") in got
    # offsets are exact
    for m in validator.find(text):
        assert text[m.start:m.end].strip() == text[m.start:m.end]


def test_rejects_clinical_lookalikes(validator):
    clean = ("Take metformin 500 mg twice daily; BP 128/82, HR 74. ICD-10 E11.9 and I10, CPT 99213. "
             "Dispensed NDC 0002-8215-01, lot AB12X. Follow up 2024-05-03 or 05/03/2024. Version 1.2.3.4 released. "
             "Ratio 3.5 to 1. Invalid SSN 666-12-3456. NPI 1234567890 is wrong. Order #1234567890.")
    assert found(validator, clean) == []


def test_bare_digit_runs_need_a_keyword(validator):
    assert found(validator, "Reference 123456789 attached.") == []                       # 9 digits, no SSN keyword
    assert found(validator, "Social security number: 123456789.") == [("123456789", Category.SSN, "ssn")]
    assert found(validator, "Batch 6175550123 shipped.") == []                           # 10 digits, no phone keyword
    got = found(validator, "Call 6175550123 tomorrow.")
    assert [(t, c) for t, c, _ in got] == [("6175550123", Category.PHONE)]
    assert found(validator, "1234567893 is the identifier") == []                         # NPI needs its keyword


def test_fax_vs_phone_nearest_keyword(validator):
    got = found(validator, "Fax: (617) 555-0123, phone (212) 555-0199")
    assert [(t, c) for t, c, _ in got] == [("(617) 555-0123", Category.FAX), ("(212) 555-0199", Category.PHONE)]


def test_surrogate_area_code_is_still_a_phone(validator):
    # 555 area code is not assigned (libphonenumber rejects it) but a correctly formatted NANP number still leaks
    got = found(validator, "Daughter reachable at 555-283-4417.")
    assert [(t, c) for t, c, _ in got] == [("555-283-4417", Category.PHONE)]


def test_mrn_keywords_min_digits_and_checksum():
    v = StructuredIdValidator(kinds=("mrn",))
    assert found(v, "MR# 7731942 on the wristband")[0][0] == "7731942"
    assert found(v, "Medical record number is H4417290.")[0][0] == "H4417290"
    assert found(v, "Patient MRN-123-45-6789 seen today.")[0][0] == "MRN-123-45-6789"   # attached keyword kept
    assert found(v, "MRN 1234 only") == []                                              # fewer than 5 digits
    assert found(v, "the record shows 00482913") == []                                  # no MRN keyword
    luhn = StructuredIdValidator(kinds=("mrn",), mrn_checksum="luhn")
    assert found(luhn, "MRN 79927398713")[0][0] == "79927398713"
    assert found(luhn, "MRN 79927398710") == []
    custom = StructuredIdValidator(kinds=("mrn",), mrn_checksum=lambda d: d.startswith("9"))
    assert found(custom, "MRN 9123456") and not found(custom, "MRN 8123456")


def test_kinds_subset():
    v = StructuredIdValidator(kinds=("email",))
    assert found(v, "SSN 123-45-6789, x@example.org") == [("x@example.org", Category.EMAIL, "email")]


# ---------------------------------------------------------------- detector merge (real gpt2-backed engine)
@pytest.fixture(scope="module")
def engine(gpt2_checkpoint):
    from redactx.models.openjev import OpenJevVaultGemmaEngine
    return OpenJevVaultGemmaEngine.from_pretrained(gpt2_checkpoint, device="cpu")


def test_detector_adds_validator_findings(engine):
    from redactx.production.detectors import RedactXDetector, merge_findings, validator_findings
    text = "Discharge note. SSN 123-45-6789 and NPI 1234567893 on file; email rosalind.abernathy@northmail.net."
    model_only = RedactXDetector(engine, max_chars=300, overlap=100, batch_size=4, validators=None).detect(text)
    with_val = RedactXDetector(engine, max_chars=300, overlap=100, batch_size=4).detect(text)
    assert "validator" in with_val.detectors and "validator" not in model_only.detectors
    # every validator span is covered by a finding carrying the validator source and the validator category
    v = StructuredIdValidator()
    for m in v.find(text):
        hits = [f for f in with_val.findings if f.start <= m.start and f.end >= m.end]
        assert hits, text[m.start:m.end]
        assert "validator" in hits[0].sources and hits[0].category == m.category
    # validators never remove model spans: every model character is still covered
    model_chars = {i for f in model_only.findings for i in range(f.start, f.end)}
    merged_chars = {i for f in with_val.findings for i in range(f.start, f.end)}
    assert model_chars <= merged_chars
    # the merge is the same as merging the two finding lists explicitly
    explicit = merge_findings(text, list(model_only.findings) + validator_findings(text, v))
    assert [(f.start, f.end) for f in explicit] == [(f.start, f.end) for f in with_val.findings]


def test_merge_validator_category_wins_over_model():
    from redactx.production.detectors import Finding, merge_findings
    text = "SSN 123-45-6789"
    model = Finding(4, 10, text[4:10], Category.PHONE, 0.7, ("redactx",))
    val = Finding(4, 15, text[4:15], Category.SSN, 0.99, ("validator",))
    (f,) = merge_findings(text, [model, val])
    assert (f.start, f.end, f.category) == (4, 15, Category.SSN)
    assert set(f.sources) == {"redactx", "validator"}


# ---------------------------------------------------------------- settings
def test_settings_validator_env():
    from redactx.production.settings import Settings
    base = {"REDACTX_MODE": "presidio", "REDACTX_ALLOW_NO_AUTH": "1"}
    s = Settings.from_env(env=base, dotenv_path=None)
    assert s.validators == KINDS and s.mrn_checksum is None and s.mrn_min_digits == 5
    s = Settings.from_env(env={**base, "REDACTX_VALIDATORS": "none"}, dotenv_path=None)
    assert s.validators == ()
    s = Settings.from_env(env={**base, "REDACTX_VALIDATORS": "ssn,mrn", "REDACTX_MRN_CHECKSUM": "LUHN",
                               "REDACTX_MRN_MIN_DIGITS": "7"}, dotenv_path=None)
    assert s.validators == ("ssn", "mrn") and s.mrn_checksum == "luhn" and s.mrn_min_digits == 7
    with pytest.raises(ValueError) as ei:
        Settings.from_env(env={**base, "REDACTX_VALIDATORS": "ssn,passport", "REDACTX_MRN_CHECKSUM": "crc",
                               "REDACTX_MRN_MIN_DIGITS": "0"}, dotenv_path=None)
    msg = str(ei.value)
    assert "REDACTX_VALIDATORS" in msg and "REDACTX_MRN_CHECKSUM" in msg and "REDACTX_MRN_MIN_DIGITS" in msg
