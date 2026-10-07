"""
Structured-identifier validators: verified detection of SSN, e-mail, phone, MRN and NPI.

These are NOT loose regexes. Every match must pass a structural check AND a validity rule, and the formats that are
ambiguous on their own (bare digit runs, record numbers) additionally need a context keyword in front of them:

  SSN    ddd-dd-dddd (same separator) + SSA issuance rules: area not 000 / 666 / 900-999, group not 00,
         serial not 0000, not a publicly voided number. Space-separated or bare 9 digits need an SSN keyword.
  EMAIL  local@domain + RFC 5321/1035 limits: local part <= 64 chars, no leading/trailing/double dots, domain
         <= 253 chars, 2+ labels of 1-63 [A-Za-z0-9-] not starting/ending with '-', alphabetic TLD (or xn--).
  PHONE  libphonenumber (`phonenumbers`, a Presidio dependency) with Leniency.VALID: the number must be valid
         against the numbering-plan metadata for its region. In addition (and without the library), separator-
         formatted numbers that pass the NANP rules (area code and exchange [2-9]XX, not N11) are accepted, so
         unassigned / surrogate area codes are not leaked. Bare 10-digit runs always need a phone keyword.
         Labelled FAX when the nearest preceding contact keyword is 'fax'.
  MRN    a medical-record keyword (MRN, MR#, medical record number, chart #, unit no., ...) followed by an ID with
         at least `mrn_min_digits` digits; optional facility checksum (Luhn / mod-11 / custom callable).
  NPI    an NPI keyword followed by 10 digits starting with 1 or 2 that pass the CMS Luhn check
         (Luhn over "80840" + the first 9 digits).

Validators only ever ADD or CONFIRM spans; they never remove a model span (for redaction a miss is a leak).
"""

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from redactx.production.hipaa import Category

KINDS = ("ssn", "email", "phone", "mrn", "npi")


@dataclass(frozen=True)
class IdMatch:
    start: int
    end: int
    category: Category
    kind: str
    evidence: str


# ============================================================================ checksums
def luhn_valid(digits: str) -> bool:
    """Luhn mod-10 check over a string of digits (the last digit is the check digit)."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mod11_valid(digits: str) -> bool:
    """ISO 7064-style mod-11 with weights 2..7 repeating from the right; check digit 10 is written as 'X'/'0'."""
    s = digits.upper()
    if len(s) < 2 or not s[:-1].isdigit() or not (s[-1].isdigit() or s[-1] == "X"):
        return False
    body, check = s[:-1], s[-1]
    total = sum(int(c) * (2 + i % 6) for i, c in enumerate(reversed(body)))
    r = (11 - total % 11) % 11
    expected = "X" if r == 10 else str(r)
    return check == expected or (r == 10 and check == "0")


def npi_valid(npi: str) -> bool:
    """CMS NPI check: 10 digits, first digit 1 or 2, Luhn over '80840' + the 10 digits."""
    return len(npi) == 10 and npi.isdigit() and npi[0] in "12" and luhn_valid("80840" + npi)


_VOIDED_SSNS = frozenset({"078051120", "219099999", "457555462"})  # publicly circulated, voided by the SSA


def ssn_valid(area: str, group: str, serial: str) -> bool:
    if not (len(area) == 3 and len(group) == 2 and len(serial) == 4 and (area + group + serial).isdigit()):
        return False
    a = int(area)
    if a == 0 or a == 666 or a >= 900:
        return False
    if group == "00" or serial == "0000":
        return False
    return (area + group + serial) not in _VOIDED_SSNS


_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def email_valid(address: str) -> bool:
    if address.count("@") != 1 or len(address) > 254:
        return False
    local, domain = address.split("@")
    if not local or len(local) > 64 or local[0] == "." or local[-1] == "." or ".." in local:
        return False
    if not re.fullmatch(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+", local):
        return False
    if len(domain) > 253:
        return False
    labels = domain.split(".")
    if len(labels) < 2 or not all(_LABEL.match(lab) for lab in labels):
        return False
    tld = labels[-1]
    return (tld.isalpha() and 2 <= len(tld) <= 63) or (tld.lower().startswith("xn--") and len(tld) > 4)


def nanp_valid(area: str, exchange: str, line: str) -> bool:
    """North American Numbering Plan: NPA and NXX are [2-9]XX and not N11 (service codes)."""
    if not (len(area) == 3 and len(exchange) == 3 and len(line) == 4 and (area + exchange + line).isdigit()):
        return False
    if area[0] in "01" or exchange[0] in "01":
        return False
    return area[1:] != "11" and exchange[1:] != "11"


# ============================================================================ patterns
_B = r"(?<![\w@.-])"          # left boundary: not inside a word, e-mail, decimal or hyphenated token
_E = r"(?![\w@]|[.-]\d)"       # right boundary
_SSN_FMT = re.compile(_B + r"(\d{3})-(\d{2})-(\d{4})" + _E)
_SSN_LOOSE = re.compile(_B + r"(\d{3})[ ]?(\d{2})[ ]?(\d{4})" + _E)
_SSN_KW = re.compile(r"\b(?:ssn|s\.s\.n\.?|ss#|soc(?:ial)?\.?\s*sec(?:urity)?(?:\s*(?:no\.?|number|#))?)"
                     r"(?:\s+(?:is|was|of|on\s+file))?\W*$", re.I)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
                    r"@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9]")
_PHONE_NANP = re.compile(_B + r"(?:\+?1[\s.-]?)?(?:\((\d{3})\)\s?|(\d{3})[\s.-])(\d{3})[\s.-](\d{4})" + _E)
_PHONE_BARE = re.compile(_B + r"(?:\+?1)?(\d{3})(\d{3})(\d{4})" + _E)
_PHONE_WORDS = r"phone|ph|tel(?:ephone)?|cell(?:ular)?|mobile|call|pager|beeper|contact"
_PHONE_KW = re.compile(r"\b(?:" + _PHONE_WORDS + r"|fax)\b", re.I)
_PHONE_ONLY_KW = re.compile(r"\b(?:" + _PHONE_WORDS + r")\b", re.I)
_FAX_KW = re.compile(r"\bfax\b", re.I)
_MRN = re.compile(
    r"\b(?P<kw>MRN|M\.R\.N\.?|MR\s*#|medical\s+record(?:\s+(?:number|no\.?|num|#))?|med\.?\s*rec(?:ord)?\.?(?:\s*(?:number|no\.?|#))?"
    r"|chart\s*(?:number|no\.?|#)|unit\s*(?:number|no\.?|#)|hospital\s*(?:number|no\.?|#)|patient\s*(?:id|number|no\.?|#))"
    r"(?![A-Za-z])"
    r"(?P<sep>-(?=\d)|\s*(?:is|of)?\s*[:#=]?\s*)"
    r"(?P<id>(?-i:[A-Z]{0,4})[-\s]?\d[\d-]{2,22}(?-i:[\dA-Z]))(?![\w-])",
    re.I)
_NPI = re.compile(r"\b(?:NPI|national\s+provider\s+(?:identifier|id|number))(?:\s*(?:number|no\.?|#))?\s*(?:is\s*)?[:#=]?\s*"
                  r"(?P<id>\d{10})(?!\d)", re.I)


def _preceded_by(text: str, start: int, pattern: re.Pattern, window: int) -> bool:
    return bool(pattern.search(text[max(0, start - window):start]))


# ============================================================================ validator
class StructuredIdValidator:
    """
    Finds verified structured identifiers in text. `find(text)` returns non-overlapping IdMatch objects in
    original-text offsets.

    mrn_checksum: None, "luhn", "mod11", or a callable(str_of_digits) -> bool for facility-specific MRN check digits.
    """

    def __init__(self, kinds: Sequence[str] = KINDS, mrn_min_digits: int = 5,
                 mrn_checksum: Optional[object] = None, context_window: int = 32, phone_region: str = "US",
                 use_phonenumbers: Optional[bool] = None):
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown validator kinds: {sorted(unknown)}; choose from {KINDS}")
        self.kinds = tuple(kinds)
        self.mrn_min_digits = int(mrn_min_digits)
        self.context_window = int(context_window)
        self.phone_region = phone_region
        if mrn_checksum in (None, "", "none"):
            self._mrn_check: Optional[Callable[[str], bool]] = None
        elif mrn_checksum == "luhn":
            self._mrn_check = luhn_valid
        elif mrn_checksum == "mod11":
            self._mrn_check = mod11_valid
        elif callable(mrn_checksum):
            self._mrn_check = mrn_checksum
        else:
            raise ValueError("mrn_checksum must be None, 'luhn', 'mod11' or a callable")
        self._pn = None
        if use_phonenumbers is not False:
            try:
                import phonenumbers
                self._pn = phonenumbers
            except ImportError:
                if use_phonenumbers:
                    raise
        self.phone_backend = "libphonenumber" if self._pn is not None else "nanp-rules"

    # ------------------------------------------------------------------ public
    def find(self, text: str) -> List[IdMatch]:
        if not text:
            return []
        found: List[IdMatch] = []
        # Order = priority when matches overlap: the most specific evidence first.
        if "npi" in self.kinds:
            found += self._npi(text)
        if "mrn" in self.kinds:
            found += self._mrn(text)
        if "ssn" in self.kinds:
            found += self._ssn(text)
        if "email" in self.kinds:
            found += self._email(text)
        if "phone" in self.kinds:
            found += self._phone(text)
        return _drop_overlaps(found)

    # ------------------------------------------------------------------ kinds
    def _ssn(self, text: str) -> List[IdMatch]:
        out = []
        for m in _SSN_FMT.finditer(text):
            if ssn_valid(*m.groups()):
                kw = _preceded_by(text, m.start(), _SSN_KW, self.context_window)
                out.append(IdMatch(m.start(), m.end(), Category.SSN, "ssn",
                                   "ddd-dd-dddd + SSA rules" + (" + keyword" if kw else "")))
        for m in _SSN_LOOSE.finditer(text):
            if "-" in m.group(0) or not _preceded_by(text, m.start(), _SSN_KW, self.context_window):
                continue
            if ssn_valid(*m.groups()):
                out.append(IdMatch(m.start(), m.end(), Category.SSN, "ssn", "SSN keyword + 9 digits + SSA rules"))
        return out

    def _email(self, text: str) -> List[IdMatch]:
        out = []
        for m in _EMAIL.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in ".-":
                e -= 1
            if email_valid(text[s:e]):
                out.append(IdMatch(s, e, Category.EMAIL, "email", "RFC 5321 structure"))
        return out

    def _phone_category(self, text: str, start: int) -> Category:
        """FAX if the nearest preceding contact keyword is 'fax', else PHONE."""
        ctx = text[max(0, start - self.context_window):start]
        fax = [m.start() for m in _FAX_KW.finditer(ctx)]
        tel = [m.start() for m in _PHONE_ONLY_KW.finditer(ctx)]
        return Category.FAX if fax and (not tel or fax[-1] > tel[-1]) else Category.PHONE

    def _phone(self, text: str) -> List[IdMatch]:
        out = []
        if self._pn is not None:
            pn = self._pn
            for m in pn.PhoneNumberMatcher(text, self.phone_region, leniency=pn.Leniency.VALID):
                raw = m.raw_string
                digits = re.sub(r"\D", "", raw)
                # libphonenumber also accepts bare digit runs; those need a phone keyword (could be any number)
                if digits == raw.lstrip("+") and not _preceded_by(text, m.start, _PHONE_KW, self.context_window):
                    continue
                out.append(IdMatch(m.start, m.end, self._phone_category(text, m.start), "phone",
                                   "libphonenumber VALID"))
        # Separator-formatted NANP numbers that pass the NANP rules are kept even when libphonenumber rejects them:
        # the metadata only knows ASSIGNED area codes, and de-identified / surrogate data (n2c2) and new area codes
        # still have to be redacted. Overlaps with libphonenumber matches are dropped in find().
        for m in _PHONE_NANP.finditer(text):
            area = m.group(1) or m.group(2)
            if nanp_valid(area, m.group(3), m.group(4)):
                out.append(IdMatch(m.start(), m.end(), self._phone_category(text, m.start()), "phone",
                                   "NANP format + rules"))
        if self._pn is None:
            for m in _PHONE_BARE.finditer(text):
                if _preceded_by(text, m.start(), _PHONE_KW, self.context_window) and nanp_valid(*m.groups()):
                    out.append(IdMatch(m.start(), m.end(), self._phone_category(text, m.start()), "phone",
                                       "phone keyword + NANP rules"))
        return out

    def _mrn(self, text: str) -> List[IdMatch]:
        out = []
        for m in _MRN.finditer(text):
            ident = m.group("id")
            digits = re.sub(r"\D", "", ident)
            if len(digits) < self.mrn_min_digits:
                continue
            if self._mrn_check is not None and not self._mrn_check(digits):
                continue
            # "MRN-123-45-6789" is one token: keep the keyword; "MRN: 0048291" -> the number only
            attached = m.group("sep") == "-"
            start = m.start("kw") if attached else m.start("id")
            out.append(IdMatch(start, m.end("id"), Category.MRN, "mrn",
                               "MRN keyword + id" + (" + checksum" if self._mrn_check else "")))
        return out

    def _npi(self, text: str) -> List[IdMatch]:
        return [IdMatch(m.start("id"), m.end("id"), Category.OTHER_ID, "npi", "NPI keyword + CMS Luhn")
                for m in _NPI.finditer(text) if npi_valid(m.group("id"))]


def _drop_overlaps(matches: Iterable[IdMatch]) -> List[IdMatch]:
    """Keeps the first-found (highest-priority) match where two overlap; returns them sorted by position."""
    kept: List[IdMatch] = []
    for m in matches:
        if all(m.end <= k.start or m.start >= k.end for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda x: (x.start, x.end))


def parse_kinds(value: Optional[str]) -> Tuple[str, ...]:
    """'ssn,email' -> ('ssn', 'email'); 'all' / '' / None -> every kind; 'none' -> ()."""
    v = (value or "all").strip().lower()
    if v in ("all", "1", "true", "yes", "on"):
        return KINDS
    if v in ("none", "0", "false", "no", "off"):
        return ()
    kinds = tuple(k.strip() for k in v.split(",") if k.strip())
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown validator kinds: {sorted(unknown)}; choose from {KINDS}")
    return kinds
