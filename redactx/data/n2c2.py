"""
n2c2 2014 (i2b2 2014 de-identification track) loader.

The corpus is distributed under a Data Use Agreement from the DBMI Data Portal (https://portal.dbmi.hms.harvard.edu)
and is NOT downloaded by this code. After approval, unpack it and point --n2c2-dir at the folder. Standard layout:

    training-PHI-Gold-Set1/*.xml   (521 records together with Set2 = the training split)
    training-PHI-Gold-Set2/*.xml
    testing-PHI-Gold-fixed/*.xml   (514 records = the test split)

Each file:
    <deIdi2b2>
      <TEXT><![CDATA[ ...note... ]]></TEXT>
      <TAGS>
        <DATE id="P0" start="16" end="26" text="2067-05-03" TYPE="DATE" comment="" />
        <NAME id="P1" start=".." end=".." text=".." TYPE="PATIENT" comment="" />
        ...

Every tag's offsets are VERIFIED against its `text` attribute; a tag whose offsets do not match is re-located by
searching for its text near the stated position, and counted in the load report (never silently trusted).

HIPAA mapping (45 CFR 164.514(b)(2), Safe Harbor): each n2c2 TYPE maps to a redactx Category and a flag saying
whether it is a Safe Harbor identifier. Notable choices:
  DOCTOR names, PROFESSION, HOSPITAL / ORGANIZATION / DEPARTMENT / ROOM, STATE, COUNTRY are n2c2 PHI but are not
  Safe Harbor identifiers of the patient; AGE is a Safe Harbor identifier only when > 89.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from redactx.production.hipaa import Category

SPLITS = {
    "train": ("training-PHI-Gold-Set1", "training-PHI-Gold-Set2"),
    "test": ("testing-PHI-Gold-fixed", "testing-PHI-Gold"),
}

# n2c2 TYPE -> (Category, Safe Harbor identifier?)
TYPE_MAP: Dict[str, Tuple[Category, bool]] = {
    "PATIENT": (Category.NAME, True),
    "DOCTOR": (Category.NAME, False),
    "USERNAME": (Category.OTHER_ID, True),
    "PROFESSION": (Category.DEMOGRAPHIC, False),
    "ROOM": (Category.LOCATION, False),
    "DEPARTMENT": (Category.ORGANIZATION, False),
    "HOSPITAL": (Category.ORGANIZATION, False),
    "ORGANIZATION": (Category.ORGANIZATION, False),
    "STREET": (Category.LOCATION, True),
    "CITY": (Category.LOCATION, True),
    "STATE": (Category.COARSE_LOCATION, False),
    "COUNTRY": (Category.COARSE_LOCATION, False),
    "ZIP": (Category.LOCATION, True),
    "LOCATION-OTHER": (Category.LOCATION, True),
    "AGE": (Category.AGE, False),            # True only when > 89, decided per span in hipaa_flag()
    "DATE": (Category.DATE, True),
    "PHONE": (Category.PHONE, True),
    "FAX": (Category.FAX, True),
    "EMAIL": (Category.EMAIL, True),
    "URL": (Category.URL, True),
    "IPADDR": (Category.IP_ADDRESS, True),
    "SSN": (Category.SSN, True),
    "MEDICALRECORD": (Category.MRN, True),
    "HEALTHPLAN": (Category.HEALTH_PLAN_ID, True),
    "ACCOUNT": (Category.ACCOUNT_NUMBER, True),
    "LICENSE": (Category.LICENSE_NUMBER, True),
    "VEHICLE": (Category.VEHICLE_ID, True),
    "DEVICE": (Category.DEVICE_ID, True),
    "BIOID": (Category.BIOMETRIC, True),
    "IDNUM": (Category.OTHER_ID, True),
}


def hipaa_flag(n2c2_type: str, text: str) -> bool:
    t = n2c2_type.upper()
    if t == "AGE":
        m = re.search(r"\d+", text)
        return bool(m) and int(m.group()) > 89
    return TYPE_MAP.get(t, (Category.UNKNOWN, False))[1]


def parse_record(path_or_xml: str, from_string: bool = False) -> Tuple[Dict, Dict[str, int]]:
    """
    Returns (doc, report). doc = {"id", "text", "spans": [{"start","end","text","tag","type","category","hipaa"}]}.
    report counts: tags, exact, relocated, dropped.
    """
    root = ET.fromstring(path_or_xml) if from_string else ET.parse(path_or_xml).getroot()
    text_el = root.find("TEXT")
    if text_el is None or text_el.text is None:
        raise ValueError(f"no <TEXT> in {'string' if from_string else path_or_xml}")
    text = text_el.text
    report = {"tags": 0, "exact": 0, "relocated": 0, "dropped": 0}
    spans: List[Dict] = []
    tags = root.find("TAGS")
    for el in (list(tags) if tags is not None else []):
        report["tags"] += 1
        try:
            s, e = int(el.get("start")), int(el.get("end"))
        except (TypeError, ValueError):
            report["dropped"] += 1
            continue
        value = el.get("text") or ""
        if text[s:e] == value:
            report["exact"] += 1
        else:
            # offsets off (e.g. CRLF normalisation by the XML parser): find the value near the stated position
            lo = max(0, s - 64)
            k = text.find(value, lo, min(len(text), e + 64 + len(value))) if value else -1
            if k < 0:
                report["dropped"] += 1
                continue
            s, e = k, k + len(value)
            report["relocated"] += 1
        typ = (el.get("TYPE") or el.tag).upper()
        cat = TYPE_MAP.get(typ, (Category.UNKNOWN, False))[0]
        spans.append({"start": s, "end": e, "text": text[s:e], "tag": el.tag.upper(), "type": typ,
                      "category": cat.value, "hipaa": hipaa_flag(typ, text[s:e])})
    spans.sort(key=lambda d: (d["start"], d["end"]))
    doc_id = "string" if from_string else os.path.splitext(os.path.basename(path_or_xml))[0]
    return {"id": doc_id, "text": text, "spans": spans}, report


def find_split_dirs(root: str, split: str) -> List[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    found = []
    for name in SPLITS[split]:
        hits = [p for p in glob.glob(os.path.join(root, "**", name), recursive=True) if os.path.isdir(p)]
        found.extend(hits)
    if split == "test":
        fixed = [p for p in found if p.endswith("testing-PHI-Gold-fixed")]
        found = fixed or found       # prefer the corrected release when both are present
    return sorted(set(found))


def resolve_n2c2_dir(root: Optional[str] = None) -> str:
    """
    Resolves the directory containing n2c2 2014 data.
    If an explicit root path is provided, it is returned directly.
    Otherwise, checks:
      1. Environment variable N2C2_DIR or REDACTX_N2C2_DIR
      2. Standard local locations under data/ (e.g. data/n2c2-NLP-Research-Data-Sets/...)
    """
    if root and str(root).strip():
        return str(root)
    env_dir = os.environ.get("N2C2_DIR") or os.environ.get("REDACTX_N2C2_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "data", "n2c2-NLP-Research-Data-Sets", "2014 - Deidentification & Heart Disease"),
        os.path.join(repo_root, "data", "n2c2-2014"),
        os.path.join(repo_root, "data"),
        os.path.abspath("data/n2c2-NLP-Research-Data-Sets/2014 - Deidentification & Heart Disease"),
        os.path.abspath("data/n2c2-2014"),
        os.path.abspath("data"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            for s in ("train", "test"):
                try:
                    if find_split_dirs(c, s):
                        return c
                except Exception:
                    pass
    return candidates[0]



PARTS = ("all", "calib", "eval")


def note_part(doc_id: str) -> str:
    """
    Deterministic assignment of a note to 'calib' or 'eval' based on sha1 hash of its ID.
    Because training used the train split, we can partition the test split into:
      'calib': 50% of test notes (used for threshold calibration)
      'eval':  50% of test notes (strictly held out for final benchmark)
    """
    import hashlib
    h = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16)
    return "calib" if h % 2 == 0 else "eval"


def load_n2c2(root: Optional[str] = None, split: str = "test", limit: Optional[int] = None,
              part: str = "all") -> Tuple[List[Dict], Dict]:
    """Loads one split. Auto-discovers root if omitted. Raises FileNotFoundError with instructions if not found."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    root = resolve_n2c2_dir(root)
    dirs = find_split_dirs(root, split)
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.xml")))
    if not files:
        raise FileNotFoundError(
            f"No n2c2 2014 '{split}' XML files under {root!r}. Expected folders {SPLITS[split]}. The corpus requires a "
            "Data Use Agreement from https://portal.dbmi.hms.harvard.edu (2014 De-identification track).")
    if part != "all":
        files = [f for f in files if note_part(os.path.splitext(os.path.basename(f))[0]) == part]
    if limit:
        files = files[:limit]
    docs, total = [], {"files": 0, "tags": 0, "exact": 0, "relocated": 0, "dropped": 0}
    for f in files:
        doc, rep = parse_record(f)
        docs.append(doc)
        total["files"] += 1
        for k in ("tags", "exact", "relocated", "dropped"):
            total[k] += rep[k]
    total["dirs"] = dirs
    return docs, total


def windows_with_spans(doc: Dict, max_chars: int = 600, overlap: int = 0) -> List[Dict]:
    """
    Cuts a note into windows (production chunker boundaries) and re-bases its spans. Spans crossing a window edge
    are clipped. Returns [{"text", "spans": [(s, e, category)], "doc_id", "offset"}].
    """
    from redactx.production.chunking import chunk_text
    out = []
    for w in chunk_text(doc["text"], max_chars, overlap):
        spans = []
        for sp in doc["spans"]:
            s, e = max(sp["start"], w.start), min(sp["end"], w.end)
            if e > s:
                spans.append((s - w.start, e - w.start, sp["category"]))
        out.append({"text": w.slice(doc["text"]), "spans": spans, "doc_id": doc["id"], "offset": w.start})
    return out
