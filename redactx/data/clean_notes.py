"""
Synthetic clinical notes for hard-negative / contrastive training.

Each note is built twice from the SAME skeleton:
  * PHI version   identifiers filled in (names, dates, ages, demographics, phones, MRNs, addresses, ...) with exact
                  character spans and labels
  * clean twin    every identifier slot replaced by natural non-identifying wording ("the patient", "an adult",
                  "on admission", "the number on file"), so the pair differs ONLY in the presence of PHI

plus stand-alone clean notes (clinical narrative, vitals, labs, medication lists, billing / NDC / lot codes) that
contain no identifiers at all. These are the hard negatives: dense clinical text that is not PHI.

Label policy (so the clean side never contradicts the positive side):
  * clean text never contains ages, gender words, ethnicity, occupation, clock times or calendar dates; relative
    times ("on hospital day 2", "for three days") are used instead
  * AGE spans cover the number only ("67" in "67-year-old"); DEMOGRAPHIC spans cover the descriptor
  * state abbreviations are not labelled (not a Safe Harbor identifier)

Disjointness: names, places, facilities, conditions, medications and sentence templates here are deliberately
different from redactx/data/generator.py (held-out set G) and the handwritten set H in validate_model.py, so training
on these notes does not leak those benchmarks.
"""

import random
from typing import Dict, List, Optional, Tuple

Span = Tuple[int, int, str]

FIRST = ["Rosalind", "Teodoro", "Imani", "Bartholomew", "Xiomara", "Desmond", "Lucinda", "Rafferty", "Esperanza",
         "Cornelius", "Annika", "Thaddeus", "Marisol", "Ezekiel", "Florence", "Ignatius", "Paloma", "Quentin",
         "Rosalyn", "Severin", "Tamsin", "Ulrich", "Valentina", "Winifred", "Yusuf", "Zainab", "Bronwyn", "Cyrus",
         "Delphine", "Evander", "Giselle", "Horatio", "Isadora", "Jasper", "Kalani", "Lorenzo", "Magnolia",
         "Nikolai", "Ophelia", "Percival", "Rhiannon", "Soren", "Thalia", "Ugo", "Vivienne", "Wendell", "Ximena",
         "Yolanda", "Zebulon", "Anouk", "Bashir", "Cosima", "Dashiell", "Eleni", "Faustino", "Greta", "Hollis",
         "Ines", "Joaquin", "Kirsten", "Lazarus", "Mireille", "Nnamdi", "Oksana", "Pilar", "Rocco", "Saoirse",
         "Tobias", "Urszula", "Vikram"]
LAST = ["Abernathy", "Balogun", "Castellanos", "Delacroix", "Eriksen", "Fairweather", "Grabowski", "Hollingsworth",
        "Iwasaki", "Jablonski", "Kaczmarek", "Lindgren", "Mbeki", "Nakashima", "Oyelaran", "Pemberton", "Quintero",
        "Rasmussen", "Szymanski", "Thibodeaux", "Underwood", "Valdivia", "Whitfield", "Yamamoto", "Zielinski",
        "Achterberg", "Bergstrom", "Cavanaugh", "Dimitriou", "Echeverria", "Fitzgerald", "Gutierrez-Paz", "Huang",
        "Ibekwe", "Jorgensen", "Kowalczyk", "Lachance", "Montgomery", "Nwachukwu", "Ostrowski", "Pacheco",
        "Rutherford", "Sorensen", "Tanaka", "Uchenna", "Vasquez", "Wojcik", "Yilmaz", "Zapata", "Okwuosa"]
CITIES = [("Tulsa", "OK", "74103"), ("Albany", "NY", "12208"), ("Fresno", "CA", "93721"), ("Omaha", "NE", "68102"),
          ("Spokane", "WA", "99201"), ("Duluth", "MN", "55802"), ("Savannah", "GA", "31401"),
          ("Lubbock", "TX", "79401"), ("Burlington", "VT", "05401"), ("Billings", "MT", "59101"),
          ("Tallahassee", "FL", "32301"), ("Lexington", "KY", "40507"), ("Provo", "UT", "84601"),
          ("Bangor", "ME", "04401"), ("Wichita", "KS", "67202"), ("Madison", "WI", "53703"),
          ("Knoxville", "TN", "37902"), ("Reno", "NV", "89501"), ("Eugene", "OR", "97401"),
          ("Fargo", "ND", "58102"), ("Mobile", "AL", "36602"), ("Toledo", "OH", "43604")]
STREET_NAMES = ["Larkspur", "Quarry Hill", "Bramblewood", "Harbor View", "Juniper", "Old Mill", "Saddleback",
                "Tamarack", "Foxglove", "Wexford", "Cobblestone", "Ridgecrest", "Kingfisher", "Lantern", "Silverbell"]
STREET_TYPES = ["Road", "Drive", "Court", "Lane", "Way", "Circle", "Avenue", "Place", "Trail"]
FACILITIES = ["Riverbend Regional Medical Center", "St. Agnes Community Hospital", "Lakeshore Family Health Clinic",
              "Prairie View Rehabilitation Institute", "Harborpoint Surgical Pavilion", "Maple Valley Cancer Institute",
              "Northgate Behavioral Health Center", "Summit Ridge Orthopedic Clinic", "Bayside Women's Hospital",
              "Pinecrest Veterans Medical Center", "Meadowlands Kidney Care", "Copper Canyon Urgent Care"]
EMAIL_DOMAINS = ["northmail.net", "outlook.com", "gmail.com", "yahoo.com", "icloud.com", "proton.me"]
RELATIVES = [("daughter", "her"), ("son", "his"), ("wife", "her"), ("husband", "his"), ("sister", "her"),
             ("brother", "his"), ("niece", "her"), ("granddaughter", "her"), ("partner", "their")]
SEX = ["woman", "man", "female", "male", "gentleman", "lady"]
ETHNICITY = ["Hispanic", "African American", "Caucasian", "Vietnamese", "Somali", "Puerto Rican", "Navajo",
             "Korean American", "Haitian", "Ashkenazi Jewish"]
RELIGION = ["Jehovah's Witness", "Catholic", "Muslim", "Orthodox Jewish", "Seventh-day Adventist"]
OCCUPATION = ["retired machinist", "school bus driver", "registered nurse", "long-haul trucker", "dairy farmer",
              "software engineer", "construction worker", "hair stylist", "former coal miner", "high school teacher"]
MARITAL = ["widowed", "divorced", "married", "single"]
LANGUAGE = ["Spanish-speaking", "Mandarin-speaking", "Haitian Creole-speaking", "Arabic-speaking"]

CONDITIONS = ["community-acquired pneumonia", "cellulitis of the right lower leg", "new-onset atrial flutter",
              "acute pancreatitis", "diabetic ketoacidosis", "symptomatic cholelithiasis", "pyelonephritis",
              "heart failure with preserved ejection fraction", "small bowel obstruction", "alcohol withdrawal",
              "acute kidney injury", "sickle cell pain crisis", "migraine without aura",
              "benign paroxysmal positional vertigo", "iron deficiency anemia", "obstructive sleep apnea",
              "gout flare of the left first MTP joint", "herpes zoster", "Clostridioides difficile colitis",
              "rheumatoid arthritis"]
MEDS = ["apixaban 5 mg twice daily", "ceftriaxone 1 g IV every 24 hours", "insulin glargine 18 units at bedtime",
        "prednisone 40 mg daily with a taper", "tamsulosin 0.4 mg nightly", "hydrochlorothiazide 25 mg daily",
        "ondansetron 4 mg every 8 hours as needed", "oxycodone 5 mg every 6 hours as needed",
        "colchicine 0.6 mg twice daily", "oral vancomycin 125 mg four times daily", "empagliflozin 10 mg daily",
        "spironolactone 25 mg daily", "levetiracetam 500 mg twice daily", "folic acid 1 mg daily",
        "bupropion XL 150 mg daily", "montelukast 10 mg nightly", "nitrofurantoin 100 mg twice daily",
        "ferrous sulfate 325 mg daily", "sumatriptan 50 mg as needed", "methotrexate 15 mg weekly"]
PROCEDURES = ["CT of the abdomen and pelvis with contrast", "a chest radiograph", "a transthoracic echocardiogram",
              "a renal ultrasound", "MRI of the lumbar spine", "a bedside paracentesis", "a lumbar puncture",
              "an upper endoscopy", "a colonoscopy", "a 12-lead ECG"]
FINDINGS = ["without acute abnormality", "showing a small right pleural effusion", "consistent with early appendicitis",
            "demonstrating mild hydronephrosis", "notable for diffuse colonic wall thickening",
            "with normal biventricular function", "unchanged from the prior study"]
PLANS = ["continue current antibiotics and trend inflammatory markers", "advance diet as tolerated",
         "physical therapy evaluation before discharge", "hold anticoagulation pending a repeat hemoglobin",
         "diabetes education and endocrinology follow-up", "repeat a basic metabolic panel in the morning",
         "wean supplemental oxygen as tolerated", "refer for an outpatient sleep study",
         "smoking cessation counseling was provided", "return precautions were reviewed"]
REL_TIMES = ["for the past three days", "since yesterday evening", "over the last two weeks", "earlier this week",
             "for several months", "intermittently for a year"]
ICD10 = ["J18.9", "L03.115", "I48.92", "K85.90", "E10.10", "K80.20", "N10", "I50.33", "K56.609", "F10.239", "N17.9",
         "D57.00", "G43.909", "H81.10", "D50.9", "G47.33", "M10.072", "B02.9", "A04.72", "M06.9"]
CPT = ["99223", "99232", "99238", "71046", "74177", "93306", "49083", "62270", "43235", "45378", "93000"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


class _Pair:
    """Builds the PHI text and its clean twin side by side."""

    def __init__(self):
        self.phi: List[str] = []
        self.clean: List[str] = []
        self.spans: List[Span] = []
        self._plen = 0

    def lit(self, s: str) -> "_Pair":
        self.phi.append(s)
        self.clean.append(s)
        self._plen += len(s)
        return self

    def diff(self, phi_s: str, clean_s: str) -> "_Pair":
        """Non-identifier text that must differ between the two versions (e.g. 'an 83-year-old' / 'an adult')."""
        self.phi.append(phi_s)
        self.clean.append(clean_s)
        self._plen += len(phi_s)
        return self

    def id(self, value: str, label: str, generic: str, span_of: Optional[str] = None) -> "_Pair":
        """Identifier slot. span_of = the part of `value` that is the identifier (e.g. '67' in '67-year-old')."""
        target = span_of or value
        off = value.index(target)
        self.spans.append((self._plen + off, self._plen + off + len(target), label))
        self.phi.append(value)
        self.clean.append(generic)
        self._plen += len(value)
        return self

    def extend(self, other: "_Pair") -> "_Pair":
        for s, e, lab in other.spans:
            self.spans.append((s + self._plen, e + self._plen, lab))
        self.phi.extend(other.phi)
        self.clean.extend(other.clean)
        self._plen += other._plen
        return self

    @property
    def phi_text(self) -> str:
        return "".join(self.phi)

    @property
    def clean_text(self) -> str:
        return "".join(self.clean)


# ---------------------------------------------------------------- identifier values
def _name(rng) -> str:
    f, l = rng.choice(FIRST), rng.choice(LAST)
    return f"{f} {rng.choice('ABCDEFGHJKLMNPRSTW')}. {l}" if rng.random() < 0.2 else f"{f} {l}"


def _clinician(rng) -> str:
    l = rng.choice(LAST)
    r = rng.random()
    if r < 0.6:
        return f"Dr. {l}"
    return f"{rng.choice(FIRST)} {l}, " + ("NP" if r < 0.8 else "PA-C")


def _date(rng) -> str:
    y, m, d = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"{m:02d}/{d:02d}/{y}", f"{m}/{d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}",
                       f"{d} {MONTHS[m - 1][:3]} {y}", f"{m}/{d}", f"{MONTHS[m - 1]} {d}"])


def _age(rng) -> int:
    return rng.randint(90, 101) if rng.random() < 0.12 else rng.randint(19, 89)


def _phone(rng) -> str:
    while True:
        a = f"{rng.randint(2, 9)}{rng.randint(0, 8)}{rng.randint(0, 9)}"
        x = f"{rng.randint(2, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}"
        if a[1:] != "11" and x[1:] != "11":
            break
    line = f"{rng.randint(0, 9999):04d}"
    return rng.choice([f"({a}) {x}-{line}", f"{a}-{x}-{line}", f"{a}.{x}.{line}"])


def _ssn(rng) -> str:
    while True:
        a, g, s = rng.randint(1, 899), rng.randint(1, 99), rng.randint(1, 9999)
        if a != 666:
            return f"{a:03d}-{g:02d}-{s:04d}"


def _street(rng) -> str:
    s = f"{rng.randint(12, 9899)} {rng.choice(STREET_NAMES)} {rng.choice(STREET_TYPES)}"
    return s + (f", Unit {rng.randint(1, 30)}{rng.choice('ABCD')}" if rng.random() < 0.25 else "")


def _mrn(rng) -> str:
    return rng.choice([f"{rng.randint(10**6, 10**7 - 1)}", f"{rng.randint(10**7, 10**8 - 1)}",
                       f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
                       f"{rng.choice('ABEHKMRT')}{rng.randint(10**6, 10**7 - 1)}"])


def _email(rng, name: Optional[str] = None) -> str:
    parts = (name or f"{rng.choice(FIRST)} {rng.choice(LAST)}").replace(".", "").split()
    user = rng.choice([f"{parts[0]}.{parts[-1]}", f"{parts[0][0]}{parts[-1]}", f"{parts[0]}{rng.randint(10, 99)}"])
    return f"{user.lower().replace(chr(39), '')}@{rng.choice(EMAIL_DOMAINS)}"


def _num(rng, lo, hi, nd=1) -> str:
    return f"{rng.uniform(lo, hi):.{nd}f}"


# ---------------------------------------------------------------- sentence families
def _article(age: int) -> str:
    return "an" if str(age).startswith("8") or age in (11, 18) else "a"


def _s_intro(rng) -> _Pair:
    p = _Pair()
    age = _age(rng)
    form = rng.choice(["{a}-year-old", "{a} y/o", "{a} yo", "{a} year old"])
    age_txt = form.format(a=age)
    if rng.random() < 0.6:
        p.id(_name(rng), "name", "The patient").lit(" is ")
    else:
        p.lit("Patient is ")
    p.diff(_article(age) + " ", "an ")
    p.id(age_txt, "age", "adult", span_of=str(age))
    if rng.random() < 0.5:
        eth = rng.choice(ETHNICITY)
        p.id(" " + eth, "race_ethnicity", "", span_of=eth)
    sex = rng.choice(SEX)
    p.id(" " + sex, "gender", "", span_of=sex)
    if rng.random() < 0.4:
        occ = rng.choice(OCCUPATION)
        p.id(f", a {occ},", "occupation", "", span_of=occ)
    p.lit(f" with a history of {rng.choice(CONDITIONS)} who presents with {rng.choice(CONDITIONS)} "
          f"{rng.choice(REL_TIMES)}. ")
    return p


def _s_social(rng) -> _Pair:
    p = _Pair().lit("Social history: ")
    marital = rng.choice(MARITAL)
    p.id(f"{marital}; ", "marital_status", "", span_of=marital)
    p.lit(rng.choice(["lives with ", "lives near ", "recently moved in with "]))
    rel, pron = rng.choice(RELATIVES)
    relative = _name(rng)
    p.lit(f"{pron} {rel}").id(" " + relative, "name", "", span_of=relative).lit(" in ").id(
        rng.choice(CITIES)[0], "city", "the area").lit(". ")
    if rng.random() < 0.5:
        p.lit("Primary language: ").id(rng.choice(LANGUAGE), "language", "documented in the chart").lit(". ")
    if rng.random() < 0.3:
        p.lit("Identifies as ").id(rng.choice(RELIGION), "religious_belief", "having stated preferences").lit(
            " and declines blood products. " if rng.random() < 0.5 else ". ")
    return p


def _s_admission(rng) -> _Pair:
    p = _Pair().lit(rng.choice(["Admitted to ", "Transferred to ", "Seen in the emergency department at "]))
    p.id(rng.choice(FACILITIES), "location", "this facility").lit(" on ").id(_date(rng), "date", "the day of admission")
    p.lit(f" for {rng.choice(CONDITIONS)}; ").id(_clinician(rng), "name", "the attending physician").lit(
        " was notified. ")
    return p


def _s_contact(rng, name: Optional[str] = None) -> _Pair:
    p = _Pair()
    r = rng.random()
    rel, pron = rng.choice(RELATIVES)
    if r < 0.4:
        p.lit(f"Emergency contact is {pron} {rel}, ").id(name or _name(rng), "name", "listed in the chart").lit(
            ", reachable at ").id(_phone(rng), "phone_number", "the number on file").lit(". ")
    elif r < 0.7:
        p.lit("Home address on file: ").id(_street(rng), "street_address", "verified")
        city, st, z = rng.choice(CITIES)
        p.lit(", ").id(city, "city", "").lit(f", {st} ").id(z, "zipcode", "").lit(". ")
        p.clean = ["Home address on file: verified. "]
    else:
        p.lit("Discharge paperwork sent to ").id(_email(rng, name), "email", "the patient portal").lit(
            "; pharmacy fax ").id(_phone(rng), "fax_number", "on record").lit(". ")
    return p


def _s_identifiers(rng) -> _Pair:
    p = _Pair()
    r = rng.random()
    if r < 0.45:
        p.lit(rng.choice(["MRN ", "MRN: ", "Medical record number ", "MR# "])).id(_mrn(rng), "medical_record_number",
                                                                               "on file").lit(". ")
    elif r < 0.7:
        p.lit("Insurance member ID ").id(f"{rng.choice(['XG', 'HMB', 'QW', 'ZK'])}{rng.randint(10**7, 10**9)}",
                                          "health_plan_beneficiary_number", "verified").lit(" was verified at registration. ")
        p.clean = ["Insurance coverage was verified at registration. "]
    elif r < 0.85:
        p.lit("SSN ").id(_ssn(rng), "ssn", "on file").lit(" confirmed for billing. ")
        p.clean = ["Billing information confirmed. "]
    else:
        p.lit("Date of birth ").id(_date(rng), "date_of_birth", "verified").lit(" confirmed with two identifiers. ")
        p.clean = ["Identity confirmed with two identifiers. "]
    return p


def _s_followup(rng) -> _Pair:
    p = _Pair().lit("Follow up with ").id(_clinician(rng), "name", "the primary care provider")
    if rng.random() < 0.6:
        p.lit(" on ").id(_date(rng), "date", "within one week")
    else:
        p.lit(" at ").id(rng.choice(FACILITIES), "location", "the clinic")
    return p.lit(". ")


# Clinical-only sentences (no identifiers): shared by twins and stand-alone hard negatives.
def _c_vitals(rng) -> str:
    sbp = rng.randint(96, 168)
    return (f"Vitals: BP {sbp}/{rng.randint(54, min(98, sbp - 25))}, HR {rng.randint(56, 118)}, "
            f"RR {rng.randint(12, 24)}, SpO2 {rng.randint(89, 100)}% on room air, temperature "
            f"{_num(rng, 97.1, 102.4)} F. ")


def _c_labs(rng) -> str:
    items = [f"WBC {_num(rng, 3.1, 19.8)} K/uL", f"hemoglobin {_num(rng, 7.2, 15.9)} g/dL",
             f"lactate {_num(rng, 0.6, 4.8)} mmol/L", f"BNP {rng.randint(40, 2400)} pg/mL",
             f"lipase {rng.randint(12, 980)} U/L", f"INR {_num(rng, 0.9, 3.4)}", f"glucose {rng.randint(68, 512)} mg/dL",
             f"creatinine {_num(rng, 0.5, 4.2, 2)} mg/dL", f"TSH {_num(rng, 0.2, 8.9, 2)} mIU/L",
             "troponin below the reporting limit"]
    rng.shuffle(items)
    return "Labs notable for " + ", ".join(items[:rng.randint(3, 5)]) + ". "


def _c_imaging(rng) -> str:
    proc = rng.choice(PROCEDURES)
    return f"{proc[0].upper()}{proc[1:]} was obtained, {rng.choice(FINDINGS)}. "


def _c_meds(rng) -> str:
    m = rng.sample(MEDS, k=rng.randint(2, 4))
    return rng.choice(["Started on ", "Continued ", "Medications at discharge: "]) + "; ".join(m) + ". "


def _c_plan(rng) -> str:
    return "Plan: " + "; ".join(rng.sample(PLANS, k=rng.randint(2, 3))) + ". "


def _c_codes(rng) -> str:
    return rng.choice([
        f"Encounter coded ICD-10 {rng.choice(ICD10)} and {rng.choice(ICD10)}, CPT {rng.choice(CPT)}. ",
        f"Dispensed NDC {rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}, "
        f"lot {rng.choice('ABCDEFGHKLMNP')}{rng.choice('ABCDEFGHKLMNP')}{rng.randint(10, 99)}{rng.choice('XYZ')}. ",
        f"Billing: CPT {rng.choice(CPT)} with modifier 25; diagnosis {rng.choice(ICD10)}. ",
    ])


def _c_course(rng) -> str:
    return rng.choice([
        f"On hospital day {rng.randint(2, 6)} the patient remained afebrile and tolerated a regular diet. ",
        f"Symptoms improved after {rng.randint(24, 72)} hours of therapy. ",
        "No acute events overnight; pain controlled on the current regimen. ",
        "Repeat imaging was deferred given clinical improvement. ",
    ])


_CLINICAL = [_c_vitals, _c_labs, _c_imaging, _c_meds, _c_plan, _c_codes, _c_course]
_HEADERS = ["HISTORY OF PRESENT ILLNESS: ", "Progress note. ", "Discharge summary. ", "ED provider note. ",
            "Consult note. ", "Interval history: ", ""]


def generate_clinical_pair(rng: random.Random, max_chars: int = 600) -> Dict:
    """One note with PHI + its clean twin: {"text", "spans": [(s, e, label)], "twin"}."""
    while True:
        note = _Pair().lit(rng.choice(_HEADERS))
        families = [_s_intro(rng)]
        k = rng.randint(1, 3)
        for f in rng.sample([_s_social, _s_admission, _s_contact, _s_identifiers, _s_followup], k=k):
            families.append(f(rng))
        clinical = [_Pair().lit(c(rng)) for c in rng.sample(_CLINICAL, k=rng.randint(1, 3))]
        order = families[:1] + rng.sample(families[1:] + clinical, k=len(families) - 1 + len(clinical))
        for part in order:
            if len(note.phi_text) + len(part.phi_text) > max_chars:
                break
            note.extend(part)
        text = note.phi_text.rstrip()
        spans = [(s, e, lab) for s, e, lab in note.spans if e <= len(text)]
        if spans:
            return {"text": text, "spans": spans, "twin": " ".join(note.clean_text.split())[:max_chars]}


def generate_clean_note(rng: random.Random, max_chars: int = 600) -> str:
    """A note with no identifiers: clinical narrative, vitals, labs, medications, codes."""
    parts = [rng.choice(_HEADERS), rng.choice([
        f"Patient presents with {rng.choice(CONDITIONS)} {rng.choice(REL_TIMES)}. ",
        f"Known history of {rng.choice(CONDITIONS)} and {rng.choice(CONDITIONS)}. ",
        f"Reason for visit: {rng.choice(CONDITIONS)}. ",
    ])]
    for c in rng.sample(_CLINICAL, k=rng.randint(2, 5)):
        s = c(rng)
        if len("".join(parts)) + len(s) > max_chars:
            break
        parts.append(s)
    return "".join(parts).strip()


def generate_clinical_pairs(n: int, seed: int = 2027, max_chars: int = 600) -> List[Dict]:
    rng = random.Random(seed)
    return [generate_clinical_pair(rng, max_chars) for _ in range(n)]


def generate_clean_notes(n: int, seed: int = 2028, max_chars: int = 600) -> List[str]:
    rng = random.Random(seed)
    return [generate_clean_note(rng, max_chars) for _ in range(n)]
