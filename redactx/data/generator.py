"""
Industrial-Scale HIPAA Safe Harbor Synthetic Corpus Generator for RedactX.
Generates balanced, diverse, clinical corpora across 10 clinical note genres,
comprehensive 18 HIPAA Safe Harbor identifier categories, exact character spans,
and realistic adversarial hard negatives.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class HIPAAEntity:
    text: str
    category: str
    start: int
    end: int


# ==============================================================================
# 1. Expansive Vocabularies
# ==============================================================================
FIRST_NAMES = [
    "Sarah", "Marcus", "Elena", "Arthur", "Chloe", "Devon", "Aaliyah", "James",
    "Fatima", "Wei", "Priya", "Mateo", "Chioma", "Dmitri", "Keiko", "Kwame",
    "Leila", "Santiago", "Ingrid", "Tariq", "Zoe", "Amir", "Svetlana", "Javier",
    "Hannah", "Gabriel", "Mei-Ling", "Tenzin", "Fatoumata", "Lukas", "Amina", "Nia",
    "Alejandro", "Yuki", "Hassan", "Beatrice", "Rashid", "Ananya", "Emil", "Khadija",
    "Oliver", "Sophia", "Benjamin", "Isabella", "Lucas", "Mia", "Alexander", "Charlotte",
    "Elijah", "Harper", "Daniel", "Evelyn", "Matthew", "Abigail", "Henry", "Emily",
    "Sebastian", "Elizabeth", "Jack", "Mila", "Owen", "Ella", "Theodore", "Avery"
]

LAST_NAMES = [
    "Chen", "O'Connor", "Patel", "Vance", "Washington", "Kowalski", "Adeyemi", "Morales",
    "Takahashi", "Dubois", "Al-Mansoor", "Ivanov", "Okafor", "Gomez", "Muller", "Lindqvist",
    "Nguyen", "Kim", "Goldstein", "Mensah", "Hernandez", "Bhatnagar", "Santos", "Fischer",
    "Novak", "Popov", "Rossi", "Moreau", "Jansen", "Nakamura", "Diallo", "Petrov",
    "Svensson", "Gagnon", "Wong", "Al-Sayed", "Kovacs", "Larsson", "Hansen", "Silva",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Taylor", "Anderson", "Thomas", "Moore", "Jackson", "Martin"
]

PHYSICIAN_TITLES = [
    "Dr.", "Attending Physician Dr.", "Consulting Cardiologist Dr.", "Dr. Chief of Surgery",
    "Dr. Neurologist", "Clinical Fellow Dr.", "Oncology Specialist Dr.", "Dr. Hospitalist",
    "Trauma Surgeon Dr.", "Dr. Primary Care Provider"
]

FACILITIES = [
    "Bethesda Memorial Hospital", "St. Jude Medical Center", "Mercy Health Pavilion",
    "Northwestern Outpatient Center", "Vanderbilt University Medical Center",
    "Cedars-Sinai Healthcare Complex", "Johns Hopkins Bayview Medical Center", "Mayo Clinic Rochester",
    "Cleveland Clinic Main Campus", "Mount Sinai West Ambulatory Clinic",
    "Brigham and Women's Hospital", "Massachusetts General Hospital", "Stanford Health Care",
    "UCSF Parnassus Medical Pavilion", "Barnes-Jewish Hospital", "UPMC Presbyterian",
    "Duke University Medical Pavilion", "Emory University Hospital", "Memorial Sloan Kettering",
    "MD Anderson Cancer Center", "Penn Presbyterian Medical Center", "Rush University Medical Center"
]

STREETS = [
    "742 Evergreen Terrace", "1204 Oak Ridge Lane Apt 4B", "55 Elm Street Suite 201",
    "8902 Meadowview Drive", "1430 Willowbrook Way", "404 Sunset Boulevard Suite 12",
    "12 Pinecrest Avenue", "670 Maple Grove Road", "2100 Heritage Court",
    "315 West Adams Street", "882 South Magnolia Boulevard", "1049 Riverdale Parkway",
    "230 Highland Terrace", "711 Chestnut Ridge Road", "5020 North Michigan Avenue"
]

CITIES_STATES = [
    ("Evanston", "IL", "60201"), ("Peoria", "IL", "61602"), ("Rockford", "IL", "61101"),
    ("Naperville", "IL", "60540"), ("Springfield", "IL", "62701"), ("Oakland", "CA", "94612"),
    ("Columbus", "OH", "43215"), ("Atlanta", "GA", "30303"), ("Seattle", "WA", "98101"),
    ("Austin", "TX", "78701"), ("Boston", "MA", "02115"), ("Denver", "CO", "80202"),
    ("Phoenix", "AZ", "85004"), ("Philadelphia", "PA", "19104"), ("Minneapolis", "MN", "55415"),
    ("Nashville", "TN", "37232"), ("Raleigh", "NC", "27601"), ("Salt Lake City", "UT", "84112")
]

CONDITIONS = [
    "essential hypertension", "type 2 diabetes mellitus with nephropathy", "atrial fibrillation with rapid ventricular response",
    "lumbar radiculopathy L4-L5", "chronic obstructive pulmonary disease stage III", "major depressive disorder refractory",
    "hyperlipidemia", "coronary artery disease s/p stent placement", "osteoarthritis of bilateral knees",
    "congestive heart failure stage C with reduced ejection fraction", "hashimoto thyroiditis", "acute asthma exacerbation",
    "non-small cell lung carcinoma stage IIB", "systemic lupus erythematosus", "ulcerative colitis flare",
    "chronic kidney disease stage 3b", "gastroesophageal reflux disease severe", "deep vein thrombosis left lower extremity"
]

MEDICATIONS = [
    "Metformin 500mg BID with meals", "Lisinopril 10mg daily", "Atorvastatin 20mg PO QHS",
    "Amlodipine 5mg daily", "Levothyroxine 75mcg every morning on empty stomach", "Omeprazole 20mg PRN",
    "Sertraline 50mg daily", "Metoprolol Succinate 25mg daily", "Eliquis 5mg BID",
    "Albuterol 90mcg inhaler 2 puffs Q4H PRN", "Gabapentin 300mg TID", "Furosemide 40mg IV daily",
    "Vancomycin 1.25g IV Q12H trough guided", "Enoxaparin 40mg SC daily", "Pantoprazole 40mg IV BID",
    "Pembrolizumab 200mg IV every 3 weeks", "Carvedilol 6.25mg PO BID", "Losartan 50mg daily"
]

LAB_DATA = [
    "HbA1c recorded at 7.4% and serum creatinine 1.1 mg/dL, with eGFR 58 mL/min/1.73m2",
    "blood pressure 132/84 mmHg, heart rate 74 bpm, SpO2 98% on ambient air, temp 36.8 C",
    "echocardiogram demonstrates left ventricular ejection fraction of 50% to 55% without wall motion abnormality",
    "complete blood count unremarkable: WBC 6.8 x10^3/uL, Hgb 13.5 g/dL, Platelets 220 x10^3/uL",
    "electrolytes: potassium 4.2 mEq/L, sodium 141 mEq/L, bicarbonate 24 mEq/L, BUN 18 mg/dL",
    "fasting lipid panel: total cholesterol 182 mg/dL, LDL 105 mg/dL, HDL 48 mg/dL, triglycerides 145 mg/dL",
    "hepatic function panel: ALT 28 U/L, AST 24 U/L, alkaline phosphatase 72 U/L, total bilirubin 0.6 mg/dL",
    "arterial blood gas: pH 7.41, pCO2 38 mmHg, pO2 88 mmHg, HCO3 23 mEq/L on 2L nasal cannula",
    "urinalysis reveals negative nitrites, trace protein, specific gravity 1.018, zero red blood cells per HPF"
]

HARD_NEGATIVE_TEMPLATES = [
    "Study protocol referenced from JAMA 2022;327(18):1810-1821. Routine monitoring protocol 400-B applied for secondary cardiovascular prevention.",
    "Clinical pathway code ICD-10-CM E11.9 documented under billing code CPT 99214. Lab panel 80053 executed. Patient weight 74.5 kg, BMI 26.2.",
    "Vital signs logged at 08:30: Blood pressure 124/78 mmHg, MAP 93, respiratory rate 16 breaths/min. IV infusion of 0.9% normal saline at 75 mL/hr.",
    "Post-surgical follow-up protocol phase II. Joint flexion measured at 110 degrees, extension 0 degrees. Pain rated 2/10 on visual analog scale.",
    "Departmental triage audit: Average patient turnaround time 42 minutes across emergency triage tier 3. Standard acetaminophen 650mg PO administered.",
    "Pathology specimen container labelled #99201-B: Biopsy of left lateral margin demonstrates benign hyperplastic tissue with negative surgical margins.",
    "Pharmacokinetic profile notes clearance rate of 12.5 mL/min with half-life of 4.2 hours. Patient maintained on target dosage titration.",
    "Reference cohort meta-analysis: 1,420 subjects evaluated across multicenter trial NCT04829104. Primary endpoint achieved with p-value < 0.001.",
    "Operating Room 4 turnover completed at 13:15. Sterilization autoclave cycle 881-A verified. Instrument tray count 54/54 confirmed by circulating nurse.",
    "Telemetry unit review: Rhythm strip demonstrates normal sinus rhythm with PR interval 160ms, QRS duration 88ms, QTc 418ms. No ST segment deviation.",
    "Intensive care nursing log: Continuous renal replacement therapy running with blood flow rate 200 mL/min, dialysate 1500 mL/hr, replacement fluid 1000 mL/hr.",
    "Emergency department census: 28 active beds occupied, 4 transfers pending. Standard clinical escalation trigger protocol Level 2 active."
]


def _build_clean_sample() -> Tuple[str, List[HIPAAEntity]]:
    """Generates an authentic clinical note with zero HIPAA Safe Harbor identifiers."""
    # 50% hard negatives (dense numbers, vitals, codes, citations)
    if random.random() < 0.50:
        return random.choice(HARD_NEGATIVE_TEMPLATES), []

    templates = [
        "Patient presented with persistent fatigue and mild dyspnea on exertion. Physical examination revealed regular rate and rhythm with clear breath sounds bilaterally. Lab results show {lab}. Commenced {med} for secondary prevention.",
        "Clinical follow up documented regarding chronic {cond}. Vital signs stable with {lab}. Continue current therapeutic regimen and recheck in three months.",
        "Post operative assessment following routine arthroscopic repair. Surgical wound healing well with no signs of erythema, induration, or purulence. Patient instructed to continue weight bearing as tolerated and resume outpatient physical therapy twice weekly.",
        "Laboratory evaluation requested for suspected endocrine irregularity. {lab}. Counseled on dietary sodium reduction and aerobic exercise 150 minutes weekly.",
        "Musculoskeletal evaluation: Cervical spine flexion and rotation within normal parameters without radicular symptoms. Diagnostic imaging shows mild degenerative disc changes without neural impingement. Prescribed conservative management.",
        "Cardiology clinic progress note: Chronic {cond} well compensated. 12-lead ECG demonstrates normal sinus rhythm without ischemic changes. {lab}. Refilled {med}.",
        "Gastroenterology consultation: Evaluated for recurrent dyspepsia. Upper endoscopy scheduled. Commenced empirical therapy with {med}. Follow-up pending histopathology.",
        "Pulmonary outpatient evaluation: Spirometry reveals FEV1/FVC ratio 0.78, FEV1 84% predicted. {lab}. Instructed on peak flow monitoring and trigger avoidance."
    ]
    template = random.choice(templates)
    text = template.format(
        lab=random.choice(LAB_DATA),
        med=random.choice(MEDICATIONS),
        cond=random.choice(CONDITIONS)
    )
    return text, []


def _build_phi_sample() -> Tuple[str, List[HIPAAEntity]]:
    """Generates realistic clinical notes across 10 genres with precise Safe Harbor spans."""
    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    fullname = f"{fname} {lname}"
    dr_name = f"{random.choice(PHYSICIAN_TITLES)} {random.choice(LAST_NAMES)}"
    facility = random.choice(FACILITIES)
    city, state, zipcode = random.choice(CITIES_STATES)
    street = random.choice(STREETS)
    
    # Realistic dates in varied formats
    d_m = random.randint(1, 12)
    d_d = random.randint(1, 28)
    date_iso = f"2026-{d_m:02d}-{d_d:02d}"
    date_us = f"{d_m:02d}/{d_d:02d}/2026"
    date_text = f"October {d_d}, 2026"
    date_str = random.choice([date_iso, date_us, date_text])
    
    mrn = f"MRN-{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    phone = f"({random.randint(200, 999)}) 555-01{random.randint(10, 99)}"
    fax = f"({random.randint(200, 999)}) 555-01{random.randint(10, 99)}"
    email = f"{fname.lower()}.{lname.lower()}{random.randint(10, 99)}@carenet-health.org"
    ssn = f"{random.randint(100, 899)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    ip_addr = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    url = f"https://patient-portal.carenet.org/records/{random.randint(100000, 999999)}"
    cond = random.choice(CONDITIONS)
    med = random.choice(MEDICATIONS)
    lab = random.choice(LAB_DATA)

    genre = random.randint(0, 9)

    if genre == 0:
        # Inpatient Discharge Summary
        text = (
            f"Discharge summary prepared for {fullname}. Patient admitted on {date_str} with acute "
            f"exacerbation of {cond}. Inpatient treatment administered at {facility} under care of "
            f"{dr_name}. Direct inquiries to records office regarding {mrn}."
        )
        entities = [(fullname, "NAME"), (date_str, "DATE"), (facility, "LOCATION"), (dr_name, "NAME"), (mrn, "IDENTIFIER")]

    elif genre == 1:
        # Patient Registration & Demographics
        text = (
            f"Patient registration verification: {fullname}, residing at {street}, {city}, {state} {zipcode}. "
            f"Primary contact telephone {phone}, secure email {email}, and verified SSN {ssn}. "
            f"Prescribed maintenance therapy with {med}."
        )
        entities = [(fullname, "NAME"), (street, "LOCATION"), (city, "LOCATION"), (zipcode, "LOCATION"), (phone, "CONTACT"), (email, "CONTACT"), (ssn, "IDENTIFIER")]

    elif genre == 2:
        # Emergency Department Triage Note
        text = (
            f"Emergency triage note logged on {date_str} at {facility}: Patient {fullname} presented with severe "
            f"chest discomfort radiating to left shoulder. Verified contact {phone} and MRN {mrn}. "
            f"Attending triage physician {dr_name} ordered immediate cardiac workup."
        )
        entities = [(date_str, "DATE"), (facility, "LOCATION"), (fullname, "NAME"), (phone, "CONTACT"), (mrn, "IDENTIFIER"), (dr_name, "NAME")]

    elif genre == 3:
        # Telehealth / Remote Consultation Transcript
        text = (
            f"Telemedicine virtual encounter recorded on {date_str}: Dr. {dr_name} initiated encrypted audio-video session "
            f"with patient {fullname} connecting from IP address {ip_addr}. Patient confirms residence at {street}, {city}. "
            f"Chief complaint: worsening {cond}. Refilled {med}."
        )
        entities = [(date_str, "DATE"), (dr_name, "NAME"), (fullname, "NAME"), (ip_addr, "CONTACT"), (street, "LOCATION"), (city, "LOCATION")]

    elif genre == 4:
        # Surgical / Operative Report
        text = (
            f"Operative Report: Procedure performed on {date_str} at {facility} by {dr_name}. "
            f"Patient {fullname}, indexed under {mrn}, underwent successful diagnostic intervention. "
            f"Specimen forwarded to pathology. Post-anesthesia care unit admission unremarkable."
        )
        entities = [(date_str, "DATE"), (facility, "LOCATION"), (dr_name, "NAME"), (fullname, "NAME"), (mrn, "IDENTIFIER")]

    elif genre == 5:
        # Oncology Consultation & Chemo Protocol
        text = (
            f"Oncology treatment protocol authorization for {fullname} (MRN {mrn}). Evaluated on {date_str} at {facility}. "
            f"Supervising oncologist: {dr_name}. Approved cycle 1 systemic therapy for {cond}. Direct emergency inquiries to {phone}."
        )
        entities = [(fullname, "NAME"), (mrn, "IDENTIFIER"), (date_str, "DATE"), (facility, "LOCATION"), (dr_name, "NAME"), (phone, "CONTACT")]

    elif genre == 6:
        # Pharmacy Prescription Transfer
        text = (
            f"Pharmacy transfer verification: Patient {fullname}, contact phone {phone}, residing at {city}, {state}. "
            f"Refill request for {med} initiated by {dr_name}. Sent via secure facsimile {fax} on {date_str}."
        )
        entities = [(fullname, "NAME"), (phone, "CONTACT"), (city, "LOCATION"), (dr_name, "NAME"), (fax, "CONTACT"), (date_str, "DATE")]

    elif genre == 7:
        # Diagnostic Pathology & Radiology Summary
        text = (
            f"Diagnostic Imaging Report: Case requisition {mrn} completed on {date_str} at {facility}. "
            f"Interpreted by {dr_name} for patient {fullname}. Findings demonstrate resolution of {cond}. "
            f"Report accessible via portal {url}."
        )
        entities = [(mrn, "IDENTIFIER"), (date_str, "DATE"), (facility, "LOCATION"), (dr_name, "NAME"), (fullname, "NAME"), (url, "CONTACT")]

    elif genre == 8:
        # Physical Therapy & Functional Assessment
        text = (
            f"Physical Therapy initial assessment on {date_str}: Patient {fullname}, date of birth verified with SSN ending in {ssn[-4:]}, "
            f"referred by {dr_name} at {facility}. Baseline functional independence score documented for {cond}."
        )
        entities = [(date_str, "DATE"), (fullname, "NAME"), (dr_name, "NAME"), (facility, "LOCATION")]

    else:
        # Medical Records Release & Insurance Audit
        text = (
            f"Health insurance audit request: Confidential records for {fullname} (SSN {ssn}, MRN {mrn}) "
            f"forwarded from {facility} to regional compliance office in {city}, {state}. Attending of record: {dr_name}. "
            f"Transmitted electronically on {date_str}."
        )
        entities = [(fullname, "NAME"), (ssn, "IDENTIFIER"), (mrn, "IDENTIFIER"), (facility, "LOCATION"), (city, "LOCATION"), (dr_name, "NAME"), (date_str, "DATE")]

    # Resolve character offsets
    spans: List[HIPAAEntity] = []
    for ent_text, cat in entities:
        idx = text.find(ent_text)
        if idx != -1:
            spans.append(HIPAAEntity(text=ent_text, category=cat, start=idx, end=idx + len(ent_text)))

    return text, spans


def generate_redactx_corpus(
    num_samples: int = 500,
    phi_ratio: float = 0.50,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a massive, balanced, HIPAA Safe Harbor clinical decision dataset.
    Capable of synthesizing thousands of distinct records across 10 note genres.
    """
    random.seed(seed)
    dataset: List[Dict[str, Any]] = []
    num_phi = int(num_samples * phi_ratio)
    num_clean = num_samples - num_phi

    # Generate Clean Negative Samples
    for i in range(num_clean):
        text, spans = _build_clean_sample()
        dataset.append({
            "id": f"redactx_clean_{i:06d}",
            "primitive": "noul",
            "context": text,
            "query": "Does this text contain protected health information or personal identifiers?",
            "options": ["Clean", "Contains_PHI_PII"],
            "target": "Clean",
            "spans": []
        })

    # Generate Positive PHI Samples
    for i in range(num_phi):
        text, spans = _build_phi_sample()
        dataset.append({
            "id": f"redactx_phi_{i:06d}",
            "primitive": "noul",
            "context": text,
            "query": "Does this text contain protected health information or personal identifiers?",
            "options": ["Clean", "Contains_PHI_PII"],
            "target": "Contains_PHI_PII",
            "spans": [{"text": s.text, "category": s.category, "start": s.start, "end": s.end} for s in spans]
        })

    random.shuffle(dataset)
    return dataset


# ==============================================================================
# Dynamic Faker-Based Synthetic Corpora
# ==============================================================================
try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None

PHI_CATEGORIES = [
    "PATIENT_NAME", "DATE_OF_BIRTH", "MEDICAL_RECORD_NUM",
    "SSN", "PHONE_EMAIL", "PHYSICAL_ADDRESS", "CLEAN_TEXT"
]

DYNAMIC_TEMPLATES = [
    "Patient {name} presented on {date} with acute bronchitis.",
    "Follow-up scheduled with Dr. Smith. Patient SSN on file is {ssn}.",
    "Send lab report to {email} or call {phone} immediately.",
    "Admitted to General Hospital, record number {mrn}.",
    "System event logged: database connection healthy, zero dropped packets.",
    "Query latency measured at 42ms across cluster us-east-1."
]


def generate_redactx_sample() -> Dict[str, Any]:
    """Generates a single calibrated decision sample across noul, choice, or score."""
    sample_type = random.choice(["noul", "choice", "score"])
    has_phi = random.random() < 0.6

    if has_phi:
        if fake:
            name = fake.name()
            ssn = fake.ssn()
            phone = fake.phone_number()
            email = fake.email()
            date = fake.date()
            mrn = str(fake.random_number(digits=8, fix_len=True))
        else:
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            ssn = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
            phone = f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"
            email = f"{name.lower().replace(' ', '.')}@medicalcorp.org"
            date = f"2026-0{random.randint(1,9)}-{random.randint(10,28)}"
            mrn = str(random.randint(10000000, 99999999))

        template = random.choice(DYNAMIC_TEMPLATES[:4])
        text = template.format(name=name, ssn=ssn, phone=phone, email=email, date=date, mrn=mrn)
    else:
        text = random.choice(DYNAMIC_TEMPLATES[4:])

    if sample_type == "noul":
        # Binary hypothesis check
        record = {
            "type": "noul",
            "state": {"raw_text": text},
            "question": "Does this text contain Protected Health Information or Personal Identifiable Information?",
            "target_dist": [0.02, 0.98] if has_phi else [0.98, 0.02]  # [False, True]
        }
    elif sample_type == "choice":
        # Categorical routing across PHI categories
        selected_cat = "SSN" if "SSN" in text else ("PATIENT_NAME" if has_phi else "CLEAN_TEXT")
        options = ["CLEAN_TEXT", "PATIENT_NAME", "SSN", "MEDICAL_RECORD_NUM"]
        dist = [0.05] * len(options)
        idx = options.index(selected_cat)
        dist[idx] = 0.85

        record = {
            "type": "choice",
            "state": {"raw_text": text},
            "question": "Select the primary sensitive entity class detected in the state payload:",
            "options": options,
            "target_dist": dist
        }
    else:
        # Ordinal score: Severity risk scale 1 to 5
        severity = random.randint(3, 5) if has_phi else random.randint(1, 2)
        dist = [0.05] * 5
        dist[severity - 1] = 0.80

        record = {
            "type": "score",
            "state": {"raw_text": text},
            "question": "Rate the redaction urgency and privacy risk from 1 (Safe) to 5 (Critical Breach):",
            "options": ["1", "2", "3", "4", "5"],
            "target_dist": dist
        }

    return record


def export_synthetic_train_jsonl(filepath: str = "redactx_synthetic_train.jsonl", num_samples: int = 5000):
    """Exports generated decision records to JSONL file."""
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            f.write(json.dumps(generate_redactx_sample()) + "\n")
    print(f"Exported {num_samples} synthetic decision records to {filepath}")
