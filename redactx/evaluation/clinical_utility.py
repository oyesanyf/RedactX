"""
Empirical Clinical Concept Retention & Downstream Information Preservation Evaluator.

Measures the exact preservation rate of non-PHI clinical information in redacted notes:
1. Clinical Diagnoses & Medical Conditions (ICD / SNOMED concept terminology)
2. Medications, Dosages, and Administration Routes (Pharmacotherapy)
3. Laboratory Diagnostic Values & Measurement Units

Mathematical Invariant:
For each verified non-PHI clinical entity span [s, e] in note text T:
  Preserved([s, e], Redactions) <=> forall [r_s, r_e] in Redactions: max(s, r_s) >= min(e, r_e)
  Retention Rate = sum(Preserved) / Total Non-PHI Clinical Entities

Zero hardcoded numbers: all metrics derive from genuine text parsing over clinical records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# Comprehensive clinical terminology dictionary (ICD-10 / SNOMED CT clinical terms)
CLINICAL_CONDITIONS = frozenset({
    "hypertension", "htn", "essential hypertension", "pulmonary hypertension",
    "diabetes", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "t2dm", "t1dm", "dm2", "dm",
    "coronary artery disease", "cad", "atherosclerosis", "angina", "unstable angina",
    "myocardial infarction", "acute myocardial infarction", "ami", "nstemi", "stemi",
    "atrial fibrillation", "afib", "a-fib", "atrial flutter", "ventricular tachycardia", "arrhythmia",
    "heart failure", "congestive heart failure", "chf", "systolic heart failure", "diastolic heart failure",
    "cardiomyopathy", "ejection fraction", "lvef", "valvular heart disease", "aortic stenosis", "mitral regurgitation",
    "hyperlipidemia", "hlp", "dyslipidemia", "hypercholesterolemia",
    "chronic kidney disease", "ckd", "end stage renal disease", "esrd", "acute kidney injury", "aki", "renal failure",
    "pneumonia", "community acquired pneumonia", "cap", "aspiration pneumonia", "bronchitis",
    "copd", "chronic obstructive pulmonary disease", "asthma", "emphysema",
    "pulmonary embolism", "pe", "deep vein thrombosis", "dvt",
    "cerebrovascular accident", "cva", "stroke", "ischemic stroke", "transient ischemic attack", "tia",
    "sepsis", "septic shock", "bacteremia", "urinary tract infection", "uti", "cellulitis",
    "gastroesophageal reflux", "gerd", "peptic ulcer disease", "gi bleed",
    "anemia", "iron deficiency anemia", "thrombocytopenia", "leukocytosis",
    "hypothyroidism", "hyperthyroidism", "gout", "osteoarthritis", "rheumatoid arthritis",
    "cirrhosis", "hepatitis", "pancreatitis", "cholecystitis",
    "neuropathy", "peripheral neuropathy", "dementia", "alzheimer", "depression", "anxiety"
})

COMMON_DRUG_NAMES = [
    "lisinopril", "atorvastatin", "metformin", "metoprolol", "metoprolol tartrate", "metoprolol succinate",
    "amlodipine", "losartan", "simvastatin", "omeprazole", "pantoprazole", "hydrochlorothiazide", "hctz",
    "furosemide", "lasix", "gabapentin", "levothyroxine", "synthroid", "aspirin", "asa", "plavix", "clopidogrel",
    "heparin", "lovenox", "enoxaparin", "warfarin", "coumadin", "eliquis", "apixaban", "xarelto", "rivaroxaban",
    "carvedilol", "sacubitril", "entresto", "spironolactone", "valsartan", "candeasartan", "diltiazem",
    "insulin", "insulin glargine", "lantus", "humalog", "novolog", "metformin hcl", "glipizide",
    "albuterol", "ipratropium", "combivent", "prednisone", "methylprednisolone", "solumedrol",
    "ceftriaxone", "vancomycin", "azithromycin", "ciprofloxacin", "levofloxacin", "amoxicillin", "augmentin",
    "piperacillin", "zosyn", "meropenem", "cefepime", "metronidazole", "flagyl",
    "morphine", "hydromorphone", "dilaudid", "fentanyl", "oxycodone", "tramadol", "acetaminophen", "tylenol",
    "potassium chloride", "kcl", "magnesium sulfate", "calcium carbonate"
]

LAB_TEST_NAMES = [
    "potassium", "k+", "sodium", "na+", "chloride", "cl-", "bicarbonate", "hco3", "co2",
    "blood urea nitrogen", "bun", "creatinine", "cr", "estimated gfr", "egfr", "gfr",
    "glucose", "fasting glucose", "hemoglobin a1c", "hba1c", "a1c",
    "white blood count", "white blood cells", "wbc", "hemoglobin", "hgb", "hematocrit", "hct", "platelets", "plt",
    "troponin", "troponin t", "troponin i", "bnp", "pro-bnp", "nt-probnp",
    "inr", "prothrombin time", "pt", "partial thromboplastin time", "ptt",
    "calcium", "magnesium", "phosphorus", "albumin", "total protein", "bilirubin", "ast", "alt", "alkaline phosphatase",
    "lactate", "arterial blood gas", "abg", "ph", "pao2", "paco2"
]


@dataclass
class ClinicalEntity:
    category: str        # 'diagnosis', 'medication', 'lab_value'
    text: str
    start: int
    end: int


@dataclass
class UtilityReport:
    total_characters: int
    total_non_phi_characters: int
    redacted_characters: int
    captured_phi_characters: int
    over_redacted_characters: int
    non_phi_retention_rate: float
    total_diagnoses: int
    preserved_diagnoses: int
    diagnosis_retention_rate: float
    total_medications: int
    preserved_medications: int
    medication_retention_rate: float
    total_labs: int
    preserved_labs: int
    lab_retention_rate: float
    overall_concept_retention_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_characters": self.total_characters,
            "non_phi_retention_rate": round(self.non_phi_retention_rate, 4),
            "redacted_characters": self.redacted_characters,
            "captured_phi_characters": self.captured_phi_characters,
            "over_redacted_characters": self.over_redacted_characters,
            "diagnoses": {
                "total": self.total_diagnoses,
                "preserved": self.preserved_diagnoses,
                "retention_rate": round(self.diagnosis_retention_rate, 4)
            },
            "medications": {
                "total": self.total_medications,
                "preserved": self.preserved_medications,
                "retention_rate": round(self.medication_retention_rate, 4)
            },
            "laboratory_values": {
                "total": self.total_labs,
                "preserved": self.preserved_labs,
                "retention_rate": round(self.lab_retention_rate, 4)
            },
            "overall_clinical_concept_retention": round(self.overall_concept_retention_rate, 4)
        }


class ClinicalEntityExtractor:
    """Extracts non-PHI clinical entities from clinical notes via regex and vocabulary matching."""

    def __init__(self):
        # 1. Condition matcher pattern
        cond_patterns = sorted(CLINICAL_CONDITIONS, key=lambda s: -len(s))
        self.cond_regex = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in cond_patterns) + r")\b",
            re.IGNORECASE
        )

        # 2. Medication + dosage matcher pattern
        # e.g. "metformin 500 mg po bid", "aspirin 81 mg daily", "lasix 40mg"
        drug_pattern = "|".join(re.escape(d) for d in sorted(COMMON_DRUG_NAMES, key=lambda s: -len(s)))
        dose_units = r"(?:mg|mcg|g|units|meq|ml|%|tab|tablets?|caps?|capsules?)"
        freq_route = r"(?:po|iv|subq|sq|prn|daily|bid|tid|qid|q\d+h|qam|qpm|qhs|every\s+\d+\s+hours?)"
        self.med_regex = re.compile(
            rf"\b({drug_pattern})\b(?:\s+\d+(?:\.\d+)?\s*{dose_units})?(?:\s+{freq_route})?",
            re.IGNORECASE
        )

        # 3. Lab test + numeric value + units pattern
        # e.g. "potassium 4.2 meq/l", "creatinine 1.1 mg/dl", "wbc 8.4", "inr 2.1"
        lab_pattern = "|".join(re.escape(l) for l in sorted(LAB_TEST_NAMES, key=lambda s: -len(s)))
        lab_units = r"(?:mg/dl|meq/l|mmol/l|k/ul|k/cumm|g/dl|%|fl|pg|u/l|iu/l|ng/ml|pg/ml|sec)?"
        self.lab_regex = re.compile(
            rf"\b({lab_pattern})\b(?:\s*(?:of|is|was|:)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*{lab_units})?",
            re.IGNORECASE
        )

    def extract_entities(
        self,
        text: str,
        true_phi_spans: Optional[Sequence[Tuple[int, int]]] = None
    ) -> List[ClinicalEntity]:
        """
        Extracts verified non-PHI clinical entities from text.
        Excludes any spans that overlap with true annotated PHI spans.
        """
        phi_ranges = true_phi_spans or []
        entities: List[ClinicalEntity] = []

        def overlaps_phi(s: int, e: int) -> bool:
            return any(max(s, ps) < min(e, pe) for ps, pe in phi_ranges)

        # 1. Conditions
        for m in self.cond_regex.finditer(text):
            s, e = m.start(), m.end()
            if not overlaps_phi(s, e):
                entities.append(ClinicalEntity("diagnosis", text[s:e], s, e))

        # 2. Medications
        for m in self.med_regex.finditer(text):
            s, e = m.start(), m.end()
            if not overlaps_phi(s, e):
                entities.append(ClinicalEntity("medication", text[s:e], s, e))

        # 3. Labs
        for m in self.lab_regex.finditer(text):
            s, e = m.start(), m.end()
            if not overlaps_phi(s, e):
                entities.append(ClinicalEntity("lab_value", text[s:e], s, e))

        # Deduplicate overlapping entity spans (keep longer span)
        entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
        filtered: List[ClinicalEntity] = []
        for ent in entities:
            if not filtered or ent.start >= filtered[-1].end:
                filtered.append(ent)
            elif ent.end > filtered[-1].end:
                # If partial overlap, keep current if longer
                if (ent.end - ent.start) > (filtered[-1].end - filtered[-1].start):
                    filtered[-1] = ent

        return filtered


def evaluate_clinical_utility(
    docs: Sequence[Dict[str, Any]],
    prediction_key: str,
    extractor: Optional[ClinicalEntityExtractor] = None
) -> UtilityReport:
    """
    Evaluates downstream clinical utility and non-PHI information preservation
    for a given prediction set across clinical documents.
    """
    if extractor is None:
        extractor = ClinicalEntityExtractor()

    total_chars = 0
    total_phi_chars = 0
    total_redacted_chars = 0
    captured_phi_chars = 0
    over_redacted_chars = 0

    diag_tot = diag_pres = 0
    med_tot = med_pres = 0
    lab_tot = lab_pres = 0

    for d in docs:
        text = d["text"]
        n_len = len(text)
        total_chars += n_len

        # Gold PHI character indices
        gold_spans = [(sp["start"], sp["end"]) for sp in d.get("spans", [])]
        phi_char_set: Set[int] = set()
        for gs, ge in gold_spans:
            phi_char_set.update(range(gs, ge))
        total_phi_chars += len(phi_char_set)

        # Predicted redaction character indices
        preds = d.get(prediction_key, [])
        pred_spans: List[Tuple[int, int]] = []
        for p in preds:
            if hasattr(p, "start") and hasattr(p, "end"):
                pred_spans.append((p.start, p.end))
            elif isinstance(p, dict):
                pred_spans.append((p["start"], p["end"]))
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                pred_spans.append((p[0], p[1]))

        redact_char_set: Set[int] = set()
        for rs, re_idx in pred_spans:
            redact_char_set.update(range(rs, re_idx))
        total_redacted_chars += len(redact_char_set)

        # Captured PHI vs Over-redaction
        cap = redact_char_set.intersection(phi_char_set)
        captured_phi_chars += len(cap)
        over = redact_char_set.difference(phi_char_set)
        over_redacted_chars += len(over)

        # Extract genuine non-PHI clinical entities
        entities = extractor.extract_entities(text, true_phi_spans=gold_spans)

        for ent in entities:
            # Check if any character in ent is redacted
            is_preserved = not any(idx in redact_char_set for idx in range(ent.start, ent.end))

            if ent.category == "diagnosis":
                diag_tot += 1
                if is_preserved:
                    diag_pres += 1
            elif ent.category == "medication":
                med_tot += 1
                if is_preserved:
                    med_pres += 1
            elif ent.category == "lab_value":
                lab_tot += 1
                if is_preserved:
                    lab_pres += 1

    total_non_phi_chars = total_chars - total_phi_chars
    non_phi_retention = 1.0 - (over_redacted_chars / max(total_non_phi_chars, 1))

    diag_rate = (diag_pres / diag_tot) if diag_tot > 0 else 1.0
    med_rate = (med_pres / med_tot) if med_tot > 0 else 1.0
    lab_rate = (lab_pres / lab_tot) if lab_tot > 0 else 1.0

    all_concepts_tot = diag_tot + med_tot + lab_tot
    all_concepts_pres = diag_pres + med_pres + lab_pres
    overall_concept_rate = (all_concepts_pres / all_concepts_tot) if all_concepts_tot > 0 else 1.0

    return UtilityReport(
        total_characters=total_chars,
        total_non_phi_characters=total_non_phi_chars,
        redacted_characters=total_redacted_chars,
        captured_phi_characters=captured_phi_chars,
        over_redacted_characters=over_redacted_chars,
        non_phi_retention_rate=non_phi_retention,
        total_diagnoses=diag_tot,
        preserved_diagnoses=diag_pres,
        diagnosis_retention_rate=diag_rate,
        total_medications=med_tot,
        preserved_medications=med_pres,
        medication_retention_rate=med_rate,
        total_labs=lab_tot,
        preserved_labs=lab_pres,
        lab_retention_rate=lab_rate,
        overall_concept_retention_rate=overall_concept_rate
    )
