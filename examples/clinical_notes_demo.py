"""
Clinical Note De-identification & PHI Detection Demo for RedactX.
Evaluates clinical notes against HIPAA Safe Harbor identifiers:
- Patient & Physician Names ([NAME])
- Dates of Service, Admission, Birth ([DATE])
- Medical Record Numbers, SSNs, Account Numbers ([MRN] / [IDENTIFIER])
- Hospitals, Clinics, Geographic Addresses ([LOCATION])
- Phone, Fax, Email, Contact details ([PHONE] / [CONTACT])

Usage:
  # 1. Test a single clinical note string:
  python examples/clinical_notes_demo.py "Discharge summary for Robert Chen admitted on 10/12/2026 to Mercy Health by Dr. Patel."

  # 2. Run built-in clinical demos across 5 medical specialties:
  python examples/clinical_notes_demo.py --demo

  # 3. Interactive console mode (paste any note):
  python examples/clinical_notes_demo.py --interactive
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any

import torch
from redactx.models.openjev import OpenJevVaultGemmaEngine
from redactx.production.masking import mask_text


DEMO_CLINICAL_NOTES = [
    {
        "specialty": "Inpatient Cardiology Discharge Summary",
        "text": (
            "DISCHARGE SUMMARY\n"
            "PATIENT NAME: Marcus Kowalski\n"
            "MRN: 948-21-4820 | ADMIT DATE: October 14, 2026 | DISCHARGE DATE: October 18, 2026\n"
            "ATTENDING PHYSICIAN: Dr. Sarah Vance, MD (Cardiology)\n"
            "FACILITY: Bethesda Memorial Hospital, Inpatient Ward 4B\n"
            "PRIMARY DIAGNOSIS: Acute congestive heart failure exacerbation with reduced ejection fraction (EF 35%).\n"
            "HOSPITAL COURSE: 62-year-old male admitted with 3-day history of worsening bilateral lower extremity "
            "edema and orthopnea. Diuresed aggressively with IV Furosemide 40mg BID. Weight decreased by 4.2 kg. "
            "Telemetry confirmed sinus rhythm without ischemic changes.\n"
            "DISCHARGE MEDICATIONS: Sacubitril/Valsartan 24/26mg PO BID, Carvedilol 12.5mg PO BID, Spironolactone 25mg daily.\n"
            "FOLLOW-UP: Return to Cardiology Clinic in 2 weeks. For emergency questions, contact clinic nurse at (555) 019-2831."
        ),
    },
    {
        "specialty": "Emergency Department (ED) Triage Note",
        "text": (
            "EMERGENCY DEPARTMENT TRIAGE NOTE\n"
            "TIME: 2026-10-16 03:42 EST | TRIAGE NURSE: Elena Chen, RN\n"
            "PATIENT: Tariq Al-Mansoor | DOB: 05/21/1988 | CONTACT: (555) 012-9844\n"
            "PRESENTING COMPLAINT: Sudden onset pleuritic chest pain radiating to left scapula.\n"
            "VITALS: BP 148/92 mmHg, HR 104 bpm, RR 20 breaths/min, SpO2 97% on ambient air, Temp 37.1 C.\n"
            "INITIAL EVALUATION: 12-lead ECG shows sinus tachycardia without acute ST elevation. Troponin I < 0.01 ng/mL.\n"
            "ASSESSMENT: Attending physician Dr. James Dubois ordered CT pulmonary angiogram to rule out pulmonary embolism."
        ),
    },
    {
        "specialty": "Oncology Outpatient Progress Note",
        "text": (
            "OUTPATIENT ONCOLOGY CLINIC NOTE\n"
            "PATIENT: Mei-Ling Takahashi (MRN: 382-90-1120)\n"
            "DATE OF SERVICE: 10/19/2026 | LOCATION: Cedars-Sinai Healthcare Complex, Suite 302\n"
            "PROVIDER: Consulting Oncologist Dr. Dmitri Ivanov\n"
            "CLINICAL IMPRESSION: Metastatic non-small cell lung carcinoma, EGFR exon 19 deletion.\n"
            "TREATMENT PLAN: Patient tolerating Osimertinib 80mg daily with grade 1 xerosis. ECOG performance status 1.\n"
            "Laboratory panel unremarkable: CBC with ANC 2,400/uL, serum creatinine 0.9 mg/dL. Re-staging chest CT in 8 weeks."
        ),
    },
    {
        "specialty": "Hard Negative Control (Pure De-identified Medical Literature / Protocol)",
        "text": (
            "CLINICAL GUIDELINE PROTOCOL - ACC/AHA 2026 GUIDELINES\n"
            "Protocol Reference: Guideline-Directed Medical Therapy (GDMT) for HFrEF.\n"
            "Recommendation Class 1A: Quadruple therapy consisting of ARNI, beta-blocker, MRA, and SGLT2 inhibitor.\n"
            "Dosing titration protocol: Initiate Empagliflozin 10mg once daily regardless of glycemic status if eGFR > 20 mL/min/1.73m2.\n"
            "Monitoring parameters: Check basic metabolic panel for serum potassium and creatinine within 1 to 2 weeks of initiation.\n"
            "ICD-10-CM coding reference: I50.22 (Chronic systolic heart failure)."
        ),
    },
]


def evaluate_clinical_note(engine: OpenJevVaultGemmaEngine, text: str, title: str = "Clinical Note"):
    print("\n" + "=" * 78)
    print(f" EVALUATING: {title}")
    print("=" * 78)
    print("--- ORIGINAL CLINICAL NOTE ---")
    print(text.strip())
    print("-" * 78)

    t0 = time.perf_counter()
    decision = engine.evaluate_text(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    phi_prob = decision.phi_probability
    is_phi = phi_prob >= engine.doc_threshold
    verdict_str = "SENSITIVE PHI DETECTED" if is_phi else "SAFE / NO PHI (CLEAN)"

    print("\n[DECISION ANALYSIS]")
    print(f"  * Verdict:             {verdict_str}")
    print(f"  * Calibrated PHI Prob: {phi_prob:.4f} (Threshold: {engine.doc_threshold:.4f})")
    print(f"  * Model Confidence:    {decision.confidence * 100:.1f}%")
    print(f"  * Latency:             {latency_ms:.2f} ms")
    print(f"  * Gateway Action:      {'[BLOCKED / DE-IDENTIFY]' if is_phi else '[PASSED / SAFE]'}")

    spans = decision.spans
    if spans:
        print(f"\n[IDENTIFIED HIPAA SAFE HARBOR SPANS ({len(spans)})]:")
        for idx, s in enumerate(spans, 1):
            conf_str = f"{getattr(s, 'confidence', 1.0):.2f}"
            print(f"   {idx}. [{s.category:<12}] \"{s.text}\" (chars {s.start}:{s.end}, conf {conf_str})")

    # Generate Safe Harbor Redacted Note
    redacted_note = mask_text(text, spans, policy="safe_harbor")
    print("\n--- DE-IDENTIFIED CLINICAL NOTE (HIPAA SAFE HARBOR) ---")
    print(redacted_note.strip())
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Test Any Clinical Note with RedactX")
    parser.add_argument("text", nargs="?", type=str, default=None, help="Clinical note text to scan")
    parser.add_argument("--file", type=str, default=None, help="Path to file containing clinical note")
    parser.add_argument("--interactive", action="store_true", default=False, help="Interactive console mode")
    parser.add_argument("--demo", action="store_true", default=False, help="Run demonstration across clinical notes")
    parser.add_argument("--model-dir", type=str, default="./models/RedactX-v3", help="Model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if not args.text and not args.file and not args.interactive and not args.demo:
        parser.print_help()
        print("\n[TIP] Quick start examples:")
        print('  python examples/clinical_notes_demo.py "Discharge note for patient Marcus Chen seen on 10/12/2026."')
        print("  python examples/clinical_notes_demo.py --demo")
        print("  python examples/clinical_notes_demo.py --interactive")
        sys.exit(0)

    print(f"Initializing RedactX Engine from '{args.model_dir}' on {args.device}...")
    t_load = time.perf_counter()
    engine = OpenJevVaultGemmaEngine.from_pretrained(args.model_dir, device=args.device)
    print(f"Loaded successfully in {time.perf_counter() - t_load:.2f}s!")

    if args.demo:
        print(f"\nRunning Clinical Demo across {len(DEMO_CLINICAL_NOTES)} realistic clinical scenarios...")
        for item in DEMO_CLINICAL_NOTES:
            evaluate_clinical_note(engine, item["text"], title=item["specialty"])
    elif args.text:
        evaluate_clinical_note(engine, args.text, title="User Supplied Clinical Note")
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        evaluate_clinical_note(engine, content, title=f"File: {os.path.basename(args.file)}")
    elif args.interactive:
        print("\n" + "=" * 70)
        print(" REDACTX INTERACTIVE CLINICAL NOTE SCANNER")
        print(" Paste your clinical note below.")
        print(" When done, enter a blank line with 'END' (or press Ctrl+C to exit).")
        print("=" * 70)
        while True:
            try:
                print("\n[Ready] Paste clinical note:")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                note_text = "\n".join(lines).strip()
                if not note_text:
                    print("Empty input. Exiting.")
                    break
                evaluate_clinical_note(engine, note_text, title="Interactive Clinical Note")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive scanner.")
                break


if __name__ == "__main__":
    main()
