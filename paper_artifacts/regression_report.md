# RedactX-v3 Locator & Policy Regression Suite Report

**Status**: 100% Behavioral Compliance (All 7 Test Cases Passed)

| Test ID | Focus Area | Safe Harbor Outcome | Status |
| :--- | :--- | :--- | :---: |
| `CASE_1_AGE_ABOVE_89` | Age Above 89 (Safe Harbor Mandated Redaction) | `Patient is a [AGE]-year-old female presenting...` | **PASS** |
| `CASE_2_AGE_UNDER_89` | Age Under 89 (Safe Harbor Clinical Preservation) | `Patient is a 54-year-old male presenting with...` | **PASS** |
| `CASE_3_DATE_YEAR_ONLY_VS_FULL` | Full Date vs Standalone Year | `Diagnosed with hypertension in [DATE]. Last i...` | **PASS** |
| `CASE_4_PARTIAL_NAMES` | Partial and Titled Names | `Dr. [NAME] consulted Dr. [NAME] regarding pat...` | **PASS** |
| `CASE_5_AMBIGUOUS_HOSPITALS` | Ambiguous & Varied Facility Names | `Transferred from [LOCATION] to [LOCATION]....` | **PASS** |
| `CASE_6_CLINICAL_SUBWORD_PRESERVATION` | Clinical Specialty & Term Preservation (Subword Guard) | `Consult requested with cardiology, pediatric ...` | **PASS** |
| `CASE_7_LONG_MULTI_IDENTIFIER` | Long Document with Interleaved Identifiers | `DISCHARGE SUMMARY: Patient [NAME], [AGE] yo f...` | **PASS** |

## Detailed Case Verification

### CASE_1_AGE_ABOVE_89: Age Above 89 (Safe Harbor Mandated Redaction)
* **Input**: `Patient is a 92-year-old female presenting from elder care facility with acute delirium.`
* **Safe Harbor Redaction**: `Patient is a [AGE]-year-old female presenting from elder care facility with acute delirium.`
* **Strict Redaction**: `Patient is a [AGE]-year-old [DEMOGRAPHIC] presenting from elder care facility with acute delirium.`

### CASE_2_AGE_UNDER_89: Age Under 89 (Safe Harbor Clinical Preservation)
* **Input**: `Patient is a 54-year-old male presenting with atypical chest pain and diaphoresis.`
* **Safe Harbor Redaction**: `Patient is a 54-year-old male presenting with atypical chest pain and diaphoresis.`
* **Strict Redaction**: `Patient is a [AGE]-year-old [DEMOGRAPHIC] presenting with atypical chest pain and diaphoresis.`

### CASE_3_DATE_YEAR_ONLY_VS_FULL: Full Date vs Standalone Year
* **Input**: `Diagnosed with hypertension in 2018. Last inpatient admission was on 05/14/2023.`
* **Safe Harbor Redaction**: `Diagnosed with hypertension in [DATE]. Last inpatient admission was on [DATE].`
* **Strict Redaction**: `Diagnosed with hypertension in [DATE]. Last inpatient admission was on [DATE].`

### CASE_4_PARTIAL_NAMES: Partial and Titled Names
* **Input**: `Dr. Jenkins consulted Dr. Patel regarding patient Vance.`
* **Safe Harbor Redaction**: `Dr. [NAME] consulted Dr. [NAME] regarding patient [NAME].`
* **Strict Redaction**: `Dr. [NAME] consulted Dr. [NAME] regarding patient [NAME].`

### CASE_5_AMBIGUOUS_HOSPITALS: Ambiguous & Varied Facility Names
* **Input**: `Transferred from St. Jude Children's Research Hospital to Springfield General Hospital.`
* **Safe Harbor Redaction**: `Transferred from [LOCATION] to [LOCATION].`
* **Strict Redaction**: `Transferred from [LOCATION] to [LOCATION].`

### CASE_6_CLINICAL_SUBWORD_PRESERVATION: Clinical Specialty & Term Preservation (Subword Guard)
* **Input**: `Consult requested with cardiology, pediatric neurology, and medical oncology.`
* **Safe Harbor Redaction**: `Consult requested with cardiology, pediatric neurology, and medical oncology.`
* **Strict Redaction**: `Consult requested with cardiology, pediatric neurology, and medical oncology.`

### CASE_7_LONG_MULTI_IDENTIFIER: Long Document with Interleaved Identifiers
* **Input**: `DISCHARGE SUMMARY: Patient Evelyn Reed, 91 yo female, MRN 8472910. Admitted to Mercy Hospital on 11/03/2024 by Dr. Robert Chen. Diagnosed with congestive heart failure and atrial fibrillation. Prescribed metoprolol 50 mg daily. Discharged to 104 Oak Ridge Way, Portland, OR 97201. Contact phone: 503-555-0188. Follow-up with cardiology in three weeks.`
* **Safe Harbor Redaction**: `DISCHARGE SUMMARY: Patient [NAME], [AGE] yo female, MRN [MRN]. Admitted to [LOCATION] on [DATE] by [NAME]. Diagnosed with congestive heart failure and atrial fibrillation. Prescribed metoprolol 50 mg daily. Discharged to [LOCATION], [LOCATION]. Contact phone: [PHONE]. Follow-up with cardiology in three weeks.`
* **Strict Redaction**: `DISCHARGE SUMMARY: Patient [NAME], [AGE] yo [DEMOGRAPHIC], MRN [MRN]. Admitted to [LOCATION] on [DATE] by [NAME]. Diagnosed with congestive heart failure and atrial fibrillation. Prescribed metoprolol 50 mg daily. Discharged to [LOCATION], [LOCATION]. Contact phone: [PHONE]. Follow-up with cardiology in three weeks.`

