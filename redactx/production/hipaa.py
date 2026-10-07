"""
HIPAA Safe Harbor identifier categories (45 CFR 164.514(b)(2)) and a mapping from the entity labels used
by the training/evaluation datasets and by Microsoft Presidio onto those categories.

The mapping tables below were built from the label vocabularies actually observed in:
  nvidia/Nemotron-PII, gretelai/gretel-pii-masking-en-v1, ai4privacy/pii-masking-openpii-1m,
  Presidio's predefined recognizers, and redactx.data.generator.
"""

from enum import Enum
from typing import Dict


class Category(str, Enum):
    # --- HIPAA Safe Harbor identifiers ---
    NAME = "NAME"                          # (A) names
    LOCATION = "LOCATION"                  # (B) geographic subdivisions smaller than a state
    DATE = "DATE"                          # (C) dates (except year) related to an individual
    AGE = "AGE"                            # (C) ages over 89
    PHONE = "PHONE"                        # (D)
    FAX = "FAX"                            # (E)
    EMAIL = "EMAIL"                        # (F)
    SSN = "SSN"                            # (G)
    MRN = "MRN"                            # (H) medical record numbers
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"      # (I) health plan beneficiary numbers
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"      # (J) account numbers
    LICENSE_NUMBER = "LICENSE_NUMBER"      # (K) certificate / license numbers
    VEHICLE_ID = "VEHICLE_ID"              # (L) vehicle identifiers, license plates
    DEVICE_ID = "DEVICE_ID"                # (M) device identifiers / serial numbers
    URL = "URL"                            # (N)
    IP_ADDRESS = "IP_ADDRESS"              # (O)
    BIOMETRIC = "BIOMETRIC"                # (P)
    OTHER_ID = "OTHER_ID"                  # (R) any other unique identifying number or code
    CONTACT = "CONTACT"                    # phone / email / URL when the source does not say which
    # --- PII that is not a Safe Harbor identifier on its own ---
    COARSE_LOCATION = "COARSE_LOCATION"    # state, country
    ORGANIZATION = "ORGANIZATION"
    DEMOGRAPHIC = "DEMOGRAPHIC"            # gender, race, religion, occupation, ...
    CREDENTIAL = "CREDENTIAL"              # passwords, API keys, cookies, PINs, CVV
    UNKNOWN = "UNKNOWN"


HIPAA_IDENTIFIERS = frozenset({
    Category.NAME, Category.LOCATION, Category.DATE, Category.AGE, Category.PHONE, Category.FAX,
    Category.EMAIL, Category.SSN, Category.MRN, Category.HEALTH_PLAN_ID, Category.ACCOUNT_NUMBER,
    Category.LICENSE_NUMBER, Category.VEHICLE_ID, Category.DEVICE_ID, Category.URL, Category.IP_ADDRESS,
    Category.BIOMETRIC, Category.OTHER_ID, Category.CONTACT,
})

_EXACT: Dict[str, Category] = {
    # names
    "first_name": Category.NAME, "last_name": Category.NAME, "middle_name": Category.NAME, "name": Category.NAME,
    "givenname": Category.NAME, "surname": Category.NAME, "person": Category.NAME,
    # geography
    "street_address": Category.LOCATION, "address": Category.LOCATION, "city": Category.LOCATION,
    "county": Category.LOCATION, "postcode": Category.LOCATION, "zipcode": Category.LOCATION,
    "coordinate": Category.LOCATION, "street": Category.LOCATION, "buildingnum": Category.LOCATION,
    "location": Category.LOCATION,
    "state": Category.COARSE_LOCATION, "country": Category.COARSE_LOCATION,
    # dates
    "date": Category.DATE, "date_of_birth": Category.DATE, "date_time": Category.DATE, "time": Category.DATE,
    "age": Category.AGE,
    # contact
    "phone_number": Category.PHONE, "telephonenum": Category.PHONE, "fax_number": Category.FAX,
    "email": Category.EMAIL, "email_address": Category.EMAIL, "url": Category.URL,
    "ipv4": Category.IP_ADDRESS, "ipv6": Category.IP_ADDRESS, "ip_address": Category.IP_ADDRESS,
    "contact": Category.CONTACT,
    # government / health identifiers
    "ssn": Category.SSN, "socialnum": Category.SSN, "us_ssn": Category.SSN,
    "medical_record_number": Category.MRN,
    "health_plan_beneficiary_number": Category.HEALTH_PLAN_ID,
    "certificate_license_number": Category.LICENSE_NUMBER, "driverlicensenum": Category.LICENSE_NUMBER,
    "us_driver_license": Category.LICENSE_NUMBER, "medical_license": Category.LICENSE_NUMBER,
    # financial
    "account_number": Category.ACCOUNT_NUMBER, "credit_debit_card": Category.ACCOUNT_NUMBER,
    "credit_card_number": Category.ACCOUNT_NUMBER, "creditcardnumber": Category.ACCOUNT_NUMBER,
    "credit_card": Category.ACCOUNT_NUMBER, "bank_routing_number": Category.ACCOUNT_NUMBER,
    "swift_bic": Category.ACCOUNT_NUMBER, "iban_code": Category.ACCOUNT_NUMBER,
    "us_bank_number": Category.ACCOUNT_NUMBER,
    # vehicles / devices / biometrics
    "vehicle_identifier": Category.VEHICLE_ID, "license_plate": Category.VEHICLE_ID,
    "device_identifier": Category.DEVICE_ID, "mac_address": Category.DEVICE_ID,
    "biometric_identifier": Category.BIOMETRIC,
    # other unique identifiers
    "customer_id": Category.OTHER_ID, "employee_id": Category.OTHER_ID, "unique_id": Category.OTHER_ID,
    "unique_identifier": Category.OTHER_ID, "national_id": Category.OTHER_ID, "tax_id": Category.OTHER_ID,
    "taxnum": Category.OTHER_ID, "idcardnum": Category.OTHER_ID, "passportnum": Category.OTHER_ID,
    "us_passport": Category.OTHER_ID, "us_itin": Category.OTHER_ID, "user_name": Category.OTHER_ID,
    "identifier": Category.OTHER_ID, "uk_nhs": Category.OTHER_ID,
    # credentials
    "password": Category.CREDENTIAL, "api_key": Category.CREDENTIAL, "http_cookie": Category.CREDENTIAL,
    "pin": Category.CREDENTIAL, "cvv": Category.CREDENTIAL, "crypto": Category.CREDENTIAL,
    # organizations / demographics
    "company_name": Category.ORGANIZATION, "organization": Category.ORGANIZATION,
    "occupation": Category.DEMOGRAPHIC, "education_level": Category.DEMOGRAPHIC,
    "employment_status": Category.DEMOGRAPHIC, "race_ethnicity": Category.DEMOGRAPHIC,
    "religious_belief": Category.DEMOGRAPHIC, "political_view": Category.DEMOGRAPHIC,
    "sexuality": Category.DEMOGRAPHIC, "gender": Category.DEMOGRAPHIC, "sex": Category.DEMOGRAPHIC,
    "blood_type": Category.DEMOGRAPHIC, "language": Category.DEMOGRAPHIC, "title": Category.DEMOGRAPHIC,
    "nrp": Category.DEMOGRAPHIC,
}

_KEYWORDS = [
    (("mrn", "medical_record"), Category.MRN),
    (("ssn", "social_security", "socialnum"), Category.SSN),
    (("fax",), Category.FAX),
    (("phone", "telephone", "mobile"), Category.PHONE),
    (("email",), Category.EMAIL),
    (("url", "website", "http"), Category.URL),
    (("ipv", "ip_address"), Category.IP_ADDRESS),
    (("health_plan", "beneficiary", "insurance", "member_id"), Category.HEALTH_PLAN_ID),
    (("license_plate", "vehicle", "vin"), Category.VEHICLE_ID),
    (("license", "certificate"), Category.LICENSE_NUMBER),
    (("device", "serial", "mac_address", "imei"), Category.DEVICE_ID),
    (("biometric", "fingerprint"), Category.BIOMETRIC),
    (("account", "card", "iban", "routing", "bank"), Category.ACCOUNT_NUMBER),
    (("birth", "date", "dob"), Category.DATE),
    (("name",), Category.NAME),
    (("street", "address", "city", "zip", "postcode", "county", "coordinate", "building"), Category.LOCATION),
    (("password", "api_key", "token", "cookie", "secret"), Category.CREDENTIAL),
    (("_id", "id_", "number", "num", "identifier", "passport"), Category.OTHER_ID),
]


def to_category(label: str) -> Category:
    """Maps a dataset / Presidio / heuristic label onto a canonical Category."""
    key = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return Category.UNKNOWN
    if key in _EXACT:
        return _EXACT[key]
    if key.upper() in Category.__members__:
        return Category[key.upper()]
    for words, cat in _KEYWORDS:
        if any(w in key for w in words):
            return cat
    return Category.UNKNOWN


def is_hipaa_identifier(label_or_category) -> bool:
    cat = label_or_category if isinstance(label_or_category, Category) else to_category(str(label_or_category))
    return cat in HIPAA_IDENTIFIERS
