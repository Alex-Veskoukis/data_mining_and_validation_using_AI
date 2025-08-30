MASTER_LIST = [
    # Identifiers & PII
    "identifier_pii", "ssn", "social_security_number", "passport_number", "national_id", "tax_identifier",
    "driver_license_number", "license_plate", "account_number", "bank_account_number", "payment_card_number",
    "credit_score", "loan_amount", "account_balance", "insurance_policy_number", "nonpublic_personal_info",
    "consumer_report", "transaction_id", "customer_id", "user_id", "employee_id", "student_id",

    # Contact Info
    "contact_info", "email", "phone", "telephone", "address", "postal_address", "mailing_address",

    # Device & Online Identifiers
    "device_onlineid", "device_id", "imei", "mac_address", "uuid", "cookie_id", "pixel_id", "browser_fingerprint",
    "ip_address", "wifi_ssid", "bluetooth_id", "session_id", "tracking_id",

    # Biometric
    "biometric_id", "biometric_template", "fingerprint", "face_scan", "iris_scan", "voiceprint", "retina_scan",
    "dna", "genetic_data",

    # Location & IoT
    "location_iot", "location_data", "gps", "location_history", "address", "travel_itinerary", "pnr_element",
    "route_efficiency", "vehicle_identifier", "ambient_temperature", "ambient_humidity", "noise_level",

    "health_clinical", "health_data", "diagnosis", "treatment", "prescription", "medical_record", "disability_status",
    "mental_health", "vaccination_status", "lab_result",

    # Financial
    "financial", "income_amount", "family_income", "credit_limit", "transaction", "payment_history", "tax_information",
    "salary",
    "consumer_credit_liability_information",
    "credit",
    "credit_card",
    "credit_report",
    "credit_information",
    # Child Data
    "child_data", "minor_status", "student_record", "parental_consent",

    # Demographic
    "demographic", "age", "dob", "date_of_birth", "sex", "gender", "gender_identity", "nationality",
    "country_of_birth", "language", "marital_status", "education_level", "race", "ethnicity", "racial_ethnic_data",

    # Behavioural
    "behavioural", "user_activity", "clickstream", "browsing_history", "search_history", "app_usage",
    "purchase_history", "session_duration", "communication_content", "location_history",

    # Environmental
    "environmental", "environmental_reading", "energy_consumption", "inventory_level",

    # Operational/Business
    "operational_business", "model_parameter", "risk_score", "fraud_score", "operational_cost",

    # Sensitive/Special Categories
    "sexual_orientation", "political_opinion", "political_affiliation", "religious_belief", "trade_union_membership",
    "union_membership", "criminal_record", "disability", "genetic_data",

    # International & Synonyms
    "nhs_number", "aadhaar", "cpf", "curp", "cellphone", "mobile", "e-mail",

    # Healthcare
    "patient_id", "hospital_number", "medical_device_id", "insurance_claim", "clinical_trial_id",

    # Finance
    "iban", "swift_code", "investment_account", "portfolio_id",     "consumer_credit_liability_information",


    # Education
    "student_number", "school_id", "academic_record", "transcript",

    # Emerging Tech
    "blockchain_address", "crypto_wallet", "nft_id", "smart_contract_id",
    "voice_assistant_id", "virtual_reality_id", "augmented_reality_id",

    # Behavioral (additional)
    "mouse_movement", "keystroke_pattern", "scroll_depth", "time_on_page", "ad_click", "video_watch_history",

    # Sensitive (additional)
    "sexual_health", "fertility_status", "pregnancy_status", "military_status", "refugee_status", "immigration_status",
    "political_donation", "religious_affiliation", "philosophical_belief",

    # IoT/Sensor
    "smart_meter_id", "wearable_device_id", "fitness_tracker_id", "home_automation_id", "sensor_reading",

    # Metadata
    "data_source", "collection_timestamp", "consent_status", "data_retention_period", "data_controller", "data_processor",

    # Social/Communication
    "social_media_handle", "profile_url", "chat_id", "forum_username", "post_id", "comment_id"

 
    "communication_content",
    "consumer_report",
    "credit_history",
    "criminal_record",
    "education_information",
    "employee_id",
    "fax_number",
    "ip_address",
    "nonpublic_personal_info",
    "operational_business",
    "payment_account_number",
    "personal_data",
    "philosophical_belief",
    "political_opinion",
    "religion",
    "religious_affiliation",
    "religious_belief",
    "sexual_orientation",
    "tax_identifier",
    "trade_union_membership",
    "unsecured_protected_health_information",
    "other"
]


NORMALISE = {
    # Identifiers & PII
    "name": "identifier_pii",
    "first_name": "identifier_pii",
    "last_name": "identifier_pii",
    "ssn": "identifier_pii",
    "social_security_number": "identifier_pii",
    "passport": "passport_number",
    "passport_no": "passport_number",
    "passport_id": "passport_number",
    "national_identification_number": "national_id",
    "tax_id": "tax_identifier",
    "tax_number": "tax_identifier",
    "driver_license": "driver_license_number",
    "license": "license_plate",
    "account": "account_number",
    "bank_account": "bank_account_number",
    "card_number": "payment_card_number",
    "credit_card": "payment_card_number",
    "credit_card_number": "payment_card_number",
    "credit": "credit_score",
    "loan": "loan_amount",
    "balance": "account_balance",
    "insurance_number": "insurance_policy_number",
    "nonpublic_info": "nonpublic_personal_info",
    "transaction": "transaction_id",
    "customer": "customer_id",
    "user": "user_id",
    "employee": "employee_id",
    # Contact Info
    "contact": "contact_info",
    "email_address": "email",
    "phone_number": "phone",
    "telephone_number": "telephone",
    "postal": "postal_address",
    "mailing": "mailing_address",

    # Device & Online Identifiers
    "device": "device_id",
    "imei_number": "imei",
    "mac": "mac_address",
    "uuid_code": "uuid",
    "cookie": "cookie_id",
    "pixel": "pixel_id",
    "fingerprint_browser": "browser_fingerprint",
    "ip": "ip_address",
    "wifi": "wifi_ssid",
    "bluetooth": "bluetooth_id",
    "session": "session_id",
    "tracking": "tracking_id",

    # Biometric
    "biometric": "biometric_id",
    "biometric_data": "biometric_id",
    "finger_print": "fingerprint",
    "face": "face_scan",
    "iris": "iris_scan",
    "voice": "voiceprint",
    "retina": "retina_scan",
    "dna_sequence": "dna",
    # Location & IoT
    "location": "location_data",
    "gps_coordinates": "gps",
    "location_hist": "location_history",
    "travel": "travel_itinerary",
    "pnr": "pnr_element",
    "vehicle": "vehicle_identifier",
    "ambient_temp": "ambient_temperature",
    "ambient_humid": "ambient_humidity",
    "noise": "noise_level",

    # Health & Clinical
    "health": "health_data",
    "diagnosis_code": "diagnosis",
    "treatment_plan": "treatment",
    "prescription_drug": "prescription",
    "medical": "medical_record",
    "mental": "mental_health",
    "vaccine": "vaccination_status",
    "lab": "lab_result",

    # Financial
    "income": "income_amount",
    "family_income": "family_income",
    "credit_limit": "credit_limit",
    "transaction_history": "transaction",
    "payment": "payment_history",
    "tax": "tax_information",
    "credit_report": "credit_score",
    "credit_information": "credit_score",
    "consumer_credit_liability_information": "credit_score",

    # Child Data
    "minor": "minor_status",
    "student": "student_record",
    "parental": "parental_consent",

    # Demographic
    "birth_date": "dob",
    "date_of_birth": "dob",
    "sex": "sex",
    "gender": "gender",
    "gender_id": "gender_identity",
    "nationality": "nationality",
    "country_of_birth": "country_of_birth",
    "language": "language",
    "marital": "marital_status",
    "education": "education_level",
    "race": "race",
    "ethnicity": "ethnicity",
    "racial_ethnic": "racial_ethnic_data",

    # Behavioural
    "activity": "user_activity",
    "click": "clickstream",
    "browsing": "browsing_history",
    "search": "search_history",
    "app": "app_usage",
    "purchase": "purchase_history",
    "session_time": "session_duration",
    "communication": "communication_content",

    # Environmental
    "environment": "environmental_reading",
    "energy": "energy_consumption",
    "inventory": "inventory_level",

    # Operational/Business
    "operational": "operational_business",
    "model": "model_parameter",
    "risk": "risk_score",
    "fraud": "fraud_score",
    "cost": "operational_cost",

    # Sensitive/Special Categories
    "sexual": "sexual_orientation",
    "political": "political_opinion",
    "political_affiliation": "political_affiliation",
    "religious": "religious_belief",
    "union": "trade_union_membership",
    "criminal": "criminal_record",
    "disability": "disability_status",
    "genetic": "genetic_data",

    # International & Synonyms
    "nhs": "nhs_number",
    "aadhaar_id": "aadhaar",
    "cpf_id": "cpf",
    "curp_id": "curp",
    "cell": "cellphone",
    "mobile_phone": "mobile",
    "email": "e-mail",

    # Healthcare
    "patient": "patient_id",
    "hospital": "hospital_number",
    "medical_device": "medical_device_id",
    "insurance_claim": "insurance_claim",
    "clinical_trial": "clinical_trial_id",

    # Finance
    "iban_code": "iban",
    "swift": "swift_code",
    "investment": "investment_account",
    "portfolio": "portfolio_id",

    # Education
    "student_num": "student_number",
    "school": "school_id",
    "academic": "academic_record",
    "transcript_record": "transcript",

    # Emerging Tech
    "blockchain": "blockchain_address",
    "crypto": "crypto_wallet",
    "nft": "nft_id",
    "smart_contract": "smart_contract_id",
    "voice_assistant": "voice_assistant_id",
    "vr": "virtual_reality_id",
    "ar": "augmented_reality_id",

    # Behavioral (additional)
    "mouse": "mouse_movement",
    "keystroke": "keystroke_pattern",
    "scroll": "scroll_depth",
    "time_on_page": "time_on_page",
    "ad_click": "ad_click",
    "video_watch": "video_watch_history",

    # Sensitive (additional)
    "sexual_health": "sexual_health",
    "fertility": "fertility_status",
    "pregnancy": "pregnancy_status",
    "military": "military_status",
    "refugee": "refugee_status",
    "immigration": "immigration_status",
    "political_donation": "political_donation",
    "philosophical": "philosophical_belief",

    # IoT/Sensor
    "smart_meter": "smart_meter_id",
    "wearable": "wearable_device_id",
    "fitness_tracker": "fitness_tracker_id",
    "home_automation": "home_automation_id",
    "sensor": "sensor_reading",

    # Metadata
    "source": "data_source",
    "timestamp": "collection_timestamp",
    "consent": "consent_status",
    "retention": "data_retention_period",
    "controller": "data_controller",
    "processor": "data_processor",

    # Social/Communication
    "social_media": "social_media_handle",
    "profile": "profile_url",
    "chat": "chat_id",
    "forum": "forum_username",
    "post": "post_id",
    "comment": "comment_id",
    "affected individual": "identifier_pii",
    "biometric_information": "biometric_id",
    "biometrics": "biometric_id",
    "browser_identifier": "device_onlineid",
    "cardholder_data": "payment_card_number",
    "child abuse data": "child_data",
    "communication_content": "communication_content",
    "consumer_report": "consumer_report",
    "credit_eligibility_information": "credit_score",
    "credit_worthiness": "credit_score",
    "creditworthiness": "credit_score",
    "criminal_record": "criminal_record",
    "data personol": "personal_data",
    "education_data": "education_level",
    "employee_id": "employee_id",
    "employee_record": "employee_id",
    "employer": "operational_business",
    "employment-related information": "operational_business",
    "genetic_relative": "genetic_data",
    "government related identifier": "identifier_pii",
    "health_information": "health_clinical",
    "health_record": "health_clinical",
    "ip_address": "ip_address",
    "neural data": "health_clinical",
    "nonpublic_personal_info": "nonpublic_personal_info",
    "patient identifying information": "patient_id",
    "payment_account": "payment_account_number",
    "payment_account_number": "payment_account_number",
    "payment_information": "payment_history",
    "payment_transaction": "transaction_id",
    "personal identifiable information": "identifier_pii",
    "personal info": "personal_data",
    "personal information": "personal_data",
    "personal_data": "personal_data",
    "personal_data_breach": "other",
    "personal_info": "personal_data",
    "personal_information": "personal_data",
    "personal_insolvency_information": "financial",
    "philosophical_belief": "philosophical_belief",
    "political_opinion": "political_opinion",
    "profiling": "behavioural",
    "protected health information": "health_clinical",
    "protected_health_information": "health_clinical",
    "psychotherapy_notes": "health_clinical",
    "purchase_history": "purchase_history",
    "relevant personal data": "personal_data",
    "religious_belief": "religious_belief",
    "renseignement personnel": "personal_data",
    "repayment_history_information": "financial",
    "sensitive personal data": "personal_data",
    "sensitive personal information": "personal_data",
    "sensitive processing": "other",
    "sensitive_personal_information": "personal_data",
    "sensory_disability": "disability_status",
    "sexual_orientation": "sexual_orientation",
    "special categories of data": "other",
    "special categories of personal data": "other",
    "special_categories": "other",
    "substance use disorder": "health_clinical",
    "tax_identifier": "tax_identifier",
    "trade_union_membership": "trade_union_membership",
    "unieke identifiseerder": "identifier_pii",
    "union_membership": "union_membership",
    "unique identifier": "identifier_pii",
    "unique patient identifier": "patient_id",
    "unsecured PHR identifiable health information": "health_clinical",
    "user_activity_data": "user_activity",
    "PHR identifiable health": "health_clinical",
    "PHR identifiable health information": "health_clinical",
    "blood_type": "health_clinical",
    "credit_history": "credit_score",
    "default information": "financial",
    "education information": "education_level",
    "fax_number": "contact_info",
    "other": "other",
    "personal data": "personal_data",
    "personal data breach": "other",
    "religion": "religious_belief",
    "religious_affiliation": "religious_belief",
    "unsecured protected health information": "health_clinical"
}