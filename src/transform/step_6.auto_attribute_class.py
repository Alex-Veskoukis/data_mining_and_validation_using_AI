"""
Privacy-Aware Attribute Classification Module

Input: data/processed/validated_features.json (validated feature set).
Outputs: data/processed/attribute_classes.json and data/processed/attribute_classes.csv.

This module implements an automated classification pipeline for decision tree features into
privacy-aware attribute classes aligned with data protection regulations. The classification
employs large language model (LLM) inference with a comprehensive taxonomy of thirteen attribute
classes covering personal identifiable information (PII), sensitive data categories, and
operational attributes. This classification enables privacy impact assessment and regulatory
compliance analysis across diverse application domains.

Input Specifications:
    data/processed/validated_features.json: Validation-annotated corpus with verified
    features from the feature validation pipeline

Output Specifications:
    data/processed/attribute_classes.json: Privacy class-annotated features in JSON format
    data/processed/attribute_classes.csv: Privacy class-annotated features in CSV format

Methodological Approach:
    1. Corpus filtering for validated features with domain consistency
    2. Feature extraction and context preparation with title and abstract
    3. Deduplication across feature-document pairs
    4. LLM-based privacy classification using structured decision rules
    5. Token-based cost estimation with comprehensive logging
    6. Rate-limited API interaction for stability
    7. Dual-format persistence for analytical flexibility

Privacy Taxonomy:
    The classification schema encompasses thirteen attribute classes aligned with GDPR,
    CCPA, and related privacy frameworks: Identifier_PII, Contact_Info, Device_OnlineID,
    Biometric, Location_IoT, Health_Clinical, Financial, Child_Data, Demographic,
    Behavioural, Environmental, Operational_Business, Other. Each class is defined with
    explicit decision rules applied in hierarchical order to ensure consistent classification.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import inflect
import openai
import pandas as pd
from tqdm import tqdm

from src.utils.io import PROC
from src.utils.openai_settings import (
    COMPLETION_PRICE_PER_1000_TOKENS,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    configure_openai,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(PROC)
INPUT_JSON    = PROCESSED_DIR / "validated_features.json"
OUT_JSON      = PROCESSED_DIR / "attribute_classes.json"
OUT_CSV       = PROCESSED_DIR / "attribute_classes.csv"

RATE_DELAY = 1.2
MAX_TOK    = 60

CLASSES = [
    "Identifier_PII", "Contact_Info", "Device_OnlineID", "Biometric", "Location_IoT",
    "Health_Clinical", "Financial", "Child_Data", "Demographic", "Behavioural",
    "Environmental", "Operational_Business", "Other"
]

SYSTEM_PROMPT = (
    "Role: Compliance analyst.\n"
    "Task: Map ONE incoming feature name to EXACTLY ONE class. Provide a brief rationale (≤15 words).\n"
    "Output: Return ONLY a JSON object with keys `class` and `rationale`. No extra text.\n"
    "Do not transform, normalize, or expand the feature name. If ambiguous, choose Other.\n\n"
    "Classes:\n"
    "1. Identifier_PII (SSN, national_id, passport)\n"
    "2. Contact_Info (email, phone, messenger handle)\n"
    "3. Device_OnlineID (device_id, IP, cookie_id, session_id, ad_id, MAC, IMEI)\n"
    "4. Biometric (fingerprint, face embedding, iris, voiceprint)\n"
    "5. Location_IoT (GPS, latitude, longitude, home/work address, cell tower)\n"
    "6. Health_Clinical (diagnoses, labs, ICD codes, medications, vital signs)\n"
    "7. Financial (income, salary, credit score, card/account numbers, balance)\n"
    "8. Child_Data (data about minors or pupils)\n"
    "9. Demographic (age, gender, ethnicity, nationality, marital status)\n"
    "10. Behavioural (clicks, browsing, purchase history, time-on-page, login frequency)\n"
    "11. Environmental (weather, temperature, air quality, noise level, light)\n"
    "12. Operational_Business (SKU, product_id, transaction_id, order_id, process status)\n"
    "13. Other\n\n"
    "Decision rules (apply in order):\n"
    "1) Biometric term → Biometric.\n"
    "2) Health/clinical term or code → Health_Clinical.\n"
    "3) Device/online identifier (IP, cookie, device_id, session_id, ad_id) → Device_OnlineID.\n"
    "4) Location/address/coordinates/GPS/cell tower → Location_IoT.\n"
    "5) Government/person identifier (SSN, passport, national_id, tax_id) → Identifier_PII.\n"
    "6) Contact channel (email, phone, messenger handle) → Contact_Info.\n"
    "7) Financial value or account/card numbers → Financial.\n"
    "8) Child/minor/student-specific data → Child_Data.\n"
    "9) Human attribute like age/gender/race → Demographic.\n"
    "10) User actions/usage patterns → Behavioural.\n"
    "11) Ambient conditions/sensors not tied to identity → Environmental.\n"
    "12) Business/operational artifacts (SKU, order_id, process fields) → Operational_Business.\n"
    "13) Else → Other.\n\n"
    "Ambiguity:\n"
    "- If the term is an OUTCOME/label, still classify by its data type.\n"
    "- If the term is generic:\n"
    "  • 'user_id'/'patient_id'/'customer_id' → Identifier_PII.\n"
    "  • 'device_id'/'session_id'/'ad_id' → Device_OnlineID.\n"
    "  • 'order_id'/'transaction_id'/'sku' → Operational_Business.\n"
    "- If insufficient context, choose Other and state why.\n\n"
    "Constraints:\n"
    "- Exactly one class.\n"
    "- Rationale ≤15 words. Plain and factual.\n"
    "- Output must be valid JSON with only `class` and `rationale`.\n\n"
    "Examples (inputs → outputs):\n"
    "feature='ip_address' → {\"class\":\"Device_OnlineID\",\"rationale\":\"Network identifier used for online tracking\"}\n"
    "feature='home_address' → {\"class\":\"Location_IoT\",\"rationale\":\"Physical location of an individual\"}\n"
    "feature='age' → {\"class\":\"Demographic\",\"rationale\":\"Personal attribute describing age\"}\n"
    "feature='user_id' → {\"class\":\"Identifier_PII\",\"rationale\":\"Direct identifier of a person\"}\n"
    "feature='order_id' → {\"class\":\"Operational_Business\",\"rationale\":\"Business transaction identifier\"}\n"
    "feature='heart_rate' → {\"class\":\"Health_Clinical\",\"rationale\":\"Clinical vital sign measurement\"}\n"
    "feature='credit_card_number' → {\"class\":\"Financial\",\"rationale\":\"Financial account identifier\"}\n"
    "feature='student_grade' → {\"class\":\"Child_Data\",\"rationale\":\"Data concerning a minor’s schooling\"}\n"
    "feature='click_through_rate' → {\"class\":\"Behavioural\",\"rationale\":\"User interaction behavior metric\"}\n"
    "feature='pm2_5' → {\"class\":\"Environmental\",\"rationale\":\"Ambient air quality measure\"}\n"
    "feature='email' → {\"class\":\"Contact_Info\",\"rationale\":\"Direct electronic contact channel\"}\n"
    "feature='face_embedding' → {\"class\":\"Biometric\",\"rationale\":\"Unique biometric template\"}\n"
    "feature='unknown_feature_x' → {\"class\":\"Other\",\"rationale\":\"Insufficient information to classify reliably\"}\n"
)


configure_openai()

p = inflect.engine()


def sanitize_feature_name(name: str) -> str:
    """
    Normalize feature name through character sanitization and capitalization.
    
    This function implements feature name normalization by removing non-alphanumeric
    characters and applying title-case capitalization. The normalization ensures
    consistent feature representation across heterogeneous extraction sources while
    preserving semantic content for downstream classification tasks.
    
    Parameters
    ----------
    name : str
        Raw feature name potentially containing special characters and inconsistent casing.
    
    Returns
    -------
    str
        Sanitized feature name with alphanumeric characters only and title-case formatting.
    
    Algorithm
    ---------
    1. Remove all non-alphanumeric characters except spaces using regex
    2. Split normalized string into word tokens
    3. Capitalize first word
    4. Convert remaining words to lowercase
    5. Join words with single spaces
    """
    name = re.sub(r'[^A-Za-z0-9 ]+', '', name)
    words = name.split()
    if words:
        words[0] = words[0].capitalize()
        words[1:] = [w.lower() for w in words[1:]]
    return ' '.join(words).strip()


def classify_feature(feature_name: str, title: str, abstract: str) -> tuple[str, str, int, int]:
    """
    Execute privacy-aware classification for decision tree feature.
    
    This function implements LLM-based privacy classification using a structured taxonomy
    of thirteen attribute classes aligned with data protection regulations. The classification
    employs hierarchical decision rules and contextual information from publication metadata
    to assign features to appropriate privacy categories. Zero-temperature sampling ensures
    deterministic classification with structured JSON output including class assignment and
    concise rationale.
    
    Parameters
    ----------
    feature_name : str
        Sanitized feature name requiring privacy classification.
    title : str
        Publication title providing contextual information.
    abstract : str
        Publication abstract providing detailed feature usage context.
    
    Returns
    -------
    tuple[str, str, int, int]
        Four-element tuple containing:
        - Attribute class from predefined privacy taxonomy or 'Other'
        - Rationale explaining classification decision (≤15 words)
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct classification prompt with feature name and contextual metadata
    2. Submit classification request to Azure OpenAI ChatCompletion endpoint
    3. Apply zero-temperature sampling for deterministic privacy categorization
    4. Parse JSON response containing class and rationale
    5. Validate class membership in predefined taxonomy
    6. Handle malformed outputs with 'Other' class fallback
    7. Log errors and warnings for debugging and quality monitoring
    """
    user_msg = (
        f"Feature name: {feature_name}\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )
    try:
        response = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_msg}
            ],
            temperature=0.0,
            max_tokens=MAX_TOK
        )
        text = response.choices[0].message.content.strip()
        usage = response.usage
        if text.startswith("{") and text.endswith("}"):
            result = eval(text)
            cls = result.get("class", "").strip()
            rat = result.get("rationale", "").strip()
            if cls in CLASSES:
                return cls, rat, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            return "Other", f"Invalid class '{cls}'", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return "Other", text[:15] + "...", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except Exception as e:
        logger.warning(f"API error for '{feature_name}': {e}")
        return "Other", "API_error", 0, 0

def main() -> None:
    """
    Execute corpus-wide privacy classification pipeline for validated features.
    
    This function orchestrates the complete privacy classification workflow including corpus
    ingestion, validation filtering, feature extraction with contextual metadata, deduplication
    across feature-document pairs, iterative LLM-based privacy classification, cost estimation,
    and dual-format persistence. The pipeline processes only features that passed validation
    and have consistent domain assignments, ensuring classification quality. Rate limiting
    (1.2-second intervals) ensures API compliance with comprehensive logging.
    
    Returns
    -------
    None
        Results are persisted to OUT_JSON and OUT_CSV with no return value.
    
    Algorithm
    ---------
    1. Corpus Ingestion: Load validation-annotated bibliographic records from JSON
    2. Dual Filtering: Select 'Valid' features with matching domain validation
    3. Feature Expansion: Extract individual features with publication context
    4. Name Sanitization: Normalize feature names for consistent representation
    5. Deduplication: Remove duplicate feature-context combinations
    6. Iterative Classification: Process each feature through LLM privacy classifier
    7. Token Accumulation: Aggregate prompt and completion token usage
    8. Cost Estimation: Calculate cumulative API costs using pricing constants
    9. Result Annotation: Append privacy classes and rationales to feature records
    10. Dual Persistence: Write privacy-annotated features to CSV and JSON formats
    11. Summary Logging: Report token counts, estimated costs, and output locations
    
    Output Side Effects
    -------------------
    Writes privacy-classified features to PROCESSED_DIR in two formats:
        - attribute_classes.csv: Tabular CSV format
        - attribute_classes.json: Structured JSON format
    
    Logs comprehensive progress information including:
        - Feature-context row preparation statistics
        - Progress bar for classification iterations
        - Total token consumption statistics
        - Estimated API costs
        - Classification warnings and errors
    """
    df = pd.read_json(str(INPUT_JSON))
    df = df[df["feature_validation"] == "Valid"]
    df = df[df["domain"] == df["domain_validated"]]
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        doi = row.get("doi", "")
        domain_validated = row.get("domain_validated", "")
        features = [sanitize_feature_name(f) for f in str(row.get("features", "")).split(";") if f.strip()]
        for feat in features:
            rows.append({
                "feature_clean": feat,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "domain_validated": domain_validated
            })

    features_df = pd.DataFrame(rows)
    features_df = features_df.drop_duplicates(subset=["feature_clean", "title", "abstract", "doi", "domain_validated"])
    logger.info(f"Prepared {len(features_df)} feature-context rows for classification.")

    classes: list[str] = []
    notes: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Classifying features"):
        cls, rat, p_tok, c_tok = classify_feature(row["feature_clean"], row["title"], row["abstract"])
        classes.append(cls)
        notes.append(rat)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        time.sleep(RATE_DELAY)

    features_df["attribute_class"] = classes
    features_df["notes"]           = notes

    features_df.to_json(str(OUT_JSON), orient="records", indent=2)
    features_df.to_csv(str(OUT_CSV), index=False)
    logger.info(f"[ok] wrote attribute_classes.json and attribute_classes.csv with {len(features_df)} rows.")

    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}, cost=${total_cost:.4f}")

if __name__ == "__main__":
    main()
