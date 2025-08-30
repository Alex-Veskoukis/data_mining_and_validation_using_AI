"""
Classify each distinct feature from validated_features.json into one of the privacy-aware
predefined attribute classes using Azure OpenAI, and save results to JSON and CSV.

Inputs:
  - data/processed/validated_features.json

Outputs:
  - data/processed/attribute_classes.json
  - data/processed/attribute_classes.csv
"""

import time
import logging
import pandas as pd
import openai
from pathlib import Path
from tqdm import tqdm
import re
import inflect
from utils.io import PROC
from utils.openai_settings import (
    configure_openai,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    COMPLETION_PRICE_PER_1000_TOKENS
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
    "You are a compliance analyst. Assign each incoming feature name to exactly one "
    "of these privacy-relevant classes and provide a brief rationale (≤15 words). "
    "Return only JSON with keys `class` and `rationale`.\n\n"
    "Classes:\n"
    "1. Identifier_PII (e.g., SSN, passport number)\n"
    "2. Contact_Info (e.g., email, phone)\n"
    "3. Device_OnlineID (e.g., device ID, IP address)\n"
    "4. Biometric (e.g., fingerprint, face scan)\n"
    "5. Location_IoT (e.g., GPS, address)\n"
    "6. Health_Clinical\n"
    "7. Financial\n"
    "8. Child_Data (data about minors)\n"
    "9. Demographic (e.g., age, gender)\n"
    "10. Behavioural\n"
    "11. Environmental\n"
    "12. Operational_Business\n"
    "13. Other\n\n"
    'Example response: {"class":"Demographic","rationale":"Age indicates personal attribute that the current context mentions that it should not be disclosed."}'
)

# Configure OpenAI
configure_openai()

p = inflect.engine()

def sanitize_feature_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9 ]+', '', name)
    words = name.split()
    words = [p.singular_noun(w) if p.singular_noun(w) else w for w in words]
    if words:
        words[0] = words[0].capitalize()
        words[1:] = [w.lower() for w in words[1:]]
    return ' '.join(words).strip()

def classify_feature(feature_name: str, title: str, abstract: str) -> tuple[str, str, int, int]:
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

def main():
    df = pd.read_json(INPUT_JSON)
    df = df[df["feature_validation"] == "Valid"]

    # Prepare expanded feature rows with context
    rows = []
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

    classes, notes = [], []
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

    features_df.to_json(OUT_JSON, orient="records", indent=2)
    features_df.to_csv(OUT_CSV, index=False)
    logger.info(f"[ok] wrote attribute_classes.json and attribute_classes.csv with {len(features_df)} rows.")

    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}, cost=${total_cost:.4f}")

if __name__ == "__main__":
    main()