"""
MAIN.8.validate_feature_regulation.py

Validates whether features (merged by attribute class with regulations) are actually 
mentioned to be regulated in the regulatory text.

This module takes features that were matched with regulatory clauses based on 
attribute class and validates using OpenAI whether the specific feature is 
actually regulated according to the quoted regulatory text.

Inputs:
  - data/processed/attribute_classes.csv (features with their attribute classes)
  - data/processed/reg_sections_clauses.csv (regulatory clauses by attribute class)

Outputs:
  - data/processed/validated_feature_regulation.csv
  - data/processed/validated_feature_regulation.json
  - data/processed/feature_regulation_validation_summary.xlsx
"""

import time
import json
import logging
from collections import OrderedDict

import pandas as pd
import openai
from pathlib import Path
from tqdm import tqdm
from utils.io import PROC
from utils.openai_settings import (
    configure_openai,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    COMPLETION_PRICE_PER_1000_TOKENS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
configure_openai()

# File paths
PROCESSED_DIR = Path(PROC)
FEATURES_FILE = PROCESSED_DIR / "attribute_classes.csv"
CLAUSES_FILE = PROCESSED_DIR / "reg_sections_clauses.csv"
logger.info(PROCESSED_DIR)
# Output files
OUTPUT_CSV = PROCESSED_DIR / "validated_feature_regulation.csv"
OUTPUT_JSON = PROCESSED_DIR / "validated_feature_regulation.json"
SUMMARY_XLSX = PROCESSED_DIR / "feature_regulation_validation_summary.xlsx"

# Rate limiting
RATE_DELAY = 1.0

def validate_feature_regulation(feature_name, attribute_class, notes, quoted_text):
    """
    Validate whether a specific feature is actually regulated based on the regulatory text.

    Returns:
        tuple: (regulation_status, confidence, rationale, prompt_tokens, completion_tokens)
    """

    system_prompt = """You are a legal expert analyzing whether specific machine learning features are regulated by privacy and data protection laws.

Your task is to determine if a specific feature mentioned in a research paper is actually regulated according to the provided regulatory text.

Respond with exactly one of these regulation statuses:
- "Regulated": The feature is clearly covered by the regulation either because the whole attribute class is mentioned to be regulated or this exact feature.
- "Not Regulated": The feature is not mentioned to be regulated neither its whole attribute class.

Also provide a confidence level: "High", "Medium", or "Low"

Format your response as:
STATUS: [Regulated|Not Regulated]
CONFIDENCE: [High|Medium|Low]
RATIONALE: [Brief explanation of your reasoning]


Example 1 — Regulated:
STATUS: Regulated
CONFIDENCE: High
RATIONALE: 'The regulatory text explicitly refers to the collection and processing of personally identifiable information, including biometric identifiers such as fingerprints. Since the feature "fingerprint pattern" falls within the Biometric_Identifier attribute class, and this class is directly covered by the regulation, it is considered regulated.'

Example 2 — Not Regulated:
STATUS: Not Regulated
CONFIDENCE: High
RATIONALE: 'The regulatory text provided focuses on the confidentiality and handling of substance use disorder patient records, and does not mention any specific features related to health clinical attributes, including this specific feature "05 khz" or hearing loss screening. The Health_Clinical category is not addressed in the regulations, indicating that it is not regulated under the provided context.'
"""

    user_prompt = f"""FEATURE TO VALIDATE: {feature_name}
FEATURE ATTRIBUTE CLASS: {attribute_class}
Notes: {notes}

REGULATORY CONTEXT:
REGULATORY TEXT:
"{quoted_text}"

QUESTION: Is the feature "{feature_name}" regulated either specifically or its whole category {attribute_class},
 according to this regulatory context?
"""

    try:
        response = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()
        usage = response.usage

        # Parse response
        lines = content.split('\n')
        status = "Not Clearly Regulated"  # default
        confidence = "Low"  # default
        rationale = "Unable to parse OpenAI response"

        for line in lines:
            if line.startswith("STATUS:"):
                status = line.replace("STATUS:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()

        return (status, confidence, rationale,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    except Exception as e:
        logger.error(f"OpenAI API error for {feature_name}: {e}")
        return ("Not Clearly Regulated", "Low", f"API error: {str(e)}", 0, 0)

def load_and_prepare_data():
    """Load and merge all necessary data files."""
    logger.info("Loading data files...")

    # Load features with attribute classes
    features_df = pd.read_csv(FEATURES_FILE)
    logger.info(f"Loaded {len(features_df)} features with attribute classes")

    # Load regulatory clauses
    clauses_df = pd.read_csv(CLAUSES_FILE)

    logger.info(f"Loaded {len(clauses_df)} regulatory clauses")

    return features_df, clauses_df

group_cols = [
    "feature_clean", "title", "abstract", "doi",
    "domain_validated", "attribute_class", "notes",
    "regulated", "reg_id",
]

def _concat_unique_semicolon(series: pd.Series) -> str:
    """Concatenate unique non-null strings in first-occurrence order with ';'."""
    seen = set()
    ordered = []
    for x in series:
        if pd.isna(x):
            continue
        s = str(x)
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ";".join(ordered)

def _map_article_to_quotes(article_ref: pd.Series, quoted_text: pd.Series) -> dict:
    """
    Map each article_ref to an order-preserving, de-duplicated list of quotes.
    Rows with null article_ref or null quoted_text are skipped.
    """
    od = OrderedDict()
    for a, q in zip(article_ref, quoted_text):
        if pd.isna(a) or pd.isna(q):
            continue
        a_str = str(a)
        q_str = str(q)
        if a_str not in od:
            od[a_str] = []
        # de-duplicate while preserving first occurrence
        if q_str not in od[a_str]:
            od[a_str].append(q_str)
    return dict(od)


def merge_features_with_regulations(features_df, clauses_df):
    """
    Merge features with regulatory clauses based on attribute class.
    Each feature will be matched with all relevant regulatory clauses.
    """
    logger.info("Merging features with regulatory clauses...")

    # Merge features with clauses based on attribute_class
    # --- 1. Split and trim -------------------------------------------------------
    clauses_df["attribute_class"] = (
        clauses_df["attribute_class"]
        .str.split(";")  # ➊ break into lists
        .apply(lambda lst: [s.strip() for s in lst])  # ➋ remove stray spaces
    )

    # --- 2. Explode --------------------------------------------------------------
    clauses_df = clauses_df.explode("attribute_class", ignore_index=True)
    clauses_df = clauses_df[clauses_df["attribute_class"] != "Other"]

    merged_df = features_df.merge(
        clauses_df,
        on='attribute_class',
        how='inner'
    )
    # --- Aggregation -------------------------------------------------------------
    result_df = (
        merged_df
        .groupby(group_cols, dropna=False, as_index=False)[["article_ref", "quoted_text"]]
        .apply(
            lambda g: pd.Series({
                "article_ref": _concat_unique_semicolon(g["article_ref"]),
                "quoted_text": _map_article_to_quotes(g["article_ref"], g["quoted_text"]),
            })
        )
        .reset_index()  # grouping columns are now in the index → bring them back
    )
    logger.info(f"Created {len(result_df)} feature-regulation pairs")

    # Remove rows where no regulatory clause was found
    regulated_df = result_df.dropna(subset=['quoted_text'])

    important_privacy_regulations = [
        # EU - ranked by importance
        "GDPR",  # Core EU privacy framework, global benchmark
        "ePrivacy Directive",  # Key for cookies, electronic comms, and marketing consent
        "NIS2",  # Critical for security obligations, personal data protection
        "PSD2",  # Payment data sharing & security rules
        "EU eHealth Network",  # Cross-border health data guidance

        # US - ranked by importance
        "CCPA",  # Baseline US consumer privacy (California)
        "CPRA",  # Strengthens CCPA, adds sensitive data category
        "HIPAA",  # Health data privacy and security
        "HITECH",  # Expands HIPAA with breach notification and EHR rules
        "GLBA",  # Financial services customer data protection
        "COPPA",  # Privacy for children under 13 online
        "FERPA",  # Student education record privacy
        "ECPA",  # Protection for electronic communications
        # "VPPA"  # Narrow but relevant for streaming/viewing history
    ]

    regulated_df = regulated_df[regulated_df["reg_id"].isin(important_privacy_regulations)]
    logger.info(f"Found {len(regulated_df)} feature-regulation matches")

    return regulated_df

def validate_all_features(regulated_df):
    """
    Validate all feature-regulation pairs using OpenAI.
    """
    logger.info("Starting feature regulation validation...")

    validation_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    reg_status_counts = {}


    logger.info(f"Validating {regulated_df.feature_clean.nunique()} unique features...")


    for _, row in tqdm(regulated_df.iterrows(), total=len(regulated_df),
                      desc="Validating features"):


        reg_status, confidence, rationale, p_tokens, c_tokens = validate_feature_regulation(
            feature_name=row['feature_clean'],
            attribute_class=row['attribute_class'],
            notes=row['notes'],
            quoted_text=row['quoted_text']
        )

        # Count and print distinct reg_status values
        reg_status_counts[reg_status] = reg_status_counts.get(reg_status, 0) + 1
        print(f"\n Current reg_status counts: {dict(sorted(reg_status_counts.items()))}")

        validation_results.append({
            'feature_clean': row['feature_clean'],
            'attribute_class': row['attribute_class'],
            'title': row['title'],
            'abstract': row['abstract'],
            'doi': row['doi'],
            'domain_validated': row['domain_validated'],
            'reg_id': row['reg_id'],
            'article_ref': row['article_ref'],
            'quoted_text': row['quoted_text'],
            # 'keyword': row['keyword'],
            # 'reg_rationale': row['rationale'],
            'regulation_status': reg_status,
            'confidence': confidence,
            'validation_rationale': rationale
        })

        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        time.sleep(RATE_DELAY)

    logger.info(f"Validation complete. Total tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}")

    # Calculate cost
    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Total OpenAI cost: ${total_cost:.4f}")

    return pd.DataFrame(validation_results)


def main():
    """Main execution function."""
    logger.info("Starting feature regulation validation...")

    # Load data
    features_df, clauses_df = load_and_prepare_data()

    # Merge features with regulations
    regulated_df = merge_features_with_regulations(features_df, clauses_df)

    # Validate features using OpenAI
    validation_df = validate_all_features(regulated_df)

    # # Add unregulated features
    # final_df = add_unregulated_features(validation_df, unregulated_df)

    # Save results
    logger.info("Saving results...")
    validation_df.to_csv(OUTPUT_CSV, index=False)
    validation_df.to_json(OUTPUT_JSON, orient='records', indent=2)

    # # Generate summary report
    # generate_summary_report(final_df)

    logger.info(f"✅ Validation complete! Results saved to:")
    logger.info(f"   - {OUTPUT_CSV}")
    logger.info(f"   - {OUTPUT_JSON}")
    logger.info(f"   - {SUMMARY_XLSX}")

if __name__ == "__main__":
    main()