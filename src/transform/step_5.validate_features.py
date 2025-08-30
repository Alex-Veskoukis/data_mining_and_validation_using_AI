"""
Validate features extracted from papers classified as relevant.

Inputs:
  - data/processed/merged_features.json

Outputs:
  - data/processed/validated_features.csv
  - data/processed/validated_features.json
"""

import time
import logging
import json
from pathlib import Path

import pandas as pd
import openai
from tqdm import tqdm

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
INPUT_JSON = PROCESSED_DIR / "merged_features.json"
OUTPUT_CSV = PROCESSED_DIR / "validated_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "validated_features.json"

# Configure OpenAI
configure_openai()

# Validation prompt template
PROMPT_TEMPLATE = """
You are validating features extracted from a paper. For the paper below, validate that the listed features are explicitly mentioned in the abstract as features used as predictors in a decision-tree model.

Return only one of the following responses:
- "Valid" if all the listed features are explicitly mentioned in the abstract as predictors in the decision-tree model.
- "Not valid" if any of the listed features are not explicitly mentioned in the abstract as predictors in the decision-tree model.

Paper:
<<<
Title: {title}

Abstract: {abstract}

Features: {features}
>>>
"""

def validate_features(title: str, abstract: str, features: str) -> tuple[str, int, int]:
    """
    Validate that the features are explicitly mentioned in the abstract as predictors in a decision-tree model.
    """
    prompt = PROMPT_TEMPLATE.format(
        title=title or "N/A",
        abstract=abstract or "N/A",
        features=features or "N/A"
    )
    messages = [
        {"role": "system", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            max_tokens=10  # Small response: "Valid" or "Not valid"
        )

        label = response.choices[0].message.content.strip()
        usage = response.usage
        return label, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    except Exception as e:
        logger.warning(f"API error for '{title[:30]}...': {e}")
        return "Error", 0, 0


def main():
    # Read input JSON
    df = pd.read_json(INPUT_JSON)
    logger.info(f"Loaded {len(df)} rows from {INPUT_JSON}")

    validation_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Validate each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating features"):
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        features = row.get("features", "")

        label, prompt_tokens, completion_tokens = validate_features(title, abstract, features)
        validation_results.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        time.sleep(1)  # Avoid hitting rate limits

    # Add validation results to the DataFrame
    df["feature_validation"] = validation_results

    # Save the results to JSON and CSV files
    df.to_json(OUTPUT_JSON, orient="records", indent=2)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Wrote validated results to {OUTPUT_JSON} and {OUTPUT_CSV}")

    # Calculate and log the estimated cost
    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}, cost=${total_cost:.4f}")


if __name__ == "__main__":
    main()