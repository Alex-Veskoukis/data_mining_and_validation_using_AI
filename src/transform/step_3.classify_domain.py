"""
Assign each Relevant paper to one of 13 predefined domains using Azure OpenAI ChatCompletion.

Inputs:
  - data/processed/merged_classified_decision_trees_related.json

Outputs:
  - data/processed/merged_domain_validated.json
  - data/processed/merged_domain_validated.csv
"""

import time
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI settings
configure_openai()

PROCESSED_DIR = Path(PROC)
INPUT_JSON = PROCESSED_DIR / "merged_classified_decision_trees_related.json"
OUTPUT_JSON = PROCESSED_DIR / "merged_domain_validated.json"
OUTPUT_CSV = PROCESSED_DIR / "merged_domain_validated.csv"

SYSTEM_PROMPT = (
    "You are an expert in classifying papers by domain. "
    "Read a paper’s title, abstract, keywords, and venue, and choose exactly one of these 13 domains:\n"
    "1. banking_finance\n"
    "2. healthcare_pharma\n"
    "3. insurance\n"
    "4. ecommerce_retail\n"
    "5. telecom_network_security\n"
    "6. social_media\n"
    "7. education_learning_analytics\n"
    "8. iot_smart_systems\n"
    "9. government_public_admin\n"
    "10. cybersecurity_intrusion_detection\n"
    "11. hr_recruitment\n"
    "12. transportation_logistics\n"
    "13. none_of_the_above\n"
    "Respond with exactly one domain string from the list above, and nothing else."
)

def classify_paper(title: str, abstract: str) -> tuple[str, int, int]:
    """
    Classify a paper as belonging to one of the predefined domains.
    Returns the classification label and token usage.
    """
    user_content = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract or 'N/A'}\n\n"
    )
    try:
        resp = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=10
        )
        label = resp.choices[0].message.content.strip()
        usage = resp.usage
        return label, usage["prompt_tokens"], usage["completion_tokens"]
    except Exception as e:
        logger.warning(f"API error for '{title[:30]}...': {e}")
        return "Error", 0, 0

def main():
    # Read input JSON
    df = pd.read_json(INPUT_JSON)
    df = df[df["decision_trees_related"] == "Relevant"]
    labels = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Classify each paper
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Domain classification"):
        label, prompt_tokens, completion_tokens = classify_paper(
            row.get("title", ""),
            row.get("abstract", "")
        )
        labels.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        time.sleep(1)  # Avoid hitting rate limits

    # Add classification results to the DataFrame
    df["domain_validated"] = labels

    # Save the results to JSON and CSV files
    df.to_json(OUTPUT_JSON, orient="records", indent=2)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"[ok] wrote classified results to {OUTPUT_JSON} and {OUTPUT_CSV}")

    # Calculate and log the estimated cost
    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Total prompt tokens: {total_prompt_tokens}")
    logger.info(f"Total completion tokens: {total_completion_tokens}")
    logger.info(f"Estimated cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main()