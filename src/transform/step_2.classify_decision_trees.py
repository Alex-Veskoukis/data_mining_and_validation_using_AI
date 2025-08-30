"""
Classify each paper in the merged keywords corpus as ‘Relevant’ or ‘Not relevant’
to decision‐tree–based machine learning using Azure OpenAI ChatCompletion.
This module helps refine the downloaded papers relevancy.

Inputs:
  - data/processed/merged_keywords.json

Outputs:
  - data/processed/merged_classified_decision_trees_related.json
"""

import time
import logging
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

from utils.io import PROC
from utils.openai_settings import (
    configure_openai,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    COMPLETION_PRICE_PER_1000_TOKENS)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(PROC)
INPUT_JSON = PROCESSED_DIR / "merged_corpus.json"
OUTPUT_JSON = PROCESSED_DIR / "merged_classified_decision_trees_related.json"

# Configure OpenAI settings
configure_openai()


SYSTEM_PROMPT = (
    "You are an expert in privacy-preserving machine learning. "
    "Your task is to read a paper’s title, abstract and venue, "
    "and decide whether it presents or applies a decision-tree-based machine learning algorithm. "
    "Respond with exactly one of: 'Relevant' or 'Not relevant', and nothing else."
)

def classify_paper(title: str, abstract: str, venue: str) -> tuple[str, int, int]:
    """
    Classify a paper as 'Relevant' or 'Not relevant' using OpenAI ChatCompletion.
    Returns the classification label and token usage.
    """
    user_content = (
        f"Title: {title}\n\n"
        f"Venue: {venue}\n\n"
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
            max_tokens=3
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
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Classify each paper
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        label, prompt_tokens, completion_tokens = classify_paper(
            row.get("title", ""),
            row.get("abstract", ""),
            row.get("venue", "")
        )
        results.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        time.sleep(1)  # Avoid hitting rate limits

    # Add classification results to the DataFrame
    df["decision_trees_related"] = results

    # Save the results to a JSON file
    df.to_json(OUTPUT_JSON, orient="records", indent=2)
    logger.info(f"[ok] wrote classified results to {OUTPUT_JSON}")

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