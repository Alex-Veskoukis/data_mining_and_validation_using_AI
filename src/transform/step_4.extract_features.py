# src/transform/extract_features.py
"""
Extract driver features and corresponding evidence from papers classified as relevant,
using Azure OpenAI ChatCompletion with function-calling.

Inputs:
  - data/processed/merged_domain_validated.json

Outputs:
  - data/processed/merged_features.csv
  - data/processed/merged_features.json
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
INPUT_JSON = PROCESSED_DIR / "merged_domain_validated.json"
OUTPUT_CSV = PROCESSED_DIR / "merged_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "merged_features.json"

# Configure OpenAI
configure_openai()

# Original instruction prompt template
PROMPT_TEMPLATE = """
You are building a feature table for decision-tree models. For the paper below, do the following:

1. Identify each **explicit feature (predictor or attribute)** used in the decision-tree described in the following abstract. 
   - A feature is a variable or attribute that is explicitly mentioned in the abstract as being used in the decision-tree model.
   - Do not infer or assume features that are not explicitly stated in the abstract.

2. For each feature, locate the **one full sentence** in the abstract that contains the feature name exactly (case-insensitive substring match). 
   - The sentence must explicitly mention the feature in the context of the decision-tree model.
   - Do not include multiple sentences or paragraphs as evidence. Only return the single sentence that mentions the feature.

3. IMPORTANT: If no features (predictors or attributes) are explicitly mentioned in the abstract, return an empty list.

Return only this JSON object (no extra text):

{{
  "features": [
    {{
      "name": "<short feature label—for example, “Age”>",
      "evidence": "<the full quoted sentence from the abstract>"
    }}
  ]
}}

Paper:
<<<
Title: {title}

Venue: {venue}

Abstract: {abstract}

Domain: {domain}
>>>
"""

# Define function schema
FEATURES_FUNCTION = {
    "name": "extract_features",
    "description": "Return features and evidence sentences for a paper.",
    "parameters": {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "evidence": {"type": "string"}
                    },
                    "required": ["name", "evidence"]
                }
            }
        },
        "required": ["features"]
    }
}


def find_features(title: str, abstract: str, venue: str, domain: str):
    prompt = PROMPT_TEMPLATE.format(
        title=title or "N/A",
        venue=venue or "N/A",
        abstract=abstract or "N/A",
        domain=domain or "N/A"
    )
    messages = [
        {"role": "system", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=messages,
            functions=[FEATURES_FUNCTION],
            function_call={"name": "extract_features"},
            temperature=0.0,
            max_tokens=1500
        )

        usage = response.usage
        func_call = response.choices[0].message.get("function_call", {})
        args = func_call.get("arguments", "{}")

        try:
            parsed = json.loads(args)
            features = parsed.get("features", [])
            if not isinstance(features, list):
                raise ValueError("Invalid 'features' format")

            # Validate each feature object
            for feature in features:
                if "evidence" not in feature:
                    logger.warning(f"Missing 'evidence' for feature: {feature}")
                    feature["evidence"] = "No evidence provided"

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse function output for '{title[:30]}...': {args}, error: {e}")
            features = []

        return features, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    except Exception as e:
        logger.warning(f"API error for '{title[:30]}...': {e}")
        return [], 0, 0


def main():
    df = pd.read_json(INPUT_JSON)
    df = df[df.get("decision_trees_related") == "Relevant"]
    df = df[df["domain"] == df["domain_validated"]]
    logger.info(f"Remaining shape: {df.shape[0]}, {df.shape[1]}")

    all_names, all_evidence, debug_raw = [], [], []
    total_p, total_c = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        venue = row.get("venue", "")
        domain = row.get("domain", "")

        features, p_tok, c_tok = find_features(title, abstract, venue, domain)
        total_p += p_tok
        total_c += c_tok

        names = [f.get("name", "N/A") for f in features]
        evs = [f.get("evidence", "N/A") for f in features]
        all_names.append("; ".join(names))
        all_evidence.append("; ".join(evs))
        debug_raw.append(features)

        time.sleep(1)

    df["features"] = all_names
    df["evidence"] = all_evidence
    df["debug"] = debug_raw

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_json(OUTPUT_JSON, orient="records", indent=2)
    logger.info(f"Wrote outputs to {OUTPUT_CSV} and {OUTPUT_JSON}")

    cost = (total_p / 1000) * PROMPT_PRICE_PER_1000_TOKENS + (total_c / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    logger.info(f"Tokens: prompt={total_p}, completion={total_c}, cost=${cost:.4f}")


if __name__ == "__main__":
    main()
