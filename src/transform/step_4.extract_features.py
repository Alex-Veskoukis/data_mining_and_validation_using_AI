"""
Feature Extraction Module for Decision Tree Models

Input: data/processed/merged_domain_validated.json (domain-classified corpus).
Outputs: data/processed/merged_features.csv and data/processed/merged_features.json.

This module implements an automated feature extraction pipeline for scholarly publications
describing decision tree methodologies. The extraction employs Azure OpenAI's function-calling
capabilities to identify explicit predictor variables (features) and their supporting textual
evidence from publication abstracts. This structured extraction enables systematic analysis
of feature selection patterns across domains and applications.

Input Specifications:
    data/processed/merged_domain_validated.json: Domain-classified corpus with binary
    relevance annotations and validated domain assignments

Output Specifications:
    data/processed/merged_features.csv: Feature-annotated corpus in CSV format
    data/processed/merged_features.json: Feature-annotated corpus in JSON format

Methodological Approach:
    1. Corpus filtering for domain-validated publications
    2. LLM-based feature extraction using structured function calling
    3. Evidence sentence extraction for provenance tracking
    4. Validation and error handling for malformed outputs
    5. Token-based cost estimation with comprehensive logging
    6. Rate-limited API interaction for stability

Extraction Protocol:
    Features are identified through explicit mentions in abstracts, with strict requirements
    for textual evidence. Each extracted feature includes a normalized name and the exact
    sentence providing evidence of its usage in the decision tree model. Implicit or inferred
    features are excluded to maintain extraction precision.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

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
INPUT_JSON = PROCESSED_DIR / "merged_domain_validated.json"
OUTPUT_CSV = PROCESSED_DIR / "merged_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "merged_features.json"

configure_openai()

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


def find_features(title: str, abstract: str, venue: str, domain: str) -> tuple[list[dict[str, str]], int, int]:
    """
    Extract decision tree features and evidence sentences from publication abstract.
    
    This function implements LLM-based structured extraction using OpenAI's function-calling
    mechanism to identify explicit predictor variables (features) mentioned in decision tree
    methodologies. Each feature is accompanied by a supporting evidence sentence extracted
    directly from the abstract, ensuring traceability and validation of extraction results.
    The extraction employs zero-temperature sampling for deterministic outputs and includes
    comprehensive error handling for malformed responses.
    
    Parameters
    ----------
    title : str
        Publication title providing context for feature extraction.
    abstract : str
        Publication abstract containing feature descriptions and evidence.
    venue : str
        Publication venue for contextual information.
    domain : str
        Validated application domain for domain-specific feature interpretation.
    
    Returns
    -------
    tuple[list[dict[str, str]], int, int]
        Three-element tuple containing:
        - List of feature dictionaries with 'name' and 'evidence' keys
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct structured prompt with explicit extraction instructions
    2. Submit function-calling request to Azure OpenAI ChatCompletion endpoint
    3. Apply zero-temperature sampling for deterministic feature identification
    4. Parse JSON function call arguments containing feature list
    5. Validate feature structure and evidence presence
    6. Handle malformed outputs with empty list fallback
    7. Log errors and warnings for debugging and quality monitoring
    """
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


def main() -> None:
    """
    Execute corpus-wide feature extraction pipeline for domain-validated publications.
    
    This function orchestrates the complete feature extraction workflow including corpus
    ingestion, domain validation filtering, iterative LLM-based feature extraction with
    evidence collection, cost estimation, and dual-format persistence. The pipeline processes
    only publications where the original domain assignment matches the validated domain,
    ensuring consistency in domain-specific feature analysis. Rate limiting (1-second intervals)
    ensures API compliance with comprehensive logging for production monitoring.
    
    Returns
    -------
    None
        Results are persisted to OUTPUT_CSV and OUTPUT_JSON with no return value.
    
    Algorithm
    ---------
    1. Corpus Ingestion: Load domain-classified bibliographic records from JSON
    2. Dual Filtering: Select 'Relevant' publications with matching domain validation
    3. Iterative Extraction: Process each record through LLM feature extractor
    4. Feature Aggregation: Collect feature names and evidence sentences per publication
    5. Token Accumulation: Aggregate prompt and completion token usage
    6. Cost Estimation: Calculate cumulative API costs using pricing constants
    7. Result Annotation: Append feature lists and evidence to original records
    8. Dual Persistence: Write feature-annotated corpus to CSV and JSON formats
    9. Summary Logging: Report token counts, estimated costs, and output locations
    
    Output Side Effects
    -------------------
    Writes feature-annotated corpus to PROCESSED_DIR in two formats:
        - merged_features.csv: Tabular CSV format with semicolon-delimited features
        - merged_features.json: Structured JSON format with nested feature objects
    
    Logs comprehensive progress information including:
        - Filtered corpus dimensions
        - Progress bar for feature extraction iterations
        - Total token consumption statistics
        - Estimated API costs
        - Extraction warnings and errors
    """
    df = pd.read_json(str(INPUT_JSON))
    df = df[df.get("decision_trees_related") == "Relevant"]
    df = df[df["domain"] == df["domain_validated"]]
    logger.info(f"Remaining shape: {df.shape[0]}, {df.shape[1]}")

    all_names: list[str] = []
    all_evidence: list[str] = []
    debug_raw: list[list[dict[str, str]]] = []
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

    df.to_csv(str(OUTPUT_CSV), index=False)
    df.to_json(str(OUTPUT_JSON), orient="records", indent=2)
    logger.info(f"Wrote outputs to {OUTPUT_CSV} and {OUTPUT_JSON}")

    cost = (total_p / 1000) * PROMPT_PRICE_PER_1000_TOKENS + (total_c / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    logger.info(f"Tokens: prompt={total_p}, completion={total_c}, cost=${cost:.4f}")


if __name__ == "__main__":
    main()
