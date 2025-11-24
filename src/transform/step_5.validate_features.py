"""
Feature Validation Module for Decision Tree Publications

Input: data/processed/merged_features.json (feature-annotated corpus).
Outputs: data/processed/validated_features.csv and data/processed/validated_features.json.

This module implements a rigorous validation pipeline for extracted features from decision tree
publications. The validation employs large language model (LLM) inference with strict criteria
to verify that extracted features are explicitly mentioned in publication abstracts as predictor
variables used in decision tree models. This validation step ensures extraction quality and
filters false positives resulting from automated feature extraction processes.

Input Specifications:
    data/processed/merged_features.json: Feature-annotated corpus from extraction pipeline
    with feature names and supporting evidence sentences

Output Specifications:
    data/processed/validated_features.csv: Validation-annotated corpus in CSV format
    data/processed/validated_features.json: Validation-annotated corpus in JSON format

Methodological Approach:
    1. Corpus ingestion from feature extraction pipeline
    2. LLM-based validation with explicit criteria enforcement
    3. Binary classification: 'Valid' for complete feature verification, 'Not valid' otherwise
    4. Token-based cost estimation with comprehensive logging
    5. Rate-limited API interaction for stability
    6. Dual-format persistence for analytical flexibility

Validation Protocol:
    Features are validated against strict criteria requiring explicit mention in abstracts
    as predictor variables in decision tree methodologies. The validation process rejects
    features that are implied, appear only in titles, serve as outcome variables, or are
    mentioned in contexts not related to decision tree models. Decision tree synonyms
    (CART, ID3, C4.5, C5.0, J48, CHAID) are recognized for methodology identification.
"""
from __future__ import annotations

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
INPUT_JSON = PROCESSED_DIR / "merged_features.json"
OUTPUT_CSV = PROCESSED_DIR / "validated_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "validated_features.json"

configure_openai()

PROMPT_TEMPLATE = """
You are a strict validator.

Task
- Validate whether ALL listed features are explicitly mentioned in the ABSTRACT as predictors used in a decision-tree model.

Output
- Return only one of:
  - "Valid"   -> every listed feature is explicitly mentioned in the abstract as a predictor in a decision-tree model.
  - "Not valid" -> otherwise.
- Return exactly one of the above strings. No extra text.

Key rules
1) “Explicitly mentioned” means the feature names appear in the abstract text itself. Case-insensitive match is allowed. Minor inflection/plural is allowed. Vague groups like “demographics” do NOT count unless each listed feature is named.
2) “Predictors in a decision-tree model” requires the abstract to state a decision-tree method was trained/used with those features.
   - Accept synonyms for decision-tree: “decision tree”, “CART”, “ID3”, “C4.5”, “C5.0”, “J48”, “CHAID”.
3) If any listed feature is only implied, only in Title/Keywords, or appears as an OUTCOME/target rather than a predictor, return "Not valid".
4) If any information is missing or unclear, default to "Not valid".

Paper
<<<
Title: {title}

Abstract: {abstract}

Features: {features}
>>>

Examples

[Example 1 — all features present and used with a decision tree -> Valid]
Input:
Title: Predicting Readmission Risk
Abstract: We trained a decision tree to predict readmission using age, prior admissions, and length of stay...
Features: ["age", "prior admissions", "length of stay"]
Output:
"Valid"

[Example 2 — one feature missing -> Not valid]
Input:
Title: Customer Churn Analysis
Abstract: We trained a decision tree using tenure and monthly charges to classify churn...
Features: ["tenure", "monthly charges", "contract type"]
Output:
"Not valid"

[Example 3 — features named but as outcome, not predictors -> Not valid]
Input:
Title: Estimating Age from Voice
Abstract: A decision tree predicts age from acoustic markers...
Features: ["age"]
Output:
"Not valid"

[Example 4 — only tree-based ensemble mentioned, no explicit decision tree -> Not valid]
Input:
Title: Credit Default Prediction
Abstract: We used a random forest with income and age...
Features: ["income", "age"]
Output:
"Not valid"

[Example 5 — decision-tree synonym used (C4.5) with explicit features -> Valid]
Input:
Title: Hypertension Screening
Abstract: A C4.5 decision tree was trained using BMI and systolic blood pressure...
Features: ["BMI", "systolic blood pressure"]
Output:
"Valid"

[Example 6 — vague group label in abstract -> Not valid]
Input:
Title: Loan Approval Models
Abstract: We used a decision tree with demographic factors to predict approval...
Features: ["age", "income"]
Output:
"Not valid"
"""

def validate_features(title: str, abstract: str, features: str) -> tuple[str, int, int]:
    """
    Execute binary validation for extracted decision tree features.
    
    This function implements LLM-based validation with strict criteria to verify that all
    extracted features are explicitly mentioned in the publication abstract as predictor
    variables used in decision tree models. The validation employs zero-temperature sampling
    for deterministic judgments and enforces rigorous requirements including explicit textual
    presence, predictor role verification, and decision tree methodology confirmation.
    
    Parameters
    ----------
    title : str
        Publication title providing contextual information for validation.
    abstract : str
        Publication abstract containing feature mentions and methodology description.
    features : str
        Semicolon-delimited string of extracted feature names requiring validation.
    
    Returns
    -------
    tuple[str, int, int]
        Three-element tuple containing:
        - Validation label: 'Valid', 'Not valid', or 'Error'
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct structured validation prompt with explicit criteria
    2. Submit validation request to Azure OpenAI ChatCompletion endpoint
    3. Apply zero-temperature sampling for deterministic validation
    4. Limit completion tokens to 10 for concise binary output
    5. Extract validation label and token usage from API response
    6. Handle API exceptions with error-state return value
    
    Validation Criteria
    -------------------
    Features are validated as 'Valid' if ALL of the following conditions hold:
        - Every feature is explicitly named in the abstract text
        - Features are described as predictor variables (not outcomes)
        - Abstract mentions decision tree methodology or recognized synonyms
        - No features are only implied or appear exclusively in titles
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
            max_tokens=10
        )

        label = response.choices[0].message.content.strip()
        usage = response.usage
        return label, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    except Exception as e:
        logger.warning(f"API error for '{title[:30]}...': {e}")
        return "Error", 0, 0


def main() -> None:
    """
    Execute corpus-wide feature validation pipeline for extracted features.
    
    This function orchestrates the complete validation workflow including corpus ingestion,
    iterative LLM-based feature validation with strict criteria enforcement, cost estimation,
    and dual-format persistence. The pipeline validates all extracted features to ensure they
    meet rigorous quality standards for explicit mention and predictor role verification.
    Empty feature sets are automatically classified as 'Not valid' without API calls. Rate
    limiting (1-second intervals) ensures API compliance with comprehensive logging.
    
    Returns
    -------
    None
        Results are persisted to OUTPUT_CSV and OUTPUT_JSON with no return value.
    
    Algorithm
    ---------
    1. Corpus Ingestion: Load feature-annotated bibliographic records from JSON
    2. Iterative Validation: Process each record through LLM feature validator
    3. Empty Feature Handling: Classify empty feature sets as 'Not valid' without API call
    4. Token Accumulation: Aggregate prompt and completion token usage
    5. Cost Estimation: Calculate cumulative API costs using pricing constants
    6. Result Annotation: Append validation labels to original records
    7. Dual Persistence: Write validation-annotated corpus to CSV and JSON formats
    8. Summary Logging: Report token counts, estimated costs, and output locations
    
    Output Side Effects
    -------------------
    Writes validation-annotated corpus to PROCESSED_DIR in two formats:
        - validated_features.csv: Tabular CSV format
        - validated_features.json: Structured JSON format
    
    Logs comprehensive progress information including:
        - Corpus dimensions
        - Progress bar for validation iterations
        - Total token consumption statistics
        - Estimated API costs
        - Validation warnings and errors
    """
    df = pd.read_json(str(INPUT_JSON))
    logger.info(f"Loaded {len(df)} rows from {INPUT_JSON}")

    validation_results: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating features"):
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        features = row.get("features", "")
        if features == '':
            label = "Not valid"
            validation_results.append(label)
            continue
        label, prompt_tokens, completion_tokens = validate_features(title, abstract, features)
        validation_results.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        time.sleep(1)

    df["feature_validation"] = validation_results

    df.to_json(str(OUTPUT_JSON), orient="records", indent=2)
    df.to_csv(str(OUTPUT_CSV), index=False)
    logger.info(f"Wrote validated results to {OUTPUT_JSON} and {OUTPUT_CSV}")

    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}, cost=${total_cost:.4f}")


if __name__ == "__main__":
    main()
