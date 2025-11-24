"""
Domain Classification Module for Decision Tree Publications

Input: data/processed/merged_classified_decision_trees_related.json (relevance-labeled corpus).
Outputs: data/processed/merged_domain_validated.json and data/processed/merged_domain_validated.csv.

This module implements a multi-class domain classification pipeline for scholarly publications
previously identified as relevant to decision-tree methodologies. The classification employs
large language model (LLM) inference to assign each publication to one of thirteen predefined
application domains, enabling domain-specific analysis of decision tree research trends and
applications across diverse sectors.

Input Specifications:
    data/processed/merged_classified_decision_trees_related.json: Binary-classified corpus
    with relevance annotations from decision tree classification pipeline

Output Specifications:
    data/processed/merged_domain_validated.json: Domain-annotated corpus in JSON format
    data/processed/merged_domain_validated.csv: Domain-annotated corpus in CSV format

Methodological Approach:
    1. Corpus filtering for decision tree relevant publications
    2. Zero-temperature LLM classification across 13 domain categories
    3. Token-based cost tracking with comprehensive logging
    4. Rate-limited API interaction for stability
    5. Dual-format persistence for analytical flexibility

Domain Taxonomy:
    The classification schema encompasses 13 mutually exclusive domains:
    banking_finance, healthcare_pharma, insurance, ecommerce_retail,
    telecom_network_security, social_media, education_learning_analytics,
    iot_smart_systems, government_public_admin, cybersecurity_intrusion_detection,
    hr_recruitment, transportation_logistics, none_of_the_above
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
    Execute multi-class domain classification for scholarly publication.
    
    This function implements zero-temperature LLM-based classification to assign publications
    to one of thirteen predefined application domains. The classification leverages structured
    prompting with title and abstract metadata to enable domain-specific categorization of
    decision tree research. Token usage is tracked for cost estimation and resource monitoring.
    
    Parameters
    ----------
    title : str
        Primary publication title for semantic domain analysis.
    abstract : str
        Publication abstract providing methodological and application context.
    
    Returns
    -------
    tuple[str, int, int]
        Three-element tuple containing:
        - Domain classification label from predefined taxonomy or 'Error'
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct structured prompt combining title and abstract
    2. Submit classification request to Azure OpenAI ChatCompletion endpoint
    3. Apply zero-temperature sampling for deterministic domain assignment
    4. Limit completion tokens to 10 for concise domain label output
    5. Extract domain label and token usage from API response
    6. Handle API exceptions with error-state return value
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

def main() -> None:
    """
    Execute corpus-wide domain classification pipeline for relevant publications.
    
    This function orchestrates the complete domain classification workflow including corpus
    ingestion, relevance filtering, iterative LLM-based multi-class classification, cost
    estimation, and dual-format persistence. The pipeline processes only publications marked
    as relevant to decision tree methodologies and implements rate limiting (1-second intervals)
    for API compliance with comprehensive logging for production monitoring.
    
    Returns
    -------
    None
        Results are persisted to OUTPUT_JSON and OUTPUT_CSV with no return value.
    
    Algorithm
    ---------
    1. Corpus Ingestion: Load binary-classified bibliographic records from JSON
    2. Relevance Filtering: Select only 'Relevant' decision tree publications
    3. Iterative Classification: Process each filtered record through LLM domain classifier
    4. Token Accumulation: Aggregate prompt and completion token usage
    5. Cost Estimation: Calculate cumulative API costs using pricing constants
    6. Result Annotation: Append domain classification labels to filtered records
    7. Dual Persistence: Write annotated corpus to both JSON and CSV formats
    8. Summary Logging: Report final token counts and estimated costs
    
    Output Side Effects
    -------------------
    Writes domain-annotated corpus to PROCESSED_DIR in two formats:
        - merged_domain_validated.json: Structured JSON format
        - merged_domain_validated.csv: Tabular CSV format
    
    Logs comprehensive progress information including:
        - Progress bar for classification iterations
        - Total token consumption statistics
        - Estimated API costs
        - Classification warnings and errors
    """
    df = pd.read_json(str(INPUT_JSON))
    df = df[df["decision_trees_related"] == "Relevant"]
    labels: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Domain classification"):
        label, prompt_tokens, completion_tokens = classify_paper(
            row.get("title", ""),
            row.get("abstract", "")
        )
        labels.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        time.sleep(1)

    df["domain_validated"] = labels

    df.to_json(str(OUTPUT_JSON), orient="records", indent=2)
    df.to_csv(str(OUTPUT_CSV), index=False)
    logger.info(f"[ok] wrote classified results to {OUTPUT_JSON} and {OUTPUT_CSV}")

    total_cost = (
        (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Total prompt tokens: {total_prompt_tokens}")
    logger.info(f"Total completion tokens: {total_completion_tokens}")
    logger.info(f"Estimated cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main()
