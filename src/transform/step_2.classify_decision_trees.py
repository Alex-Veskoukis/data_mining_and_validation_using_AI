"""
Decision Tree Relevance Classification Module

Input: data/processed/merged_corpus.json (merged bibliographic corpus).
Output: data/processed/merged_classified_decision_trees_related.json (relevance-labeled corpus).

This module implements an automated classification pipeline for bibliographic records using
large language models (LLMs) to assess relevance to decision-tree-based machine learning
methodologies. The classification employs Azure OpenAI's ChatCompletion API with carefully
engineered prompts to perform binary relevance judgments on scholarly publications.

Input Specifications:
    data/processed/merged_corpus.json: Unified bibliographic corpus from integration pipeline

Output Specifications:
    data/processed/merged_classified_decision_trees_related.json: Annotated corpus with
    binary relevance classifications and associated token usage metadata

Methodological Approach:
    1. Corpus ingestion from unified bibliographic database
    2. Zero-temperature LLM classification for deterministic judgments
    3. Token-based cost estimation with real-time progress monitoring
    4. Rate-limited API interaction for stability and compliance
    5. Comprehensive error handling and logging for production robustness
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
INPUT_JSON = PROCESSED_DIR / "merged_corpus.json"
OUTPUT_JSON = PROCESSED_DIR / "merged_classified_decision_trees_related.json"

configure_openai()

SYSTEM_PROMPT = (
    "You are an expert in privacy-preserving machine learning. "
    "Your task is to read a paper's title, abstract and venue, "
    "and decide whether it presents or applies a decision-tree-based machine learning algorithm. "
    "Respond with exactly one of: 'Relevant' or 'Not relevant', and nothing else."
)

def classify_paper(title: str, abstract: str, venue: str) -> tuple[str, int, int]:
    """
    Execute binary relevance classification for scholarly publication.
    
    This function implements zero-temperature LLM-based classification to determine whether
    a given scholarly publication addresses decision-tree-based machine learning methodologies.
    The classification leverages structured prompting with title, venue, and abstract metadata
    to enable informed relevance judgments. Token usage is tracked for cost estimation and
    resource monitoring purposes.
    
    Parameters
    ----------
    title : str
        Primary publication title for semantic analysis.
    abstract : str
        Publication abstract providing methodological context.
    venue : str
        Publication venue (journal or conference) indicating domain relevance.
    
    Returns
    -------
    tuple[str, int, int]
        Three-element tuple containing:
        - Classification label: 'Relevant', 'Not relevant', or 'Error'
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct structured prompt combining title, venue, and abstract
    2. Submit classification request to Azure OpenAI ChatCompletion endpoint
    3. Apply zero-temperature sampling for deterministic responses
    4. Limit completion tokens to 3 for concise binary output
    5. Extract classification label and token usage from API response
    6. Handle API exceptions with error-state return value
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

def main() -> None:
    """
    Execute corpus-wide decision tree relevance classification pipeline.
    
    This function orchestrates the complete classification workflow including corpus ingestion,
    iterative LLM-based classification with progress monitoring, cost estimation, and persistence
    of annotated results. The pipeline implements rate limiting (1-second intervals) to ensure
    API compliance and includes comprehensive logging for production monitoring and debugging.
    
    Returns
    -------
    None
        Results are persisted to OUTPUT_JSON with no return value.
    
    Algorithm
    ---------
    1. Corpus Ingestion: Load unified bibliographic records from JSON
    2. Iterative Classification: Process each record through LLM classifier
    3. Progress Monitoring: Log interim statistics every 100 records
    4. Token Accumulation: Aggregate prompt and completion token usage
    5. Cost Estimation: Calculate cumulative API costs using pricing constants
    6. Result Annotation: Append classification labels to original records
    7. Persistence: Write annotated corpus to output JSON file
    8. Summary Logging: Report final token counts and estimated costs
    
    Output Side Effects
    -------------------
    Writes annotated corpus to PROCESSED_DIR:
        - merged_classified_decision_trees_related.json: Records with relevance labels
    
    Logs comprehensive progress information including:
        - Interim progress updates every 100 records
        - Token consumption statistics
        - Estimated API costs
        - Classification warnings and errors
    """
    df = pd.read_json(str(INPUT_JSON))
    results: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Classifying"), start=1):
        label, prompt_tokens, completion_tokens = classify_paper(
            row.get("title", ""),
            row.get("abstract", ""),
            row.get("venue", "")
        )
        results.append(label)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        if i % 100 == 0:
            interim_cost = (
                    (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
                    (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
            )
            logger.info(
                f"[progress] processed={i} "
                f"prompt_tokens={total_prompt_tokens} "
                f"completion_tokens={total_completion_tokens} "
                f"estimated_cost=${interim_cost:.4f}"
            )

        time.sleep(1)

    df["decision_trees_related"] = results

    df.to_json(str(OUTPUT_JSON), orient="records", indent=2)
    logger.info(f"[ok] wrote classified results to {OUTPUT_JSON}")

    total_cost = (
            (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
            (total_completion_tokens / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Total prompt tokens: {total_prompt_tokens}")
    logger.info(f"Total completion tokens: {total_completion_tokens}")
    logger.info(f"Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
