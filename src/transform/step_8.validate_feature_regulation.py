"""
Feature-regulation validation module using large language model inference.

Inputs: data/processed/attribute_classes.json and data/processed/reg_sections_clauses.json.
Outputs: data/processed/validated_feature_regulation.csv/json and feature_regulation_validation_summary.xlsx.

This module validates whether extracted features are explicitly regulated by matching
privacy regulations based on attribute class alignment. The validation employs 
zero-temperature LLM inference to determine regulatory coverage through semantic 
analysis of regulatory text quotations.

Input Specifications
--------------------
attribute_classes.json : JSON
    Feature records with assigned privacy attribute classifications and domain metadata
reg_sections_clauses.json : JSON
    Regulatory clause quotations indexed by privacy attribute class and regulation identifier

Output Specifications
---------------------
validated_feature_regulation.csv : CSV
    Feature-regulation validation results with regulation status and confidence levels
validated_feature_regulation.json : JSON
    Structured validation records including rationale and article references
feature_regulation_validation_summary.xlsx : Excel
    Aggregated validation statistics by regulation and attribute class

Methodological Approach
------------------------
1. Load feature records with attribute class assignments from prior classification
2. Load regulatory clause quotations mapped to privacy attribute classes
3. Merge features with regulations through attribute class join operations
4. Filter to privacy-critical regulations (GDPR, CCPA, HIPAA, etc.)
5. For each feature-regulation pair, construct validation prompt with regulatory context
6. Invoke zero-temperature LLM to classify: Regulated, Not Regulated, or Not Clearly Regulated
7. Extract confidence level (High, Medium, Low) and rationale with article citations
8. Aggregate token usage and compute validation cost metrics
9. Export validated records with regulation status and supporting evidence
"""

from __future__ import annotations

import json
import logging
import time
import typing as t
from collections import OrderedDict
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
FEATURES_FILE = PROCESSED_DIR / "attribute_classes.json"
CLAUSES_FILE = PROCESSED_DIR / "reg_sections_clauses.json"
logger.info(PROCESSED_DIR)

OUTPUT_CSV = PROCESSED_DIR / "validated_feature_regulation.csv"
OUTPUT_JSON = PROCESSED_DIR / "validated_feature_regulation.json"
SUMMARY_XLSX = PROCESSED_DIR / "feature_regulation_validation_summary.xlsx"

RATE_DELAY = 1.0


def _read_json_records(path: Path) -> pd.DataFrame:
    """
    Parse JSON file containing record array into DataFrame structure.

    Parameters
    ----------
    path : Path
        Absolute path to JSON file containing either top-level array or object with list value

    Returns
    -------
    pd.DataFrame
        Parsed records as DataFrame with columns matching JSON object keys

    Raises
    ------
    FileNotFoundError
        If specified path does not exist in filesystem
    ValueError
        If JSON structure is neither top-level array nor object containing list value

    Algorithm
    ---------
    1. Open JSON file with UTF-8-BOM encoding support
    2. Parse JSON content into Python object
    3. If object is list, convert directly to DataFrame
    4. If object is dict, search for common container keys (data, items, records, rows)
    5. If no standard key found, use first list-like value in dictionary
    6. Raise error if no valid array structure identified
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return pd.DataFrame(obj)

    if isinstance(obj, dict):
        for key in ("data", "items", "records", "rows"):
            if key in obj and isinstance(obj[key], list):
                return pd.DataFrame(obj[key])
        for v in obj.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return pd.DataFrame(v)

    raise ValueError(f"Unsupported JSON shape in {path}. Expected top-level array of objects.")


def validate_feature_regulation(
    feature_name: str,
    attribute_class: str,
    notes: str,
    quoted_text: dict[str, list[str]] | str,
) -> tuple[str, str, str, int, int]:
    """
    Validate regulatory coverage of feature using LLM semantic analysis of regulatory quotations.

    Parameters
    ----------
    feature_name : str
        Normalized feature identifier from validated feature extraction
    attribute_class : str
        Privacy attribute class assignment (e.g., Biometric, Health_Clinical, Location)
    notes : str
        Contextual metadata about feature usage or definition
    quoted_text : dict[str, list[str]] or str
        Regulatory text quotations indexed by article reference or raw text string

    Returns
    -------
    tuple[str, str, str, int, int]
        regulation_status : Regulated, Not Regulated, or Not Clearly Regulated
        confidence : High, Medium, or Low
        rationale : Brief explanation with article references (≤40 words)
        prompt_tokens : Token count for input prompt
        completion_tokens : Token count for model response

    Algorithm
    ---------
    1. Format quoted_text into article-indexed quotation structure
    2. Construct system prompt defining validation decision criteria
    3. Build user prompt with feature, attribute class, notes, and regulatory context
    4. Invoke zero-temperature LLM (gpt-4) for deterministic classification
    5. Parse response into STATUS, CONFIDENCE, RATIONALE fields
    6. Normalize status to {Regulated, Not Regulated} and confidence to {High, Medium, Low}
    7. Extract token usage statistics from API response
    8. Return classification tuple with token metrics
    """
    def _format_context(qt: dict[str, list[str]] | str) -> str:
        if isinstance(qt, dict):
            lines = []
            for ref, quotes in qt.items():
                if not quotes:
                    continue
                lines.append(f"ArticleRef: {ref}")
                for q in quotes:
                    q = str(q).strip()
                    if q:
                        lines.append(f"- {q}")
            return "\n".join(lines) if lines else ""
        return str(qt)

    regulatory_context = _format_context(quoted_text)

    system_prompt = f"""You are a legal analyst.
Decide if ONE feature is regulated by the PROVIDED regulatory text only.
Do not use outside knowledge. Ignore titles, keywords, or abstracts not included here.

Decision target
- Is the feature "{feature_name}" regulated EITHER by explicit mention of:
  a) the exact feature name, OR
  b) its whole attribute class "{attribute_class}" (or unambiguous legal synonyms of that class),
  within the quoted regulatory text?

Output format (exactly three lines, no extra text)
STATUS: Regulated | Not Regulated
CONFIDENCE: High | Medium | Low
RATIONALE: ≤40 words. State decisive phrase(s). If Regulated, END with the exact refs that regulate it: <comma-separated ArticleRef(s)>. If Not Regulated, END with refs: none

Rules
1) “Regulated” when the text clearly covers the exact feature OR clearly covers the whole class it belongs to.
   - Coverage includes prohibitions, restrictions, consent requirements, safeguards, or processing conditions.
   - If the text lists the class by name (e.g., biometric data) the feature in that class counts as regulated.
2) “Not Regulated” when the text does NOT clearly mention the feature or its class.
   - Generic principles (lawful basis, transparency, security) without feature/class do NOT suffice.
   - If the text EXPLICITLY excludes the feature/class, return “Not Regulated”.
3) Confidence:
   - High: exact feature OR explicit class term appears (or an unambiguous legal synonym) with clear coverage.
   - Medium: close paraphrase or category term strongly implies coverage, but wording is less direct.
   - Low: ambiguous wording, weak implication, or conflicting clauses.
4) Ties go to “Not Regulated”. If unsure, choose “Not Regulated” with Low confidence.
5) Use only the quoted regulatory text. Notes are context, not authority.

Examples

Example 1 — Regulated (exact class term present)
STATUS: Regulated
CONFIDENCE: High
RATIONALE: Text covers “biometric data” and restricts processing; fingerprint pattern is biometric. refs: GDPR Art.9(1)

Example 2 — Not Regulated (no feature/class)
STATUS: Not Regulated
CONFIDENCE: High
RATIONALE: Record confidentiality only; no hearing metrics or Health_Clinical class. refs: none

Example 3 — Regulated (explicit feature)
STATUS: Regulated
CONFIDENCE: High
RATIONALE: “IP address” listed as personal data under processing limits. refs: GDPR Recital 30

Example 4 — Not Regulated (ambiguous/vague)
STATUS: Not Regulated
CONFIDENCE: Low
RATIONALE: Only generic ‘personal information’; no clear link to {feature_name} or {attribute_class}. refs: none
"""

    user_prompt = f"""FEATURE TO VALIDATE: {feature_name}
FEATURE ATTRIBUTE CLASS: {attribute_class}
Notes: {notes}

REGULATORY CONTEXT (use ONLY this text):
{regulatory_context}

QUESTION: Is the feature "{feature_name}" regulated either specifically or via its whole class "{attribute_class}" according to this regulatory context?
"""

    try:
        response = openai.ChatCompletion.create(
            deployment_id=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", {}) or {}

        # Parse response strictly into three fields
        lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        # Find first matching lines
        status = "Not Clearly Regulated"
        confidence = "Low"
        rationale = "Unable to parse OpenAI response"

        for ln in lines:
            if ln.upper().startswith("STATUS:"):
                status = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("CONFIDENCE:"):
                confidence = ln.split(":", 1)[1].strip()
            elif ln.upper().startswith("RATIONALE:"):
                rationale = ln.split(":", 1)[1].strip()

        # Normalize status and confidence to allowed set
        s_norm = status.strip().lower()
        if "regulat" in s_norm and "not" not in s_norm:
            status = "Regulated"
        elif "not regulat" in s_norm:
            status = "Not Regulated"
        else:
            # fallback stays as parsed text if already exact, else default
            status = status if status in {"Regulated", "Not Regulated"} else "Not Regulated"

        c_norm = confidence.strip().capitalize()
        confidence = c_norm if c_norm in {"High", "Medium", "Low"} else "Low"

        return (
            status,
            confidence,
            rationale,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    except Exception as e:
        logger.error(f"OpenAI API error for {feature_name}: {e}")
        return ("Not Clearly Regulated", "Low", f"API error: {str(e)}", 0, 0)


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature and regulatory clause datasets from JSON format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        features_df : DataFrame with feature records and attribute class assignments
        clauses_df : DataFrame with regulatory clause quotations indexed by attribute class

    Algorithm
    ---------
    1. Parse attribute_classes.json into DataFrame using flexible JSON reader
    2. Parse reg_sections_clauses.json into DataFrame using flexible JSON reader
    3. Log record counts for validation
    4. Return both DataFrames for subsequent merge operations
    """
    logger.info("Loading data files...")

    features_df = _read_json_records(FEATURES_FILE)
    logger.info(f"Loaded {len(features_df)} features with attribute classes")

    clauses_df = _read_json_records(CLAUSES_FILE)
    logger.info(f"Loaded {len(clauses_df)} regulatory clauses")

    return features_df, clauses_df


group_cols = [
    "feature_clean",
    "title",
    "abstract",
    "doi",
    "domain_validated",
    "attribute_class",
    "notes",
    "regulated",
    "reg_id",
]


def _concat_unique_semicolon(series: pd.Series) -> str:
    """
    Concatenate unique non-null strings preserving first occurrence order.

    Parameters
    ----------
    series : pd.Series
        Pandas Series potentially containing duplicate or null values

    Returns
    -------
    str
        Semicolon-delimited string of unique values in first-occurrence order

    Algorithm
    ---------
    1. Initialize set for seen values and list for ordered results
    2. Iterate through series elements
    3. Skip null values (pd.NA, None, NaN)
    4. Convert to string and check if already seen
    5. If unique, add to both seen set and ordered list
    6. Join ordered list with semicolon delimiter
    """
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


def _map_article_to_quotes(
    article_ref: pd.Series, quoted_text: pd.Series
) -> dict[str, list[str]]:
    """
    Aggregate regulatory quotations by article reference with deduplication.

    Parameters
    ----------
    article_ref : pd.Series
        Series of legal article identifiers (e.g., GDPR Art.9(1))
    quoted_text : pd.Series
        Series of corresponding regulatory text quotations

    Returns
    -------
    dict[str, list[str]]
        Mapping from article reference to deduplicated list of quotations in order

    Algorithm
    ---------
    1. Initialize OrderedDict to preserve insertion order
    2. Iterate through paired article references and quotations
    3. Skip rows where either value is null
    4. Convert both values to string format
    5. If article reference not yet in dict, initialize empty list
    6. Append quotation only if not already present for that article
    7. Convert OrderedDict to standard dict and return
    """
    od: OrderedDict[str, list[str]] = OrderedDict()
    for a, q in zip(article_ref, quoted_text):
        if pd.isna(a) or pd.isna(q):
            continue
        a_str = str(a)
        q_str = str(q)
        if a_str not in od:
            od[a_str] = []
        if q_str not in od[a_str]:
            od[a_str].append(q_str)
    return dict(od)


def merge_features_with_regulations(
    features_df: pd.DataFrame, clauses_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join feature records with regulatory clauses through attribute class matching.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature records with attribute_class column for join operation
    clauses_df : pd.DataFrame
        Regulatory clause records with attribute_class column (semicolon-delimited)

    Returns
    -------
    pd.DataFrame
        Merged records with feature metadata, regulation identifiers, and quotations

    Algorithm
    ---------
    1. Parse semicolon-delimited attribute_class values in clauses_df into lists
    2. Explode attribute_class lists to create one row per class value
    3. Filter out generic Other attribute class to focus on specific classes
    4. Perform inner join on attribute_class between features and clauses
    5. Group by feature metadata columns to aggregate multiple clause matches
    6. Concatenate article references with semicolon delimiter preserving uniqueness
    7. Map article references to deduplicated quotation lists
    8. Filter to rows with non-null quoted_text (actual regulatory matches)
    9. Filter to privacy-critical regulations (GDPR, CCPA, HIPAA, etc.)
    10. Log final feature-regulation pair count
    """
    logger.info("Merging features with regulatory clauses...")

    clauses_df["attribute_class"] = clauses_df["attribute_class"].astype(str)
    clauses_df["attribute_class"] = (
        clauses_df["attribute_class"]
        .str.split(";")
        .apply(lambda lst: [s.strip() for s in lst])
    )

    clauses_df = clauses_df.explode("attribute_class", ignore_index=True)
    clauses_df = clauses_df[clauses_df["attribute_class"] != "Other"]

    merged_df = features_df.merge(clauses_df, on="attribute_class", how="inner")

    result_df = (
        merged_df.groupby(group_cols, dropna=False, as_index=False)[
            ["article_ref", "quoted_text"]
        ]
        .apply(
            lambda g: pd.Series(
                {
                    "article_ref": _concat_unique_semicolon(g["article_ref"]),
                    "quoted_text": _map_article_to_quotes(
                        g["article_ref"], g["quoted_text"]
                    ),
                }
            )
        )
        .reset_index()
    )
    logger.info(f"Created {len(result_df)} feature-regulation pairs")

    regulated_df = result_df.dropna(subset=["quoted_text"])

    important_privacy_regulations = [
        "GDPR",
        "ePrivacy Directive",
        "NIS2",
        "PSD2",
        "EU eHealth Network",
        "CCPA",
        "CPRA",
        "HIPAA",
        "HITECH",
        "GLBA",
        "COPPA",
        "FERPA",
        "ECPA",
    ]

    regulated_df = regulated_df[regulated_df["reg_id"].isin(important_privacy_regulations)]
    logger.info(f"Found {len(regulated_df)} feature-regulation matches")

    return regulated_df


def validate_all_features(regulated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute LLM-based validation for all feature-regulation pairs with cost tracking.

    Parameters
    ----------
    regulated_df : pd.DataFrame
        Merged feature-regulation records with quotations for validation

    Returns
    -------
    pd.DataFrame
        Validation results with regulation_status, confidence, and rationale columns

    Algorithm
    ---------
    1. Initialize validation results list and token counters
    2. Iterate through each feature-regulation pair with progress bar
    3. For each pair, invoke validate_feature_regulation with feature name, attribute class, notes, and quoted text
    4. Parse returned regulation status, confidence, and rationale
    5. Accumulate prompt and completion token counts
    6. Construct result record with feature metadata and validation fields
    7. Apply rate limiting delay (1.0s) between API calls
    8. Log total token usage and compute validation cost
    9. Convert results list to DataFrame and return
    """
    logger.info("Starting feature regulation validation...")

    validation_results: list[dict[str, t.Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    reg_status_counts: dict[str, int] = {}

    logger.info(f"Validating {regulated_df.feature_clean.nunique()} unique features...")

    for _, row in tqdm(
        regulated_df.iterrows(), total=len(regulated_df), desc="Validating features"
    ):
        reg_status, confidence, rationale, p_tokens, c_tokens = validate_feature_regulation(
            feature_name=row["feature_clean"],
            attribute_class=row["attribute_class"],
            notes=row["notes"],
            quoted_text=row["quoted_text"],
        )

        reg_status_counts[reg_status] = reg_status_counts.get(reg_status, 0) + 1
        print(f"\n Current reg_status counts: {dict(sorted(reg_status_counts.items()))}")

        validation_results.append(
            {
                "feature_clean": row["feature_clean"],
                "attribute_class": row["attribute_class"],
                "title": row["title"],
                "abstract": row["abstract"],
                "doi": row["doi"],
                "domain_validated": row["domain_validated"],
                "reg_id": row["reg_id"],
                "article_ref": row["article_ref"],
                "quoted_text": row["quoted_text"],
                "regulation_status": reg_status,
                "confidence": confidence,
                "validation_rationale": rationale,
            }
        )

        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        time.sleep(RATE_DELAY)

    logger.info(
        f"Validation complete. Total tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}"
    )

    total_cost = (total_prompt_tokens / 1000) * PROMPT_PRICE_PER_1000_TOKENS + (
        total_completion_tokens / 1000
    ) * COMPLETION_PRICE_PER_1000_TOKENS
    logger.info(f"Total OpenAI cost: ${total_cost:.4f}")

    return pd.DataFrame(validation_results)


def main() -> None:
    """
    Execute complete feature-regulation validation pipeline with multi-format output.

    Algorithm
    ---------
    1. Load feature records with attribute class assignments from attribute_classes.json
    2. Load regulatory clause quotations indexed by attribute class from reg_sections_clauses.json
    3. Merge features with regulatory clauses through attribute class join
    4. Filter to privacy-critical regulations and features with quotation matches
    5. Validate each feature-regulation pair using zero-temperature LLM inference
    6. Aggregate validation results with regulation status, confidence, and rationale
    7. Save validated records to CSV format for tabular analysis
    8. Save validated records to JSON format for structured consumption
    9. Log completion status and output file paths
    """
    logger.info("Starting feature regulation validation...")

    features_df, clauses_df = load_and_prepare_data()
    regulated_df = merge_features_with_regulations(features_df, clauses_df)
    validation_df = validate_all_features(regulated_df)

    logger.info("Saving results...")
    validation_df.to_csv(str(OUTPUT_CSV), index=False)
    validation_df.to_json(str(OUTPUT_JSON), orient="records", indent=2)

    logger.info("✅ Validation complete! Results saved to:")
    logger.info(f"   - {OUTPUT_CSV}")
    logger.info(f"   - {OUTPUT_JSON}")
    logger.info(f"   - {SUMMARY_XLSX}")


if __name__ == "__main__":
    main()
