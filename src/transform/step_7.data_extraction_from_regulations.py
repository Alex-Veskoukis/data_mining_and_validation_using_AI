"""
Regulatory Text Extraction and Privacy Classification Module

Input: data/authoritative_legal_texts/correct/*.pdf (authoritative legal documents).
Outputs: data/processed/reg_sections.xlsx, *_clauses.csv/json, *_crosswalk.csv/json, audit/llm_calls.jsonl.

This module implements an automated pipeline for extracting and classifying privacy-relevant
passages from authoritative legal documents. The system employs PDF text extraction, intelligent
segmentation based on legal citation patterns, and large language model (LLM) inference to
identify regulated personal data categories. Each extracted passage is mapped to privacy-aware
attribute classes aligned with global data protection frameworks, enabling cross-regulatory
compliance analysis and privacy impact assessment.

Input Specifications:
    data/authoritative_legal_texts/correct/*.pdf: Authoritative legal documents in PDF format
    including GDPR, CCPA, HIPAA, PIPEDA, and other privacy regulations

Output Specifications:
    data/processed/reg_sections.xlsx: Excel workbook with two sheets
        - clauses: Regulatory passages with privacy classifications
        - crosswalk: Privacy class examples from regulations
    data/processed/reg_sections_clauses.csv: Classified passages in CSV format
    data/processed/reg_sections_crosswalk.csv: Privacy class crosswalk in CSV format
    data/processed/reg_sections_clauses.json: Classified passages in JSON format
    data/processed/reg_sections_crosswalk.json: Privacy class crosswalk in JSON format
    audit/llm_calls.jsonl: Complete trace log of all LLM API interactions

Methodological Approach:
    1. PDF ingestion with text extraction from authoritative legal sources
    2. Intelligent segmentation using legal citation pattern recognition
    3. LLM-based binary classification for regulatory relevance
    4. Multi-label privacy class assignment aligned with 13-class taxonomy
    5. Robust error handling with exponential backoff retry logic
    6. Token-based cost estimation with comprehensive audit logging
    7. Multi-format persistence for analytical flexibility

Privacy Taxonomy Alignment:
    The classification employs a 13-class privacy taxonomy aligned with GDPR, CCPA, HIPAA,
    and related frameworks: Identifier_PII, Contact_Info, Device_OnlineID, Biometric,
    Location_IoT, Health_Clinical, Financial, Child_Data, Demographic, Behavioural,
    Environmental, Operational_Business, Other.
"""
from __future__ import annotations

import json
import logging
import re
import time
import typing as t
from pathlib import Path

import openai
import openpyxl
import pandas as pd
import pdfplumber
import requests
from openai.error import (
    APIConnectionError,
    APIError,
    InvalidRequestError,
    OpenAIError,
    RateLimitError,
    Timeout,
)
from tqdm.auto import tqdm

from src.utils.io import PROC
from src.utils.openai_settings import (
    COMPLETION_PRICE_PER_1000_TOKENS,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    configure_openai,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR   = Path(PROC)
PDF_DIR         = PROCESSED_DIR.parent / "authoritative_legal_texts" / "correct"

OUT_XLSX        = PROCESSED_DIR / "reg_sections.xlsx"
OUT_CLAUSES_CSV = PROCESSED_DIR / "reg_sections_clauses.csv"
OUT_CROSS_CSV   = PROCESSED_DIR / "reg_sections_crosswalk.csv"
OUT_CLAUSES_JSON= PROCESSED_DIR / "reg_sections_clauses.json"
OUT_CROSS_JSON  = PROCESSED_DIR / "reg_sections_crosswalk.json"

AUDIT_LOG       = PROCESSED_DIR / "audit" / "llm_calls.jsonl"
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

RATE_DELAY  = 1.0
MAXTOK      = 120
TEMP        = 0.0
MAX_SNIP    = 450

ATTR_CLASSES = [
    "Identifier_PII", "Contact_Info", "Device_OnlineID", "Biometric",
    "Location_IoT", "Health_Clinical", "Financial", "Child_Data",
    "Demographic", "Behavioural", "Environmental",
    "Operational_Business", "Other",
]

# Mapping identical to your original – left intact
REG_ID = {
    "Australia Privacy Act 1988": "Australia Privacy Act",
    "COPPA": "COPPA",
    "California CPRA": "CPRA",
    "Canada PIPEDA": "PIPEDA",
    "China PIPL": "PIPL",
    "Consumer–General(California CCPA § 1798.140(v)(1))": "CCPA",
    "E-commerce_Retail_and_Security_OWASP-MSP_OWASP_Top_Ten_2021": "OWASP Top Ten",
    "EU Digital Markets Act (DMA)": "DMA",
    "EU Digital Services Act (DSA)": "DSA",
    "EU NIS2 Directive (Network and Information Security)": "NIS2",
    "EU eHealth Network Guidelines": "EU eHealth Network",
    "FERPA": "FERPA",
    "General Data Protection Regulation (2017)": "GDPR",
    "Healthcare_GDPR_Art9(1)": "GDPR",
    "Healthcare_HIPAA_§164.514": "HIPAA",
    "Healthcare_HITECH_Act(42 U.S.C. Ch. 156)": "HITECH",
    "India DPDP Act (Digital Personal Data Protection Act 2023)": "DPDP",
    "Insurance_NAIC_Model_Privacy_Act_MDL-672": "NAIC",
    "Japan APPI (Act on the Protection of Personal Information)": "APPI",
    "NIST SP 800-53 – Security and privacy controls.": "NIST SP 800-53",
    "New York SHIELD Act": "SHIELD",
    "PCI DSS (Payment Card Industry Data Security Standard)": "PCI DSS",
    "PSD2 (EU Payment Services Directive 2)": "PSD2",
    "Russia Federal Law on Personal Data": "Russia Personal Data Law",
    "SOX (Sarbanes-Oxley Act)": "SOX",
    "Singapore PDPA (Personal Data Protection Act 2012)": "PDPA",
    "South Africa POPIA (Protection of Personal Information Act)": "POPIA",
    "Telecommunications_and_Network_Security_ECPA(18 U.S.C. Ch. 119, §§ 2510–2523)": "ECPA",
    "Telecommunications_and_Network_Security_ePrivacy_Directive_2002:58:EC(Articles 5 & 6)": "ePrivacy Directive",
    "UK Data Protection Act (2018)": "UK DPA",
    "US 42 CFR Part 2": "42 CFR Part 2",
    "US CAN-SPAM Act": "CAN-SPAM",
    "US Genetic Information Nondiscrimination Act (GINA)": "GINA",
    "VPPA (Video Privacy Protection Act)": "VPPA",
    "banking_and_finance_FCRA_§1681": "FCRA",
    "banking_and_finance_GLBA_§6809": "GLBA",
    "banking_and_finance_bcbs239": "BCBS239",
}

SYSTEM_PROMPT = (
    "You are a legal-compliance analyst. Decide whether the law fragment "
    "mentions that any data element is regulated. If yes, list the exact matching "
    "privacy class from the list below (Other corresponds to anything else that does not match)."
    " Respond *only* with JSON:\n"
    '{\n'
    '  "regulated": true|false,\n'
    '  "classes":   [at least one class name from the attribute classes],\n'
    '  "rationale": "<≤15 words>"\n'
    '}\n'
    "Allowed class names: " + ", ".join(ATTR_CLASSES) + "."
)

HEAD_RE = re.compile(
    r"(Article\s+\d+[A-Za-z]?\b|ART\.\s*\d+|§\s*\d[\dA-Za-z\.\(\)]*|^\([A-Za-z]\)\s+|^•)",
    re.I | re.M,
)

configure_openai()


def pages(pdf: Path):
    """
    Extract text content from all pages of PDF document.
    
    This generator function implements page-by-page text extraction from PDF documents
    using the pdfplumber library. Empty pages yield empty strings to maintain page
    index correspondence for debugging and provenance tracking.
    
    Parameters
    ----------
    pdf : Path
        Absolute path to PDF file containing legal document.
    
    Yields
    ------
    str
        Extracted text content from each page, or empty string for pages without text.
    """
    with pdfplumber.open(pdf) as doc:
        for p in doc.pages:
            yield p.extract_text() or ""

def segments(text: str):
    """
    Segment page text into legal citation and content pairs.
    
    This generator function implements intelligent text segmentation based on legal citation
    patterns (Article numbers, section symbols, enumerated clauses). The segmentation employs
    regex pattern matching to identify structural boundaries in legal documents, yielding
    citation-content pairs suitable for downstream classification. Content snippets are
    truncated to MAX_SNIP characters to ensure API compliance and cost efficiency.
    
    Parameters
    ----------
    text : str
        Extracted page text containing legal provisions.
    
    Yields
    ------
    tuple[str, str]
        Two-element tuple containing:
        - Citation reference (Article, section, clause identifier)
        - Content snippet (truncated to MAX_SNIP characters, newlines normalized)
    
    Algorithm
    ---------
    1. Split text using legal citation regex pattern (HEAD_RE)
    2. Iterate through split components
    3. Identify citation references via regex full-match
    4. Associate subsequent text with most recent citation
    5. Normalize whitespace and truncate content
    6. Yield citation-content pairs when both components present
    """
    parts, ref = HEAD_RE.split(text), None
    for part in parts:
        if HEAD_RE.fullmatch(part):
            ref = part.strip().replace("\n", " ")
        else:
            snip = part.strip().replace("\n", " ")
            if ref and snip:
                yield ref, snip[:MAX_SNIP]


def _parse_model_json(raw: str) -> dict[str, t.Any]:
    """
    Parse JSON response from LLM with robust error handling.
    
    This function implements fault-tolerant JSON parsing for LLM responses that may contain
    code fences, extraneous prose, or multiple JSON objects. The parser attempts multiple
    strategies including code fence stripping and balanced brace extraction to recover valid
    JSON objects from malformed responses, maximizing extraction success rates.
    
    Parameters
    ----------
    raw : str
        Raw LLM response potentially containing JSON with formatting artifacts.
    
    Returns
    -------
    dict[str, t.Any]
        Parsed JSON object as Python dictionary.
    
    Raises
    ------
    ValueError
        If no valid JSON object can be extracted from the response.
    
    Algorithm
    ---------
    1. Validate input for empty/None values
    2. Strip markdown code fences if present
    3. Attempt direct JSON parsing (fast path)
    4. Scan for first balanced curly-brace pair
    5. Extract and parse candidate JSON substring
    6. Raise ValueError if all strategies fail
    """
    if raw is None:
        raise ValueError("empty content")
    s = raw.strip()
    if not s:
        raise ValueError("empty content")

    # Strip code fences if present
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1).strip()

    # Fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try first balanced {...}
    depth, start = 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    cand = s[start : i + 1]
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        start = None  # continue scanning

    raise ValueError("no valid JSON object found")


def call_gpt(ref: str, snippet: str,
             max_retries: int = 6,
             base_delay: float = 2.0) -> tuple[dict[str, t.Any], int, int]:
    """
    Execute robust LLM classification with exponential backoff retry logic.
    
    This function implements a fault-tolerant wrapper around Azure OpenAI ChatCompletion API
    calls with comprehensive error handling for transient failures. The retry mechanism employs
    exponential backoff for rate limiting and connection errors, while gracefully degrading
    for parse failures and invalid requests. All responses are logged for audit compliance.
    
    Parameters
    ----------
    ref : str
        Legal citation reference for the text snippet.
    snippet : str
        Regulatory text content requiring classification.
    max_retries : int, default=6
        Maximum number of retry attempts for transient errors.
    base_delay : float, default=2.0
        Base delay in seconds for exponential backoff calculation.
    
    Returns
    -------
    tuple[dict[str, t.Any], int, int]
        Three-element tuple containing:
        - Classification result dictionary with 'regulated', 'classes', 'rationale' keys
        - Prompt tokens consumed in API request
        - Completion tokens generated in API response
    
    Algorithm
    ---------
    1. Construct classification request with system and user prompts
    2. Submit to Azure OpenAI ChatCompletion endpoint with JSON mode
    3. Parse structured JSON response using robust parser
    4. Handle parse failures gracefully with error payload
    5. Retry transient errors (rate limits, connection issues) with exponential backoff
    6. Return error payload for invalid requests or exhausted retries
    7. Log warnings for all error conditions
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            kwargs = dict(
                deployment_id=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"REF: {ref}\nTEXT:\n{snippet}"},
                ],
                temperature=TEMP,
                max_tokens=MAXTOK,
            )
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = openai.ChatCompletion.create(**kwargs)
            content = (resp.choices[0].message.get("content") or "").strip()

            try:
                data = _parse_model_json(content)
            except Exception as e:
                logger.warning(f"[parse-fail] {ref}: {e}")
                data = {"regulated": False, "classes": [], "rationale": "",
                        "_raw": content, "_parse_error": str(e)}
                return (data,
                        int(resp.usage.get("prompt_tokens", 0) or 0),
                        int(resp.usage.get("completion_tokens", 0) or 0))

            data["_raw"] = content
            return (data,
                    int(resp.usage.get("prompt_tokens", 0) or 0),
                    int(resp.usage.get("completion_tokens", 0) or 0))

        except InvalidRequestError as e:
            logger.warning(f"[invalid] {ref}: {e}")
            return {"regulated": False, "classes": [], "rationale": "", "_raw": "INVALID_REQUEST"}, 0, 0

        except (APIError, APIConnectionError, RateLimitError, Timeout,
                requests.exceptions.ConnectionError, OpenAIError) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"[abort] {ref}: exceeded retries ({e})")
                return {"regulated": False, "classes": [], "rationale": "", "_raw": "ERROR_RETRIES_EXCEEDED"}, 0, 0
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"[retry {attempt}/{max_retries}] {ref}: {e} – waiting {delay:.1f}s")
            time.sleep(delay)

        except Exception as e:
            logger.exception(f"[unexpected] {ref}: {e}")
            return {"regulated": False, "classes": [], "rationale": "", "_raw": "UNEXPECTED_EXCEPTION"}, 0, 0


def extract_regulation_name(pdf_path: Path) -> str:
    """
    Extract standardized regulation identifier from PDF filename.
    
    This function implements filename-based regulation identification using a predefined
    mapping dictionary (REG_ID). The extraction supports both exact matches and partial
    matches with fallback to simplified base names, ensuring consistent regulation
    identifiers across heterogeneous filename conventions.
    
    Parameters
    ----------
    pdf_path : Path
        Path to PDF file with filename encoding regulation identity.
    
    Returns
    -------
    str
        Standardized regulation identifier from REG_ID mapping or simplified base name.
    
    Algorithm
    ---------
    1. Remove .pdf extension from filename
    2. Attempt exact match against REG_ID keys
    3. Attempt partial match (substring search) against REG_ID keys
    4. Extract base name before parentheses as fallback
    5. Return mapped identifier or base name
    """
    fname = pdf_path.name.removesuffix(".pdf")
    for key in REG_ID:
        if key in fname:
            return REG_ID[key]
    base = fname.split("(")[0].strip()
    return REG_ID.get(base, base)


def main() -> None:
    """
    Execute corpus-wide regulatory text extraction and privacy classification pipeline.
    
    This function orchestrates the complete regulatory analysis workflow including PDF
    ingestion, intelligent text segmentation, iterative LLM-based privacy classification,
    comprehensive audit logging, and multi-format persistence. The pipeline processes all
    PDF files in the authoritative legal texts directory, extracting privacy-relevant
    passages and mapping them to a 13-class privacy taxonomy. Progress tracking employs
    nested tqdm progress bars for PDFs, pages, and segments. Rate limiting (1.0-second
    intervals) ensures API compliance.
    
    Returns
    -------
    None
        Results are persisted to multiple output files with no return value.
    
    Algorithm
    ---------
    1. PDF Discovery: Glob pattern matching on authoritative legal texts directory
    2. Iterative Processing: Nested loops over PDFs, pages, and segments
    3. Text Extraction: Page-by-page PDF text extraction with pdfplumber
    4. Intelligent Segmentation: Citation-based text splitting for legal provisions
    5. LLM Classification: Binary relevance and multi-label privacy class assignment
    6. Audit Logging: JSONL trace of all API interactions for compliance
    7. Token Accumulation: Aggregate prompt and completion token usage
    8. First Example Collection: Track first occurrence of each privacy class
    9. Deduplication: Remove duplicate passages from final dataset
    10. Multi-Format Persistence: Excel, CSV, and JSON outputs
    11. Cost Estimation: Calculate cumulative API costs
    12. Summary Logging: Report passage counts, token usage, and costs
    
    Output Side Effects
    -------------------
    Writes regulatory analysis results to PROCESSED_DIR in multiple formats:
        - reg_sections.xlsx: Excel workbook with clauses and crosswalk sheets
        - reg_sections_clauses.csv: Classified passages in CSV format
        - reg_sections_crosswalk.csv: Privacy class examples in CSV format
        - reg_sections_clauses.json: Classified passages in JSON format
        - reg_sections_crosswalk.json: Privacy class examples in JSON format
        - audit/llm_calls.jsonl: Complete API interaction trace log
    
    Logs comprehensive progress information including:
        - Nested progress bars for hierarchical processing
        - Real-time regulation status counts
        - Total token consumption statistics
        - Estimated API costs
        - Classification warnings and errors
    """
    clauses: list[dict[str, t.Any]] = []
    first_example: dict[str, str] = {}
    prompt_tok = completion_tok = 0

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {PDF_DIR}")
    reg_status_counts: dict[t.Any, int] = {}
    with AUDIT_LOG.open("w", encoding="utf-8") as audit:
        for pdf in tqdm(pdfs, desc="PDFs"):
            reg = extract_regulation_name(pdf)
            for pg in tqdm(pages(pdf), desc=pdf.name, leave=False):
                for ref, snip in tqdm(list(segments(pg)), desc="Segments", leave=False):
                    res, p_tok, c_tok = call_gpt(ref, snip)
                    reg_status = res.get("regulated")
                    reg_status_counts[reg_status] = reg_status_counts.get(reg_status, 0) + 1
                    print(f"Current reg_status counts: {dict(sorted(reg_status_counts.items()))}")
                    audit.write(json.dumps({
                        "file": pdf.name, "reg_id": reg,
                        "ref": ref, "snippet": snip[:200],
                        "gpt": res.get("_raw", "")
                    }, ensure_ascii=False) + "\n")

                    prompt_tok     += p_tok
                    completion_tok += c_tok

                    if not res.get("regulated"):
                        time.sleep(RATE_DELAY)
                        continue

                    cls_list = [c for c in res["classes"] if c in ATTR_CLASSES]
                    rationale = res.get("rationale", "")
                    clauses.append({
                        "regulated":res.get("regulated"),
                        "reg_id": reg,
                        "article_ref": ref,
                        "quoted_text": snip,
                        "attribute_class": ";".join(cls_list),
                        "rationale": rationale,
                    })

                    for c in cls_list:
                        first_example.setdefault(c, snip)

                    time.sleep(RATE_DELAY)

    clauses_df = pd.DataFrame(clauses).drop_duplicates()

    cross_df = (
        pd.DataFrame(
            [(c, txt) for c, txt in sorted(first_example.items())],
            columns=["attribute_class", "first_example"]
        )
        .sort_values("attribute_class")
        .reset_index(drop=True)
    )

    clauses_df.to_csv(str(OUT_CLAUSES_CSV), index=False)
    cross_df.to_csv(str(OUT_CROSS_CSV), index=False)
    clauses_df.to_json(str(OUT_CLAUSES_JSON), orient="records", indent=2)
    cross_df.to_json(str(OUT_CROSS_JSON), orient="records", indent=2)

    with pd.ExcelWriter(str(OUT_XLSX), engine="openpyxl") as xl:
        clauses_df.to_excel(xl, sheet_name="clauses", index=False)
        cross_df.to_excel(xl, sheet_name="crosswalk", index=False)

    total_cost = (
        (prompt_tok     / 1000) * PROMPT_PRICE_PER_1000_TOKENS +
        (completion_tok / 1000) * COMPLETION_PRICE_PER_1000_TOKENS
    )
    logger.info(f"Tokens: prompt={prompt_tok}, completion={completion_tok}, cost=${total_cost:.4f}")
    print(f"[ok] wrote {len(clauses_df)} passages to {OUT_XLSX}")

if __name__ == "__main__":
    main()
