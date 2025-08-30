"""
Extract passages from authoritative legal PDFs that *mention* regulated
personal-data categories, then tag each passage with one or more of 13
privacy-aware classes.

Inputs  (unchanged):
  data/authoritative_legal_texts/correct/*.pdf

Outputs (same filenames as before):
  data/processed/reg_sections.xlsx
      • clauses     sheet – columns [reg_id, article_ref, quoted_text,
                                    attribute_class, rationale]
      • crosswalk   sheet – columns [attribute_class, first_example]
  data/processed/reg_sections_clauses.csv
  data/processed/reg_sections_crosswalk.csv
  data/processed/reg_sections_clauses.json
  data/processed/reg_sections_crosswalk.json
  audit/llm_calls.jsonl            (trace of every GPT call)
"""

import json
import time
import re
import logging
from pathlib import Path

import pdfplumber
import pandas as pd
from tqdm.auto import tqdm
import openai
from openai.error import InvalidRequestError, OpenAIError

from utils.io import PROC
from utils.openai_settings import (
    configure_openai,
    OPENAI_DEPLOYMENT,
    PROMPT_PRICE_PER_1000_TOKENS,
    COMPLETION_PRICE_PER_1000_TOKENS,
)

# ─────────────────────────── configuration ────────────────────────────
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

# New system prompt – answers only the two questions requested
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

# ────────────────────────── OpenAI setup ──────────────────────────
configure_openai()

# ─────────────────────── small helper functions ───────────────────
def pages(pdf: Path):
    with pdfplumber.open(pdf) as doc:
        for p in doc.pages:
            yield p.extract_text() or ""

def segments(text: str):
    parts, ref = HEAD_RE.split(text), None
    for part in parts:
        if HEAD_RE.fullmatch(part):
            ref = part.strip().replace("\n", " ")
        else:
            snip = part.strip().replace("\n", " ")
            if ref and snip:
                yield ref, snip[:MAX_SNIP]

from openai.error import (
    APIError, APIConnectionError, RateLimitError, Timeout,
    InvalidRequestError, OpenAIError,
)
import requests

def call_gpt(ref: str, snippet: str,
             max_retries: int = 6,
             base_delay: 2.0 = 2.0):
    """
    Robust wrapper around one ChatCompletion call.
    • Retries on transient OpenAI/network problems.
    • Returns (data_dict, prompt_tokens, completion_tokens)
      even when it finally gives up (tokens = 0,0).
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            resp = openai.ChatCompletion.create(
                deployment_id=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"REF: {ref}\nTEXT:\n{snippet}"},
                ],
                temperature=TEMP,
                max_tokens=MAXTOK,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            data["_raw"] = content   # keep for audit
            return data, resp.usage.get("prompt_tokens", 0), resp.usage.get("completion_tokens", 0)

        except InvalidRequestError as e:
            # permanent problem (e.g. too long); skip passage
            logger.warning(f"[invalid] {ref}: {e}")
            return {"regulated": False, "_raw": "INVALID_REQUEST"}, 0, 0

        except (APIError, APIConnectionError, RateLimitError, Timeout,
                requests.exceptions.ConnectionError, OpenAIError) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"[abort] {ref}: exceeded retries ({e})")
                return {"regulated": False, "_raw": "ERROR_RETRIES_EXCEEDED"}, 0, 0

            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"[retry {attempt}/{max_retries}] {ref}: {e} "
                           f"– waiting {delay:.1f}s")
            time.sleep(delay)

        except Exception as e:
            # Anything unexpected → skip but record it
            logger.exception(f"[unexpected] {ref}: {e}")
            return {"regulated": False, "_raw": "UNEXPECTED_EXCEPTION"}, 0, 0

def extract_regulation_name(pdf_path: Path) -> str:
    fname = pdf_path.name.removesuffix(".pdf")
    for key in REG_ID:
        if key in fname:
            return REG_ID[key]
    base = fname.split("(")[0].strip()
    return REG_ID.get(base, base)

# ───────────────────────────── main ──────────────────────────────
def main():
    clauses, first_example = [], {}
    prompt_tok = completion_tok = 0

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {PDF_DIR}")
    reg_status_counts = {}
    # --- Use tqdm for progress bars, no pre-counting ---
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

                    # store first example for each class
                    for c in cls_list:
                        first_example.setdefault(c, snip)

                    time.sleep(RATE_DELAY)

    # ----------- dataframe and files -----------
    clauses_df = pd.DataFrame(clauses).drop_duplicates()

    cross_df = (
        pd.DataFrame(
            [(c, txt) for c, txt in sorted(first_example.items())],
            columns=["attribute_class", "first_example"]
        )
        .sort_values("attribute_class")
        .reset_index(drop=True)
    )

    clauses_df.to_csv(OUT_CLAUSES_CSV, index=False)
    cross_df.to_csv(OUT_CROSS_CSV, index=False)
    clauses_df.to_json(OUT_CLAUSES_JSON, orient="records", indent=2)
    cross_df.to_json(OUT_CROSS_JSON, orient="records", indent=2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
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
