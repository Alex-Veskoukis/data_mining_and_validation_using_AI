"""
Bibliographic Data Integration and Deduplication Module

Inputs: Raw Crossref/OpenAlex JSON in utils.io.RAW matching {source}_{domain}.json.
Outputs: data/processed/merged_corpus.csv and data/processed/merged_corpus.json.

This module implements a comprehensive pipeline for merging and deduplicating scholarly
publication records harvested from heterogeneous bibliographic databases (Crossref and OpenAlex).
The integration process employs a multi-stage deduplication strategy based on Digital Object
Identifiers (DOIs) and exact title-year matching (after case-insensitive normalization) to 
construct a unified corpus for subsequent analytical processing.

Input Specifications:
    Raw JSON files from bibliographic sources located in the directory specified by utils.io.RAW.
    Expected filename patterns: {source}_{domain}.json where source ∈ {crossref, openalex}.

Output Specifications:
    data/processed/merged_corpus.csv: Tabular representation of the unified corpus
    data/processed/merged_corpus.json: Structured JSON representation of the unified corpus

Algorithmic Approach:
    1. Record ingestion and source-specific parsing
    2. DOI-based deduplication with source priority (Crossref preferred over OpenAlex)
    3. Title-year exact matching deduplication on normalized lowercase strings
    4. Persistence in multiple formats for downstream analysis
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import math
import pandas as pd
import typing as t
from src.utils.io import RAW, PROC, save_csv, save_json

PROCESSED_DIR = Path(PROC)

def norm_doi(x: str | None) -> str | None:
    """
    Normalize Digital Object Identifier to canonical form.
    
    This function implements DOI normalization following the International DOI Foundation
    specifications by removing HTTP(S) resolver prefixes and applying case-insensitive
    transformation. This ensures consistent identifier matching across heterogeneous
    bibliographic sources.
    
    Parameters
    ----------
    x : str | None
        Raw DOI string potentially containing URL prefix and mixed case characters.
    
    Returns
    -------
    str | None
        Normalized DOI in lowercase without resolver prefix, or None if input is invalid.
    
    Examples
    --------
    >>> norm_doi("https://doi.org/10.1038/nature12373")
    '10.1038/nature12373'
    >>> norm_doi("10.1038/NATURE12373")
    '10.1038/nature12373'
    """
    if not x or not isinstance(x, str):
        return None
    x = x.lower().strip()
    return re.sub(r"^https?://doi\\.org/", "", x)

def safe_first(value: t.Any) -> t.Any:
    """
    Extract initial element from list structures with null-safety.
    
    This utility function provides safe access to the first element of list-like
    structures while handling edge cases including empty lists, NaN values, and
    non-list types. This pattern is essential for robust data extraction from
    semi-structured bibliographic records.
    
    Parameters
    ----------
    value : Any
        Potentially list-valued object or scalar value.
    
    Returns
    -------
    Any | None
        First element if value is non-empty list, None for NaN or empty structures,
        otherwise returns the value unchanged.
    """
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def extract_year(rec: dict[str, t.Any], keys: list[str]) -> int | None:
    """
    Extract publication year from bibliographic record with fallback mechanisms.
    
    This function implements a hierarchical extraction strategy for publication dates,
    prioritizing structured date fields while employing pattern-matching fallbacks on
    unstructured text fields (DOI strings, abstracts) when primary sources are unavailable.
    This multi-strategy approach maximizes temporal metadata recovery from incomplete records.
    
    Parameters
    ----------
    rec : dict
        Bibliographic record dictionary containing potential date fields.
    keys : list[str]
        Ordered list of candidate date field names to search sequentially.
    
    Returns
    -------
    int | None
        Four-digit publication year if successfully extracted, None otherwise.
    
    Algorithm
    ---------
    1. Iterate through structured date fields specified in keys parameter
    2. Extract year from date-parts arrays (ISO 8601 structured format)
    3. If unsuccessful, apply regex pattern matching on DOI string
    4. If still unsuccessful, apply regex pattern matching on abstract text
    5. Return None if all extraction strategies fail
    """
    for key in keys:
        part = rec.get(key, {})
        dp = part.get("date-parts") if isinstance(part, dict) else None
        if isinstance(dp, list) and dp and dp[0]:
            return dp[0][0]
    doi = rec.get("DOI") or rec.get("doi")
    if doi and isinstance(doi, str):
        match = re.search(r"\b(19|20)\d{2}\b", doi)
        if match:
            return int(match.group(0))
    abstract = rec.get("abstract")
    if abstract and isinstance(abstract, str):
        match = re.search(r"\b(19|20)\d{2}\b", abstract)
        if match:
            return int(match.group(0))
    return None

def from_crossref(rec: dict[str, t.Any], domain: str) -> dict[str, t.Any]:
    """
    Transform Crossref bibliographic record to normalized schema.
    
    This function parses Crossref API responses into a standardized internal representation,
    handling the idiosyncrasies of Crossref's JSON schema including nested author structures,
    date-parts arrays, and HTML-embedded abstracts. The normalization process ensures schema
    consistency across heterogeneous bibliographic sources for downstream analytical tasks.
    
    Parameters
    ----------
    rec : dict
        Raw bibliographic record conforming to Crossref API schema.
    domain : str
        Thematic domain classification label extracted from filename.
    
    Returns
    -------
    dict
        Normalized record with standardized field names and cleaned values including:
        - title: Primary publication title
        - author: Semicolon-delimited author list
        - year: Publication year (integer)
        - venue: Journal or conference name
        - doi: Normalized DOI
        - source: Literal 'crossref'
        - domain: Thematic classification
        - abstract: HTML-stripped abstract text
        - publisher: Publishing entity
        - language: ISO language code
        - type: Publication type
        - url: Canonical URL
        - cited_by: Citation count
    """
    title = safe_first(rec.get("title"))
    year = extract_year(rec, ["published-print", "published", "issued", "created"])
    authors_field = rec.get("author")
    if isinstance(authors_field, list) and authors_field:
        names = [
            f"{a.get('given','')} {a.get('family','')}".strip()
            for a in authors_field
            if isinstance(a, dict)
        ]
        author_str = "; ".join(names) if names else None
    else:
        author_str = None
    venue = safe_first(rec.get("container-title"))
    abstract = rec.get("abstract")
    if isinstance(abstract, str):
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
    return {
        "title":    title,
        "author":   author_str,
        "year":     year,
        "venue":    venue,
        "doi":      norm_doi(rec.get("DOI")),
        "source":   "crossref",
        "domain":   domain,
        "abstract": abstract,
        "publisher": rec.get("publisher"),
        "language": rec.get("language"),
        "type":     rec.get("type"),
        "url":      rec.get("URL"),
        "cited_by": rec.get("is-referenced-by-count"),
    }

def from_openalex(rec: dict[str, t.Any], domain: str) -> dict[str, t.Any]:
    """
    Transform OpenAlex bibliographic record to normalized schema.
    
    This function parses OpenAlex API responses into the standardized internal representation
    used throughout the data processing pipeline. OpenAlex employs a nested structure with
    primary_location, source, and authorships objects that require careful traversal and
    safe dictionary access patterns to handle incomplete records gracefully.
    
    Parameters
    ----------
    rec : dict
        Raw bibliographic record conforming to OpenAlex API schema.
    domain : str
        Thematic domain classification label extracted from filename.
    
    Returns
    -------
    dict
        Normalized record with standardized field names matching the Crossref schema,
        ensuring bidirectional compatibility for merged corpus construction. Field
        mappings account for structural differences between the two API schemas while
        maintaining semantic equivalence.
    """
    def safe_get(dic: dict[str, t.Any] | None, key: str) -> t.Any:
        return dic.get(key) if isinstance(dic, dict) else None

    primary = safe_get(rec, "primary_location")
    source  = safe_get(primary, "source")
    venue   = safe_get(source, "display_name") or safe_get(source, "id")
    abstract = rec.get("abstract")
    if isinstance(abstract, str):
        abstract = abstract.strip()

    authors = []
    for a in rec.get("authorships", []):
        author_obj = safe_get(a, "author")
        if author_obj:
            authors.append(author_obj.get("display_name"))
    author_str = "; ".join(authors) or None

    return {
        "title":    rec.get("display_name") or rec.get("title"),
        "author":   author_str,
        "year":     rec.get("publication_year"),
        "venue":    venue,
        "doi":      norm_doi(rec.get("doi")),
        "source":   "openalex",
        "domain":   domain,
        "abstract": abstract,
        "publisher": safe_get(source, "display_name"),
        "language": rec.get("language"),
        "type":     rec.get("type"),
        "url":      safe_get(primary, "landing_page_url"),
        "cited_by": rec.get("cited_by_count"),
    }

PARSERS: dict[str, t.Callable[[dict[str, t.Any], str], dict[str, t.Any]]] = {
    "crossref": from_crossref,
    "openalex": from_openalex,
}

def merge() -> pd.DataFrame:
    """
    Execute bibliographic corpus integration and deduplication pipeline.
    
    This function orchestrates the complete data integration workflow including file discovery,
    source-specific parsing, record normalization, and multi-strategy deduplication. The
    deduplication algorithm prioritizes DOI-based matching with source preference (Crossref
    over OpenAlex due to publisher provenance), followed by exact title-year matching on
    case-normalized strings to capture records lacking DOI metadata.
    
    Returns
    -------
    pd.DataFrame
        Unified and deduplicated bibliographic corpus with standardized schema. Each row
        represents a unique publication identified through DOI or exact title-year matching.
    
    Raises
    ------
    RuntimeError
        If no valid JSON records are discovered in the RAW directory, indicating potential
        data harvesting failure or incorrect path configuration.
    
    Algorithm
    ---------
    1. File Discovery: Glob pattern matching on RAW directory for {source}_{domain}.json
    2. Source Identification: Filename prefix parsing to route to appropriate parser
    3. Record Normalization: Source-specific transformation to unified schema
    4. DOI Normalization: Case-insensitive DOI canonicalization
    5. Primary Deduplication: DOI-based exact matching with Crossref priority
    6. Secondary Deduplication: Exact title-year matching on lowercase normalized strings
    7. Persistence: Dual-format output (CSV for analysis, JSON for interchange)
    
    Output Side Effects
    -------------------
    Writes two files to PROCESSED_DIR:
        - merged_corpus.csv: Tabular format
        - merged_corpus.json: Structured format
    """
    records: list[dict[str, t.Any]] = []
    for file_path in Path(RAW).glob("*.json"):
        prefix, _, tail = file_path.stem.partition("_")
        domain = tail
        if prefix not in PARSERS:
            print(f"[warn] unknown prefix {prefix}; skipping {file_path.name}")
            continue

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                print(f"[warn] {file_path.name} not list-JSON; skipping")
                continue
        except Exception as e:
            print(f"[warn] could not read {file_path.name}: {e}")
            continue

        parser = PARSERS[prefix]
        for rec in data:
            try:
                norm = parser(rec, domain)
                if norm["title"]:
                    records.append(norm)
            except Exception as e:
                print(f"[warn] parse error in {file_path.name}: {e}")
                continue

    if not records:
        raise RuntimeError("No valid JSON records found in RAW directory")

    df = pd.DataFrame(records)
    df["doi"]    = df["doi"].astype(str).str.lower().str.strip().replace({"none": None, "nan": None})
    df["title_"] = df["title"].str.lower().str.strip()

    df = (
        df.sort_values(["source"])
          .drop_duplicates(subset=["doi"], keep="first")
    )

    df = (
        df.sort_values(["source"])
          .drop_duplicates(subset=["title_", "year"], keep="first")
          .drop(columns=["title_"])
    )

    save_csv(df, str(PROCESSED_DIR / "merged_corpus.csv"))
    save_json(df.to_dict(orient="records"), str(PROCESSED_DIR / "merged_corpus.json"))
    print(f"[ok] merged {len(records)} raw records into {df.shape[0]} unique rows")
    return df

if __name__ == "__main__":
    merge()
