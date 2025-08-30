"""
This script merges harvested JSON records from Crossref and OpenAlex into
a single de-duplicated corpus, outputting both CSV and JSON formats.

Inputs:
  - RAW JSON files placed in the directory specified by `utils.io.RAW`
    (e.g. filenames like `crossref_domain.json` or `openalex_domain.json`)

Outputs:
  - data/processed/merged_corpus.csv
  - data/processed/merged_corpus.json
"""

from pathlib import Path
import json
import re
import math
import pandas as pd
import typing as t

from utils.io import RAW, PROC, save_csv, save_json

PROCESSED_DIR = Path(PROC)

def norm_doi(x: str | None) -> str | None:
    """Normalize DOI by removing URL prefix and converting to lowercase."""
    if not x or not isinstance(x, str):
        return None
    x = x.lower().strip()
    return re.sub(r"^https?://doi\\.org/", "", x)

def safe_first(value):
    """Safely extract the first element from a list or return None."""
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def extract_year(rec: dict, keys: list[str]) -> int | None:
    """Extract the year of publication from a list of potential keys."""
    for key in keys:
        part = rec.get(key, {})
        dp = part.get("date-parts") if isinstance(part, dict) else None
        if isinstance(dp, list) and dp and dp[0]:
            return dp[0][0]
    # Fallback: Extract year from DOI or other fields
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

def from_crossref(rec: dict, domain: str) -> dict:
    """Parse a Crossref record into a normalized format."""
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
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()  # Remove HTML tags
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

def from_openalex(rec: dict, domain: str) -> dict:
    """Parse an OpenAlex record into a normalized format."""
    def safe_get(dic: dict | None, key: str):
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

PARSERS: dict[str, t.Callable[[dict, str], dict]] = {
    "crossref": from_crossref,
    "openalex": from_openalex,
}

def merge() -> pd.DataFrame:
    """Merge and deduplicate records from Crossref and OpenAlex."""
    records: list[dict] = []
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

    # Deduplication based on DOI
    df = (
        df.sort_values(["source"])
          .drop_duplicates(subset=["doi"], keep="first")
    )

    # Deduplication based on title and year
    df = (
        df.sort_values(["source"])
          .drop_duplicates(subset=["title_", "year"], keep="first")
          .drop(columns=["title_"])
    )

    # Save the merged and deduplicated data
    save_csv(df, PROCESSED_DIR / "merged_corpus.csv")
    save_json(df.to_dict(orient="records"), PROCESSED_DIR / "merged_corpus.json")
    print(f"[ok] merged {len(records)} raw records into {df.shape[0]} unique rows")
    return df

if __name__ == "__main__":
    merge()