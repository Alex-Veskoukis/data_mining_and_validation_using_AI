# src/harvest/openalex.py

"""
Harvest metadata from OpenAlex, reconstruct abstracts, and save as JSON and CSV.

Inputs:
  - Configuration via utils.io.load_yaml(), providing per-domain:
      • openalex_queries: list of str or {search, filter} dicts
      • max_records: int

Outputs (written to CWD):
  - openalex_<domain>.json
  - openalex_<domain>.csv
"""
from __future__ import annotations

import math
import time
import random
from pathlib import Path
from typing import Union, Dict, List

import requests
import pandas as pd

from utils.io import load_yaml, save_json, save_csv

MAILTO         = "alexios.veskouki@ac.eap.gr"
OPENALEX_API   = "https://api.openalex.org/works"
HEADERS        = {"User-Agent": f"dt-privacy-scan (+mailto:{MAILTO})"}
REQ_INTERVAL   = 60 / 200  # 200 requests per minute
Query          = Union[str, Dict[str, str]]

def inv_index_to_text(idx: Dict[str, List[int]]) -> str:
    if not isinstance(idx, dict):
        return ""
    max_pos = max((pos for positions in idx.values() for pos in positions), default=-1)
    tokens = [""] * (max_pos + 1)
    for word, positions in idx.items():
        for p in positions:
            if 0 <= p <= max_pos:
                tokens[p] = word
    return " ".join(tokens).strip()

def _fetch_page(params: dict) -> dict:
    resp = requests.get(OPENALEX_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _pull_openalex(search: str, filter_: str | None, remaining: int) -> List[dict]:
    cursor = "*"
    collected = []
    while remaining > 0 and cursor:
        per_page = min(200, remaining)
        params = {"search": search, "per_page": per_page, "cursor": cursor, "filter": "has-abstract:true"}
        if filter_:
            params["filter"] = filter_
        blob = _fetch_page(params)
        results = blob.get("results", [])
        collected.extend(results)
        cursor = blob.get("meta", {}).get("next_cursor")
        remaining -= len(results)
        time.sleep(REQ_INTERVAL)
    return collected

def harvest(domain: str, queries: List[Query], max_records: int) -> None:
    total_needed = max_records
    dedup: Dict[str, dict] = {}
    remaining_queries = len(queries)

    for idx, q in enumerate(queries, start=1):
        if total_needed <= 0:
            break
        if isinstance(q, str):
            search_term, filter_term = q, None
            display_q = q
        else:
            search_term = q.get("search", "")
            filter_term = q.get("filter")
            display_q = f'search="{search_term}"' + (f', filter="{filter_term}"' if filter_term else "")
        per_q = math.ceil(total_needed / remaining_queries)
        print(f"[info] {domain}: fetching up to {per_q} for {display_q}")
        items = _pull_openalex(search_term, filter_term, per_q)
        for it in items:
            dedup.setdefault(it.get("id"), it)
        total_needed = max_records - len(dedup)
        remaining_queries = len(queries) - idx
        time.sleep(REQ_INTERVAL)

    all_items = list(dedup.values())[:max_records]
    for it in all_items:
        inv = it.pop("abstract_inverted_index", None)
        it["abstract"] = inv_index_to_text(inv)

    df = pd.json_normalize(all_items)
    save_json(all_items, f"openalex_{domain}.json")
    save_csv(df,       f"openalex_{domain}.csv")
    print(f"[ok] {domain}: saved {len(all_items)} records")

if __name__ == "__main__":
    cfg = load_yaml()
    for domain, meta in cfg.items():
        harvest(domain, meta.get("openalex_queries", []), meta.get("max_records", 0))
        time.sleep(5)
