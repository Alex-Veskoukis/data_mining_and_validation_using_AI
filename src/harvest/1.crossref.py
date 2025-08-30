# src/harvest/crossref.py
"""
Harvest Crossref metadata with retry and exponential back-off, saving raw JSON and CSV.

Inputs:
  - configuration loaded via utils.io.load_yaml(), mapping domains to:
      • "query" (str) or "crossref_queries" (list of str)
      • "max_records" (int)

Outputs (under project/data/raw/):
  - crossref_<domain>.csv
  - crossref_<domain>.json
"""

import time
import random
import json
from pathlib import Path

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.io import load_yaml

MAILTO  = "alexios.veskouki@ac.eap.gr"
ROOT    = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API     = "https://api.crossref.org/works"
SESSION = None

def build_session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    policy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
        connect=retries,
        read=retries,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=policy))
    session.headers.update({"User-Agent": f"dt-privacy-scan (+mailto:{MAILTO})"})
    return session

def harvest(domain: str, query: str, n: int, page: int = 100) -> pd.DataFrame:
    session = build_session()
    params = {
        "query":    query,
        "rows":     page,
        "cursor":   "*",
        "mailto":   MAILTO,
        "filter":   "has-abstract:true",
    }
    items = []
    while len(items) < n:
        try:
            resp = session.get(API, params=params, timeout=90)
            resp.raise_for_status()
        except requests.exceptions.ReadTimeout:
            print(f"[warn] {domain}: ReadTimeout, retrying after 10s")
            time.sleep(10)
            continue

        msg = resp.json()["message"]
        items.extend(msg.get("items", []))
        next_cursor = msg.get("next-cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        time.sleep(random.uniform(0.8, 1.5))

    collected = items[:n]
    print(f"[ok] {domain}: collected {len(collected)} items")
    return pd.json_normalize(collected)

def save(df: pd.DataFrame, domain: str) -> None:
    csv_path  = RAW_DIR / f"crossref_{domain}.csv"
    json_path = RAW_DIR / f"crossref_{domain}.json"
    df.to_csv(csv_path, index=False)
    json_text = json.dumps(df.to_dict(orient="records"), indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    print(f"[save] wrote {csv_path.name} and {json_path.name}")

def main():
    cfg = load_yaml()
    for domain, meta in cfg.items():
        queries = meta.get("crossref_queries") or [meta.get("query", "")]
        all_dfs = []
        for q in queries:
            print(f"[info] {domain}: running Crossref query → {q!r}")
            df_q = harvest(domain, q, meta.get("max_records", 0))
            all_dfs.append(df_q)
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.drop_duplicates(subset="DOI", keep="first")
            save(df, domain)
        else:
            print(f"[warn] {domain}: no Crossref data collected")

if __name__ == "__main__":
    main()
