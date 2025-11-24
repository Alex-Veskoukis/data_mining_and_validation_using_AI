#!/usr/bin/env python3
"""
High-confidence regulated feature filtering module for validated regulation records.

Input: data/processed/validated_feature_regulation.json (full validation results).
Output: data/processed/validated_feature_regulation_regulated.json (high-confidence regulated subset).

This module filters feature-regulation validation results to retain only entries with
high-confidence positive regulatory determination. The filtering enforces strict criteria
requiring both explicit regulated status and high confidence level from LLM validation.

Input Specifications
--------------------
validated_feature_regulation.json : JSON
    Complete validation results with regulation_status and confidence fields from step_8

Output Specifications
---------------------
validated_feature_regulation_regulated.json : JSON
    Filtered records containing only features with regulation_status='Regulated' and confidence='High'

Methodological Approach
------------------------
1. Load validated feature-regulation records from JSON array format
2. Apply case-insensitive normalization to regulation_status and confidence fields
3. Filter records matching exact criteria: regulation_status='Regulated' AND confidence='High'
4. Export filtered subset preserving original record structure
5. Log filtering statistics showing retention ratio
"""
from __future__ import annotations

import json
import sys
import typing as t
from pathlib import Path

from src.utils.io import PROC

try:
    PROC
except NameError:
    print("ERROR: PROC is not defined in this runtime.", file=sys.stderr)
    sys.exit(1)

PROCESSED_DIR = Path(PROC)
IN_FILE = PROCESSED_DIR / "validated_feature_regulation.json"
OUT_FILE = PROCESSED_DIR / f"{IN_FILE.stem}_regulated.json"


def norm_str(x: t.Any) -> str | t.Any:
    """
    Normalize string values to lowercase format for case-insensitive comparison.

    Parameters
    ----------
    x : t.Any
        Value to normalize, typically string but accepts any type

    Returns
    -------
    str or t.Any
        Stripped and lowercased string if input is string, otherwise original value

    Algorithm
    ---------
    1. Check if input is string type
    2. If string, strip whitespace and convert to lowercase using casefold
    3. If not string, return original value unchanged
    """
    return x.strip().casefold() if isinstance(x, str) else x


def main() -> int:
    """
    Execute filtering pipeline to retain high-confidence regulated features.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error conditions

    Algorithm
    ---------
    1. Verify input file exists in processed data directory
    2. Load JSON array from validated_feature_regulation.json with UTF-8-BOM support
    3. Validate JSON structure is top-level array of record objects
    4. Filter records where normalized regulation_status equals 'regulated'
    5. Further filter where normalized confidence equals 'high'
    6. Write filtered records to validated_feature_regulation_regulated.json
    7. Log retention statistics (filtered count vs. total count)
    8. Return exit code 0 on success, 1 on file or JSON errors
    """
    if not IN_FILE.exists():
        print(f"ERROR: File not found: {IN_FILE}", file=sys.stderr)
        return 1

    try:
        with IN_FILE.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {IN_FILE}: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print("ERROR: Expected top-level JSON array.", file=sys.stderr)
        return 1

    filtered: list[dict[str, t.Any]] = [
        row for row in data
        if isinstance(row, dict)
        and norm_str(row.get("regulation_status")) == "regulated"
        and norm_str(row.get("confidence")) == "high"
    ]

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Retained {len(filtered)} of {len(data)} records.\nWrote: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
