#!/usr/bin/env python3
"""
Filter validated_feature_regulation.json to retain only entries with:
  regulation_status == "Regulated" AND confidence == "High"

Input  : PROCESSED_DIR / "validated_feature_regulation.json"
Output : PROCESSED_DIR / "validated_feature_regulation_regulated.json"
"""
from __future__ import annotations

import json
from pathlib import Path
from src.utils.io import PROC
import sys

# Expect PROC to be defined in the runtime (env or imported constant).
try:
    PROC  # type: ignore[name-defined]
except NameError:
    print("ERROR: PROC is not defined in this runtime.", file=sys.stderr)
    sys.exit(1)

PROCESSED_DIR = Path(PROC)  # type: ignore[name-defined]
IN_FILE = PROCESSED_DIR / "validated_feature_regulation.json"
OUT_FILE = PROCESSED_DIR / f"{IN_FILE.stem}_regulated.json"

def main() -> int:
    if not IN_FILE.exists():
        print(f"ERROR: File not found: {IN_FILE}", file=sys.stderr)
        return 1

    try:
        # utf-8-sig tolerates BOM if present
        with IN_FILE.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {IN_FILE}: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print("ERROR: Expected top-level JSON array.", file=sys.stderr)
        return 1

    def norm_str(x):
        return x.strip().casefold() if isinstance(x, str) else x

    filtered = [
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
