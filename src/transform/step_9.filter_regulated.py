#!/usr/bin/env python3
"""
Filter validated_feature_regulation.json to retain only entries with:
  regulation_status == "Regulated" AND confidence == "High"

Output file is the input name with the suffix "_regulated.json".
Example:
  Input : validated_feature_regulation.json
  Output: validated_feature_regulation_regulated.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="Path to validated_feature_regulation.json (top-level JSON array).",
    )
    args = parser.parse_args()

    in_path = Path(args.input_path).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: File not found: {in_path}", file=sys.stderr)
        return 1

    # Derive output path: <stem>_regulated.json
    out_path = in_path.with_name(f"{in_path.stem}_regulated.json")

    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {in_path}: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print("ERROR: Expected top-level JSON array.", file=sys.stderr)
        return 1

    def norm(x):
        return x.strip() if isinstance(x, str) else x

    filtered = [
        row for row in data
        if isinstance(row, dict)
        and norm(row.get("regulation_status")) == "Regulated"
        and norm(row.get("confidence")) == "High"
    ]

    # Write pretty-printed UTF-8 JSON
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(
        f"Retained {len(filtered)} of {len(data)} records.\n"
        f"Wrote: {out_path}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
