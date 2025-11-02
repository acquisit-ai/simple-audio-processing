#!/usr/bin/env python3
"""
Augment flattened fine-unit JSONL with original fine-unit definitions.

The script:
1. Reads fine_unit_rows.json to build a lookup from fine-unit IDs to their
   definition metadata.
2. Iterates over the flattened JSONL (merged data) and appends:
   - original_defs: list of definition strings pulled from the lookup for each ID.
   - pattern: for grammar entries, the pattern associated with the single ID;
              otherwise an empty string.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def build_lookup(rows_path: Path) -> Dict[str, Dict[str, Any]]:
    """Build a map from fine-unit ID to its first row metadata."""
    raw = json.loads(rows_path.read_text(encoding="utf-8"))
    lookup: Dict[str, Dict[str, Any]] = {}
    for entries in raw.values():
        for entry in entries:
            unit_id = entry.get("id")
            if not unit_id:
                continue
            lookup.setdefault(unit_id, entry)
    return lookup


def enrich(
    source_jsonl: Path,
    output_jsonl: Path,
    lookup: Dict[str, Dict[str, Any]],
) -> None:
    """Write the enriched JSONL file."""
    with source_jsonl.open("r", encoding="utf-8") as infile, output_jsonl.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {source_jsonl}"
                ) from exc

            fine_ids: List[str] = record.get("fine_unit_ids", [])
            original_defs: List[str] = []
            for unit_id in fine_ids:
                info = lookup.get(unit_id, {})
                original_defs.append(info.get("def", ""))

            record["original_defs"] = original_defs

            if record.get("kind") == "grammar" and fine_ids:
                info = lookup.get(fine_ids[0], {})
                record["pattern"] = info.get("pattern", "")
            else:
                record["pattern"] = ""

            json.dump(record, outfile, ensure_ascii=False)
            outfile.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich flattened fine-unit JSONL data with original defs and patterns."
        )
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--rows-json",
        type=Path,
        default=default_dir / "fine_unit_rows.json",
        help="fine_unit_rows.json path (default: alongside this script).",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=default_dir / "fine_unit_all_flat_sorted.jsonl",
        help="Flattened JSONL to enrich (default: fine_unit_all_flat_sorted.jsonl).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=default_dir / "fine_unit_all_flat_enriched.jsonl",
        help="Destination JSONL (default: fine_unit_all_flat_enriched.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookup = build_lookup(args.rows_json)
    enrich(args.input_jsonl, args.output_jsonl, lookup)


if __name__ == "__main__":
    main()
