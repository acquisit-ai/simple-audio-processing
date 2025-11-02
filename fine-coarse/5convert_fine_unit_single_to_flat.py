#!/usr/bin/env python3
"""
Normalize fine_unit_single.jsonl entries into the shared coarse-sense schema.

The script reshapes each row into objects that expose:
kind, label, pos, english_def, chinese_def, chinese_criteria,
chinese_label, english_label, fine_unit_ids.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable


OUTPUT_FIELDS = [
    "kind",
    "label",
    "pos",
    "english_def",
    "chinese_def",
    "chinese_criteria",
    "chinese_label",
    "english_label",
    "fine_unit_ids",
]


def transform_records(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield normalized records from the input JSONL."""
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {jsonl_path}"
                ) from exc

            normalized: Dict[str, Any] = {
                "kind": record.get("kind", ""),
                "label": record.get("label", ""),
                "pos": record.get("pos", ""),
                "english_def": record.get("english_def", ""),
                "chinese_def": record.get("chinese_def", ""),
                "chinese_criteria": record.get("chinese_criteria", ""),
                "chinese_label": record.get("chinese_label", ""),
                "english_label": record.get("english_label", ""),
                "fine_unit_ids": [record.get("id")] if record.get("id") else [],
            }

            for field in OUTPUT_FIELDS:
                default_value = [] if field == "fine_unit_ids" else ""
                normalized.setdefault(field, default_value)

            yield normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert fine_unit_single.jsonl into the flattened coarse-sense schema."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=default_dir / "fine_unit_single.jsonl",
        help="Source JSONL file (default: fine_unit_single.jsonl next to this script).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=default_dir / "fine_unit_single_flat.jsonl",
        help="Destination JSONL file (default: fine_unit_single_flat.jsonl next to this script).",
    )
    return parser.parse_args()


def convert(input_path: Path, output_path: Path) -> None:
    rows = transform_records(input_path)
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            json.dump(row, outfile, ensure_ascii=False)
            outfile.write("\n")


def main() -> None:
    args = parse_args()
    convert(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
