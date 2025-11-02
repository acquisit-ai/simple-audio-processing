#!/usr/bin/env python3
"""
Flatten fine_unit_multiple.jsonl so each coarse sense becomes its own JSONL row.

The output JSONL contains one object per coarse sense with the fields:
kind, label, pos, english_def, chinese_def, chinese_criteria,
chinese_label, english_label, fine_unit_ids.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


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


def flatten_records(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield one flattened record per coarse sense."""
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

            base = {
                "kind": record.get("kind", ""),
                "label": record.get("label", ""),
            }

            coarse_list: List[Dict[str, Any]] = record.get("coarse_senses", [])
            for coarse in coarse_list:
                flattened = {
                    **base,
                    "pos": coarse.get("pos", ""),
                    "english_def": coarse.get("english_def", ""),
                    "chinese_def": coarse.get("chinese_def", ""),
                    "chinese_criteria": coarse.get("chinese_criteria", ""),
                    "chinese_label": coarse.get("chinese_label", ""),
                    "english_label": coarse.get("english_label", ""),
                    "fine_unit_ids": coarse.get("ids", []),
                }

                # Ensure all expected fields are present even if missing in source.
                for field in OUTPUT_FIELDS:
                    default_value = [] if field == "fine_unit_ids" else ""
                    flattened.setdefault(field, default_value)

                yield flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten fine_unit_multiple.jsonl into one coarse sense per JSONL row."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=default_dir / "fine_unit_multiple.jsonl",
        help="Source JSONL file (default: fine_unit_multiple.jsonl next to this script).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=default_dir / "fine_unit_multiple_flat.jsonl",
        help="Destination JSONL file (default: fine_unit_multiple_flat.jsonl next to this script).",
    )
    return parser.parse_args()


def convert(input_path: Path, output_path: Path) -> None:
    rows = flatten_records(input_path)
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            json.dump(row, outfile, ensure_ascii=False)
            outfile.write("\n")


def main() -> None:
    args = parse_args()
    convert(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
