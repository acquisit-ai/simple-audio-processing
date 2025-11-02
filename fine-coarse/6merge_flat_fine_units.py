#!/usr/bin/env python3
"""
Merge flattened fine-unit JSONL files and sort by label.

Reads the outputs produced by flatten_fine_unit_coarse_senses.py and
convert_fine_unit_single_to_flat.py, combines all rows, sorts them by the
`label` field (lexicographically using casefold), and writes a unified JSONL.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def load_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}"
                ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine flat fine-unit JSONL files and sort by label."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--multiple-jsonl",
        type=Path,
        default=default_dir / "fine_unit_multiple_flat.jsonl",
        help="Flattened multiple-sense JSONL (default: fine_unit_multiple_flat.jsonl).",
    )
    parser.add_argument(
        "--single-jsonl",
        type=Path,
        default=default_dir / "fine_unit_single_flat.jsonl",
        help="Flattened single-sense JSONL (default: fine_unit_single_flat.jsonl).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=default_dir / "fine_unit_all_flat_sorted.jsonl",
        help="Destination JSONL file (default: fine_unit_all_flat_sorted.jsonl).",
    )
    return parser.parse_args()


def convert(multiple_path: Path, single_path: Path, output_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    rows.extend(load_rows(multiple_path))
    rows.extend(load_rows(single_path))

    rows.sort(key=lambda row: (row.get("label", ""), row.get("kind", ""), row.get("pos", "")))

    with output_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            json.dump(row, outfile, ensure_ascii=False)
            outfile.write("\n")


def main() -> None:
    args = parse_args()
    convert(args.multiple_jsonl, args.single_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
