#!/usr/bin/env python3
"""
Split fine_unit_rows.json into single-row and multi-row JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_mapping(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at root, got {type(data).__name__}")
    return data


def partition_rows(mapping: dict[str, list[dict[str, Any]]]) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]
]:
    single: dict[str, list[dict[str, Any]]] = {}
    multiple: dict[str, list[dict[str, Any]]] = {}

    for key, rows in mapping.items():
        if len(rows) == 1:
            single[key] = rows
        else:
            multiple[key] = rows

    return single, multiple


def write_json(data: dict[str, list[dict[str, Any]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split fine_unit_rows.json into single-row and multi-row JSON files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows.json",
        help="Source JSON mapping (default: fine-coarse/fine_unit_rows.json)",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows_single.json",
        help="Output path for entries with exactly one row (default: fine-coarse/fine_unit_rows_single.json)",
    )
    parser.add_argument(
        "--multiple-output",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows_multiple.json",
        help="Output path for entries with multiple rows (default: fine-coarse/fine_unit_rows_multiple.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = read_mapping(args.input)
    singles, multiples = partition_rows(mapping)

    args.single_output.parent.mkdir(parents=True, exist_ok=True)
    args.multiple_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(singles, args.single_output)
    write_json(multiples, args.multiple_output)


if __name__ == "__main__":
    main()
