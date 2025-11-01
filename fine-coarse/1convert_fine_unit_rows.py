#!/usr/bin/env python3
"""
Convert fine_unit_rows.csv into a JSON mapping keyed by [kind, label].

Each key is the string representation "[kind, label]" and the value is a list
of row dicts containing all remaining columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "r": "adverb",
}

KIND_MAP = {
    "grammar_rule": "grammar",
    "phrase_sense": "phrase",
    "word_sense": "word",
}


def build_mapping(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        try:
            kind = row["kind"]
            label_raw = row["label"]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Missing expected column: {exc}") from exc

        label_clean = label_raw.strip()

        if label_clean.isdigit():
            continue

        if label_clean.lower().endswith("th") and label_clean[:-2].isdigit():
            continue

        if label_clean.lower().endswith("s") and label_clean[:-1].isdigit():
            continue

        mapped_kind = KIND_MAP.get(kind, kind)
        key = f"[{mapped_kind}, {label_raw}]"
        # Remove grouping keys while keeping the rest of the row data intact.
        row_payload = {
            field: value
            for field, value in row.items()
            if field not in {"kind", "label", "created_at", "updated_at"}
        }

        if "pos" in row_payload:
            row_payload["pos"] = POS_MAP.get(row_payload["pos"], row_payload["pos"])
        mapping.setdefault(key, []).append(row_payload)

    return mapping


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_json(data: dict[str, list[dict[str, str]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert fine_unit_rows.csv into JSON keyed by [kind, label]."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows.csv",
        help="Path to fine_unit_rows.csv (default: fine-coarse/fine_unit_rows.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows.json",
        help="Output JSON path (default: fine-coarse/fine_unit_rows.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    mapping = build_mapping(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(mapping, args.output)


if __name__ == "__main__":
    main()
