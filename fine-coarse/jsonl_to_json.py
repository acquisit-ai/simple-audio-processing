#!/usr/bin/env python3
"""
Convert a JSON Lines file into a standard JSON array file.

Reads `coarse_senses.jsonl` (or any provided path), parses each line as JSON,
and writes the aggregated list to a `.json` file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file into a JSON array file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses.jsonl",
        help="Source JSON Lines file (default: fine-coarse/coarse_senses.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses.json",
        help="Destination JSON file (default: fine-coarse/coarse_senses.json)",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[object]:
    items: list[object] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return items


def write_json(data: list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    entries = read_jsonl(args.input)
    write_json(entries, args.output)
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
