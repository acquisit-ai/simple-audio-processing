#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Convert JSON Lines files into standard JSON array files.

By default this script checks for coarse_senses.jsonl and coarse_senses_stage2.jsonl
under fine-coarse/. For every JSONL file that exists, it writes a JSON file with the
same stem (coarse_senses.json, coarse_senses_stage2.json). Non-existent files are
silently skipped.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL files into JSON array files."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            Path("fine-coarse") / "coarse_senses.jsonl",
            Path("fine-coarse") / "coarse_senses_stage2.jsonl",
        ],
        help=(
            "JSONL files to convert. Default checks coarse_senses.jsonl and "
            "coarse_senses_stage2.jsonl in fine-coarse/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine-coarse"),
        help="Directory to write JSON outputs (default: fine-coarse/).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[object]:
    data: list[object] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return data


def write_json(data: list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    processed = 0

    for input_path in args.inputs:
        input_path = Path(input_path)
        if not input_path.exists():
            continue

        entries = read_jsonl(input_path)
        output_path = args.output_dir / (input_path.stem + ".json")
        write_json(entries, output_path)

        processed += 1
        print(f"Converted {input_path} -> {output_path} ({len(entries)} entries)")

    if processed == 0:
        print("No JSONL files found; nothing converted.")


if __name__ == "__main__":
    main()
