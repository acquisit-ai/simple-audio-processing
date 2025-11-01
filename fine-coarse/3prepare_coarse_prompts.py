#!/usr/bin/env python3
"""
Prepare prompts for converting fine-grained dictionary entries into coarse-grained ones.

Reads fine_unit_rows_multiple.json, builds ChatGPT prompts for each [kind, label],
and prints the first N prompts so they can be inspected before making real API calls.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

JSON_SCHEMA = {
    "type": "json_schema",
    "name": "coarse_groupings",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coarse_senses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "pos": {"type": "string"},
                        "english_def": {"type": "string"},
                        "chinese_def": {"type": "string"},
                    },
                    "required": ["ids", "pos", "english_def", "chinese_def"],
                },
            }
        },
        "required": ["coarse_senses"],
    },
}

INSTRUCTIONS = (
    "You are an expert lexicographer helping transform fine-grained dictionary entries into coarse-grained ones. "
    "Group related senses together, choose an appropriate coarse part of speech, and provide concise explanations "
    "in both English and Chinese. Respond strictly in JSON that matches the provided schema."
)


def load_api_key(env_path: Path) -> str | None:
    """Read OPENAI_API_KEY from a .env file without importing extra deps."""
    if not env_path.exists():
        return None

    key = None
    with env_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)
    return key


def parse_key(key: str) -> tuple[str, str]:
    """Turn the string form '[kind, label]' back into its components."""
    content = key.strip()[1:-1]
    kind, label = content.split(", ", 1)
    return kind, label


def build_prompt(key: str, rows: Iterable[dict[str, str]]) -> str:
    """Construct the message that will be sent to ChatGPT."""
    kind, label = parse_key(key)
    display_kind = kind.split("_", 1)[0] if "_" in kind else kind
    rows_payload = [
        {
            "id": row.get("id", ""),
            "pos": row.get("pos", ""),
            "def": row.get("def", ""),
        }
        for row in rows
    ]
    rows_text = json.dumps(rows_payload, ensure_ascii=False, indent=2)
    prompt = (
        f"Fine-grained key: {display_kind}:{label}\n"
        "Each entry includes an id, part of speech, and definition.\n"
        "Entries (JSON array):\n"
        f"{rows_text}\n"
    )
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print preview prompts for coarse-grained mapping requests."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows_multiple.json",
        help="Source JSON containing multi-row entries (default: fine-coarse/fine_unit_rows_multiple.json)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file holding OPENAI_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--example-key",
        type=str,
        default="[word, abstract]",
        help="Key to preview in the output (default: [word, abstract])",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses.jsonl",
        help="Path to append JSON lines with model outputs (default: fine-coarse/coarse_senses.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_api_key(args.env)

    with args.input.open(encoding="utf-8") as handle:
        data: dict[str, list[dict[str, str]]] = json.load(handle)

    target_key = args.example_key
    rows = data.get(target_key)
    if rows is None:
        print(f"No entry found for key {target_key!r}")
        return

    prompt = build_prompt(target_key, rows)
    print(f"--- Prompt for {target_key} ---")
    print("Instructions:")
    print(INSTRUCTIONS)
    print()
    print("Input:")
    print(prompt)
    print()

    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=INSTRUCTIONS,
        input=prompt,
        text={"format": JSON_SCHEMA},
        service_tier="priority",
    )

    print("--- Model Response ---")
    try:
        parsed = json.loads(response.output_text)
    except json.JSONDecodeError:
        print(response.output_text.strip())
    else:
        original_kind, original_label = parse_key(target_key)
        enriched = {
            "kind": original_kind,
            "label": original_label,
            **parsed,
        }
        print(json.dumps(enriched, ensure_ascii=False, indent=2))
        # Append to JSON Lines file for accumulation across runs.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False))
            handle.write("\n")
        print(f"(Appended to {args.output})")


if __name__ == "__main__":
    main()
