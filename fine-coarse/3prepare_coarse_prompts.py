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
import time
from pathlib import Path
from typing import Iterable, List, Dict

from openai import OpenAI
from openai import APIError, APITimeoutError

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
                        "chinese_criteria": {"type": "string"},
                        "chinese_label": {"type": "string"},
                    },
                    "required": ["ids", "pos", "english_def", "chinese_def", "chinese_criteria", "chinese_label"],
                },
            }
        },
        "required": ["coarse_senses"],
    },
}

INSTRUCTIONS = (
    "You are an expert lexicographer converting fine-grained dictionary senses into a coarser taxonomy.\n"
    "Follow these rules:\n"
    "1. Cluster senses that share a common meaning or usage context. If two senses differ semantically, keep them separate.\n"
    "2. Select one coarse part of speech that best fits each cluster. Prefer the dominant POS; if mixed, use the POS that captures the shared sense.\n"
    "3. Provide concise explanations in English and Chinese describing the cluster-level meaning (not individual senses).\n"
    "4. Supply a short Chinese label (直译或概括) summarising the cluster.\n"
    "5. Provide a short chinese_criteria note (in Chinese) explaining the inclusion/exclusion rationale for that cluster.\n"
    "6. Respond strictly in JSON matching the provided schema.\n"
    "\n"
    "Example (illustrative only):\n"
    "{\n"
    "  \"coarse_senses\": [\n"
    "    {\n"
    "      \"ids\": [\"1001\", \"1002\"],\n"
    "      \"pos\": \"noun\",\n"
    "      \"english_def\": \"A handheld computing device such as a smartphone or tablet.\",\n"
    "      \"chinese_def\": \"手持式计算设备，如智能手机或平板电脑。\",\n"
    "      \"chinese_criteria\": \"包含指代实体设备的释义；排除表示'打电话'等动词用法。\",\n"
    "      \"chinese_label\": \"掌上设备\"\n"
    "    }\n"
    "  ]\n"
    "}"
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


def build_prompt(key: str, rows: Iterable[dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    """Construct the message that will be sent to ChatGPT."""
    kind, label = parse_key(key)
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
        f"Fine-grained key: {kind}:{label}\n"
        "Each entry includes an id, part of speech, and definition.\n"
        f"{rows_text}\n"
    )
    return prompt, rows_payload


def call_openai_with_retry(
    client: OpenAI,
    *,
    instructions: str,
    prompt: str,
    json_schema: dict[str, object],
    timeout: float | None = None,
    additional_retry: int = 1,
) -> dict[str, object]:
    """
    Call OpenAI Responses API with one extra manual retry beyond the SDK defaults.

    Parameters
    ----------
    client:
        Configured OpenAI client.
    instructions:
        System-level instructions passed via `instructions=`.
    prompt:
        User input payload.
    json_schema:
        JSON schema dict passed to `text={"format": ...}`.
    timeout:
        Optional per-request timeout in seconds.
    additional_retry:
        Extra retry attempts on top of the SDK automatic retries (default: 1).
    """

    attempts = additional_retry + 1
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = client.responses.create(
                model="gpt-5-mini",
                instructions=instructions,
                input=prompt,
                text={"format": json_schema},
                service_tier="flex",
                timeout=timeout,
            )
            return json.loads(response.output_text)
        except (APIError, APITimeoutError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            # Brief pause to give the SDK time before retrying.
            time.sleep(1 + attempt)

    assert last_error is not None
    raise last_error


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
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses_failed.jsonl",
        help="Path to append JSON lines when calls fail (default: fine-coarse/coarse_senses_failed.jsonl)",
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

    prompt, rows_payload = build_prompt(target_key, rows)
    print(f"--- Prompt for {target_key} ---")
    print("Instructions:")
    print(INSTRUCTIONS)
    print()
    print("Input:")
    print(prompt)
    print()

    client = OpenAI()
    try:
        parsed = call_openai_with_retry(
            client,
            instructions=INSTRUCTIONS,
            prompt=prompt,
            json_schema=JSON_SCHEMA,
        )
    except Exception as exc:
        print("--- Model Response ---")
        print(f"OpenAI call failed after retries: {exc}")
        original_kind, original_label = parse_key(target_key)
        failure_record = {
            "kind": original_kind,
            "label": original_label,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "rows": rows_payload,
        }
        args.failed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure_record, ensure_ascii=False))
            handle.write("\n")
        print(f"(Logged failure to {args.failed_output})")
        return

    print("--- Model Response ---")
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
