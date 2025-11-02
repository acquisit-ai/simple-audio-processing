#!/usr/bin/env python3
"""
Generate learner-focused content for single-entry fine-grained dictionary senses.

This script reads fine_unit_rows_single.json (mapping "[kind, label]" -> single row),
consults the OpenAI Responses API, and produces bilingual learning notes for each entry.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import APIError, APITimeoutError, OpenAI


JSON_SCHEMA = {
    "type": "json_schema",
    "name": "single_entry_learning_content",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "learning_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "pos": {"type": "string"},
                        "english_label": {"type": "string"},
                        "chinese_label": {"type": "string"},
                        "english_def": {"type": "string"},
                        "chinese_def": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "pos",
                        "english_label",
                        "chinese_label",
                        "english_def",
                        "chinese_def",
                    ],
                },
            }
        },
        "required": ["learning_items"],
    },
}


INSTRUCTIONS = (
    "You are an experienced bilingual lexicographer creating learner-facing content for Chinese learners of English.\n"
    "\n"
    "[INPUT]\n"
    "- The headword is given by 'Fine-grained key: {kind}:{label}'.\n"
    "- One fine-grained sense is provided with fields: id (string), pos (string), def (English). Other fields may appear but are not required.\n"
    "\n"
    "[TASK]\n"
    "- Convert this single sense into learner-ready, bilingual content without inventing new senses.\n"
    "- Produce exactly one learning item in the 'learning_items' array.\n"
    "\n"
    "[HARD CONSTRAINTS]\n"
    "1) JSON only: Return a JSON object that matches the provided schema exactly; include only the required fields "
    "('id', 'pos', 'english_label', 'chinese_label', 'english_def', 'chinese_def').\n"
    "2) Fidelity: Do not add or alter senses; stay faithful to the given definition.\n"
    "3) Identity: Copy 'id' and 'pos' from the input exactly (id is a string).\n"
    "4) Language: Use Simplified Chinese for all Chinese text.\n"
    "5) No extra keys: Do NOT output 'usage_tips', 'example_sentence', or 'example_translation'.\n"
    "\n"
    "[FIELD GUIDANCE]\n"
    "- english_label: A short English gloss/near-synonym as a bare phrase (no headword, no parentheses, no extra punctuation).\n"
    "- chinese_label: A concise Simplified Chinese label that captures the core idea for learners.\n"
    "- english_def: A learner-friendly paraphrase in English that highlights the core idea and, when useful, briefly signals typical usage "
    "(e.g., common objects/prepositions or argument patterns) within the sentence.\n"
    "- chinese_def: A clear Simplified Chinese explanation aligned with the English definition. Here you may **slightly expand** to include practical cues "
    "for learners—typical collocations, common objects/prepositions, productive patterns, and brief scenario hints—integrated naturally into the prose "
    "(do not add extra fields).\n"
    "\n"
    "[STYLE & QUALITY]\n"
    "- Learner-first: Emphasize how it is used in real language; prefer plain English and clear Simplified Chinese.\n"
    "- Keep it compact but informative; avoid heavy jargon and rare vocabulary.\n"
    "- Ensure english_def and chinese_def remain semantically aligned."
)


def load_api_key(env_path: Path) -> Optional[str]:
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


def parse_key(key: str) -> Tuple[str, str]:
    content = key.strip()[1:-1]
    kind, label = content.split(", ", 1)
    return kind, label


def build_prompt(key: str, rows: Iterable[Dict[str, str]]) -> Tuple[str, Dict[str, str]]:
    kind, label = parse_key(key)
    row = next(iter(rows))
    payload = {
        "id": row.get("id", ""),
        "pos": row.get("pos", ""),
        "def": row.get("def", ""),
        "lang": row.get("lang", ""),
        "status": row.get("status", ""),
    }
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = (
        f"Fine-grained key: {kind}:{label}\n"
        "Single sense (JSON object):\n"
        f"{payload_text}\n"
    )
    return prompt, payload


def call_openai_with_retry(
    *,
    instructions: str,
    prompt: str,
    json_schema: Dict[str, object],
    timeout: Optional[float] = None,
    retries: int = 1,
) -> Dict[str, object]:
    attempts = retries + 1
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            client = OpenAI()
            response = client.responses.create(
                model="gpt-5-mini",
                instructions=instructions,
                input=prompt,
                text={"format": json_schema},
                service_tier="flex",
                timeout=timeout,
            )
            return json.loads(response.output_text)
        except (APIError, APITimeoutError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(1 + attempt)

    assert last_error is not None
    raise last_error


def process_entry(
    index: int,
    key: str,
    rows: List[Dict[str, str]],
    *,
    instructions: str,
    json_schema: Dict[str, object],
    timeout: Optional[float],
    retries: int,
) -> Tuple[int, str, Dict[str, object]]:
    prompt, payload = build_prompt(key, rows)
    kind, label = parse_key(key)
    try:
        parsed = call_openai_with_retry(
            instructions=instructions,
            prompt=prompt,
            json_schema=json_schema,
            timeout=timeout,
            retries=retries,
        )
        enriched = {
            "kind": kind,
            "label": label,
            **parsed,
        }
        return index, "ok", enriched
    except Exception as exc:
        failure = {
            "kind": kind,
            "label": label,
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "input_row": payload,
        }
        return index, "err", failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate learner-focused content for single fine-grained senses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows_single.json",
        help="JSON mapping of single-sense entries (default: fine-coarse/fine_unit_rows_single.json)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env containing OPENAI_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--only-key",
        type=str,
        default=None,
        help="Process only this key, e.g. \"[word, abandon]\"; otherwise process all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Optionally limit to the first N keys (after applying --only-key).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_single.jsonl",
        help="Output JSONL for learner content (default: fine-coarse/fine_unit_single.jsonl)",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_single_failed.jsonl",
        help="Output JSONL for failures (default: fine-coarse/fine_unit_single_failed.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Number of worker threads (default: 40).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-call timeout in seconds (default: 90).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Extra retry attempts beyond SDK defaults (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_api_key(args.env)

    with args.input.open(encoding="utf-8") as handle:
        data: Dict[str, List[Dict[str, str]]] = json.load(handle)

    if args.only_key is not None:
        keys = [args.only_key] if args.only_key in data else []
        if not keys:
            print(f"No entry found for key {args.only_key!r}")
            return
    else:
        keys = list(data.keys())

    if args.limit is not None:
        keys = keys[: args.limit]

    total = len(keys)
    if total == 0:
        print("No keys to process.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as fout, args.failed_output.open(
        "a", encoding="utf-8"
    ) as ffail:
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for idx, key in enumerate(keys):
                rows = data[key]
                fut = executor.submit(
                    process_entry,
                    idx,
                    key,
                    rows,
                    instructions=INSTRUCTIONS,
                    json_schema=JSON_SCHEMA,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                futures.append(fut)

            next_to_write = 0
            buffer: Dict[int, Tuple[str, Dict[str, object]]] = {}
            completed = 0

            for fut in as_completed(futures):
                idx, status, payload = fut.result()
                buffer[idx] = (status, payload)

                while next_to_write in buffer:
                    status2, payload2 = buffer.pop(next_to_write)
                    if status2 == "ok":
                        fout.write(json.dumps(payload2, ensure_ascii=False))
                        fout.write("\n")
                        fout.flush()
                    else:
                        ffail.write(json.dumps(payload2, ensure_ascii=False))
                        ffail.write("\n")
                        ffail.flush()
                    completed += 1
                    print(
                        f"[{completed}/{total}] wrote index {next_to_write + 1} "
                        f"({'OK' if status2 == 'ok' else 'FAIL'})"
                    )
                    next_to_write += 1

    print(
        f"Done. Learner content appended to {args.output}, failures logged to {args.failed_output}."
    )


if __name__ == "__main__":
    main()
