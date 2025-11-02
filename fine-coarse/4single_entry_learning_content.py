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


INSTRUCTIONS = (
    "You are an experienced bilingual lexicographer who writes learning materials for native Chinese speakers learning English.\n"
    "\n"
    "[Input]\n"
    "- The entry key is provided in the form \"Fine-grained key: {kind}:{label}\".\n"
    "- A fine-grained English sense of a word/phrase/grammar item is provided with the fields: id (string), pos (part-of-speech string), def (English definition).\n"
    "- Field 'pattern': leave this empty if the entry is a word or phrase; if the entry is a grammar item, this field gives common syntactic patterns or collocation cues.\n"
    "\n"
    "[Task]\n"
    "- Turn this sense into bilingual content for Chinese-native English learners. Do not fabricate or add extra senses.\n"
    "- Return exactly one learning item, formatted as a flat JSON object containing the specified fields.\n"
    "\n"
    "[Hard Constraints]\n"
    "1) Output only a JSON object, and include only these fields: 'id', 'pos', 'english_label', 'chinese_label', 'chinese_def', 'english_def'.\n"
    "2) Preserve semantic fidelity; do not alter the original sense.\n"
    "3) 'id' and 'pos' must exactly match the input.\n"
    "4) All Chinese text must use Simplified Chinese.\n"
    "5) Do not add any extra keys, such as 'usage_tips', 'example_sentence', or 'example_translation'.\n"
    "\n"
    "[Field Descriptions]\n"
    "- english_label: a concise English paraphrase or near-synonymous phrase; do not include the headword itself, brackets, or extra punctuation.\n"
    "- chinese_label: at least provide a literal Chinese translation. You may optionally add a brief core-meaning summary to aid memorization.\n"
    "- chinese_def: the core field for learning—a full Chinese explanation of the sense. You may naturally weave in typical collocations, common objects, common scenarios, or syntactic pattern tips. It can be slightly more detailed as needed.\n"
    "- english_def: an English explanation aligned with chinese_def, mirroring its meaning and key usage cues.\n"
    "\n"
    "[Style & Quality]\n"
    "- Be learner-centered and focused on language learning; explanations must be clear and easy to understand.\n"
    "- Provide sufficient information without being verbose.\n"
)


OUTPUT_TEMPLATE = {
    "id": "<string>",
    "pos": "<string>",
    "english_label": "<string>",
    "chinese_label": "<string>",
    "english_def": "<string>",
    "chinese_def": "<string>",
}


def validate_output(payload: Dict[str, object]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Response is not a JSON object.")

    required_fields = {
        "id",
        "pos",
        "english_label",
        "chinese_label",
        "english_def",
        "chinese_def",
    }
    if set(payload.keys()) != required_fields:
        raise ValueError("Response must contain only the required fields.")

    for field in required_fields:
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Field '{field}' must be a string.")


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
        "pattern": row.get("pattern", ""),
        "lang": row.get("lang", ""),
        "status": row.get("status", ""),
    }
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    template_text = json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)
    prompt = (
        f"Fine-grained key: {kind}:{label}\n"
        "Single sense (JSON object):\n"
        f"{payload_text}\n"
        "\n"
        "Output a JSON object that matches the structure below exactly. Replace the placeholder values with real content while keeping the keys.\n"
        "Return JSON only—no explanations, comments, or extra keys.\n"
        f"{template_text}\n"
    )
    return prompt, payload


def call_openai_with_retry(
    *,
    instructions: str,
    prompt: str,
    timeout: Optional[float] = None,
    retries: int = 1,
) -> Dict[str, object]:
    attempts = retries + 1
    last_error: Optional[Exception] = None
    client = OpenAI(base_url="https://api.just2chat.cn/v1")

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model="Qwen3-30B-A3B-Instruct-2507",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )
            message = response.choices[0].message
            if hasattr(message, "content"):
                content = message.content
            else:
                content = response.choices[0].text  # Legacy compatibility

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", ""))
                    else:
                        text_parts.append(str(part))
                raw_text = "".join(text_parts)
            else:
                raw_text = str(content)

            parsed = json.loads(raw_text)
            validate_output(parsed)
            return parsed
        except (
            APIError,
            APITimeoutError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
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
    timeout: Optional[float],
    retries: int,
) -> Tuple[int, str, Dict[str, object]]:
    prompt, payload = build_prompt(key, rows)
    kind, label = parse_key(key)
    try:
        parsed = call_openai_with_retry(
            instructions=instructions,
            prompt=prompt,
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
        default=None,
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
        default=10,
        help="Extra retry attempts beyond SDK defaults (default: 10).",
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
