#!/usr/bin/env python3
"""
Prepare prompts for converting fine-grained dictionary entries into coarse-grained ones.

Reads fine_unit_rows_multiple.json, builds ChatGPT prompts for selected [kind, label]
entries, and prints each prompt/response so they can be inspected before making real API calls.
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
                        "english_label": {"type": "string"},
                    },
                    "required": ["ids", "pos", "english_def", "chinese_def", "chinese_criteria", "chinese_label", "english_label"],
                },
            }
        },
        "required": ["coarse_senses"],
    },
}

TARGET_WORD_LABELS = [
    "orange",
    "apple",
    "significant",
    "elephant",
    "pornography",
    "versatile",
    "run",
    "go",
    "season",
    "design",
    "record",
]

INSTRUCTIONS = (
    "You are a senior lexicographer. Your task is to aggregate fine-grained dictionary senses into a small set of "
    "learner-oriented, coarse-grained clusters for **Chinese learners of English** (Simplified Chinese context).\n"
    "\n"
    "[INPUT]\n"
    "- The header line provides the headword: 'Fine-grained key: {kind}:{label}'.\n"
    "- You also receive a JSON array of fine senses; each item has: id (string), pos (string), def (string).\n"
    "- Group these fine senses into clusters that are most helpful for Chinese learners, and output JSON that matches the provided schema only.\n"
    "\n"
    "[HARD CONSTRAINTS]\n"
    "1) Coverage & exclusivity: Use every input id exactly once. Do not drop, duplicate, invent, or modify ids.\n"
    "2) Single POS per cluster: Each cluster must have one pos only, and that pos must be present among its member senses "
    "(e.g., 'verb', 'noun', 'adjective', 'adverb'). If the headword spans multiple POS, create separate clusters per POS.\n"
    "3) JSON only: Return JSON that matches the given schema; do not add any extra text, explanations, or reasoning.\n"
    "4) chinese_criteria policy: chinese_criteria **must be written in Simplified Chinese** and is for **internal auditing only** "
    "(not learner-facing). Do **not** put examples, collocations, or teaching tips there. Put usage/collocation cues in english_def and/or chinese_def instead.\n"
    "\n"
    "[LEARNER-CENTRIC GOALS]\n"
    "- Audience is Chinese learners. The goal is not to replicate tiny dictionary distinctions but to produce a few clear, teachable uses.\n"
    "- Ensure clusters are easy to tell apart and map to real usage (collocations, arguments, contexts) reflected in the defs.\n"
    "- No fixed limit on cluster count, but avoid unnecessary splits; each cluster should present a clear, practical usage boundary.\n"
    "- Teaching order: list clusters by priority — core/high-frequency → common extensions → domain-specific/rare.\n"
    "- **Consolidation rule for Chinese learners:** If two candidate clusters share the **same POS** and would receive **near-identical chinese_label** "
    "(i.e., the same Chinese concept for learners), **prefer merging them into a single cluster**. Represent the minor variations inside the english_def/chinese_def "
    "as usage notes or sub-variants, rather than creating separate clusters. Split only when collocations/argument patterns diverge in ways that would cause learner errors.\n"
    "\n"
    "[CLUSTERING PRINCIPLES (IN PRIORITY ORDER)]\n"
    "A) Semantic frame & argument structure: Group senses that share the same event/frame and similar argument patterns "
    "(typical objects, prepositions, who-does-what-to-what).\n"
    "B) Literal vs figurative: Split literal/physical meanings from figurative/abstract ones when collocations or syntax differ; "
    "merge only if learner usage is essentially the same.\n"
    "C) Domain-specific uses: Create dedicated clusters for finance/legal/technical uses to aid quick recognition.\n"
    "D) Same usage, small differences: Merge senses that differ only in intensity/register/style but share the same usage pattern.\n"
    "E) Process vs result; causative vs state: Merge when interchangeable for learners with the same patterns; split if the difference affects usage and error risk.\n"
    "\n"
    "[GUIDANCE FOR FILLING FIELDS]\n"
    "- english_def / chinese_def: Write learner-facing paraphrases (do not copy input defs). Include helpful usage cues such as "
    "typical collocations, common objects, argument structure, or scenarios. Keep english_def in English and chinese_def in Simplified Chinese.\n"
    "- chinese_label: A memorable Chinese label capturing the cluster’s core use (flashcard-friendly for Chinese learners).\n"
    "- english_label: An English label aligned with chinese_label that captures the same core use (for cross-language tagging).\n"
    "- chinese_criteria: Write in Simplified Chinese **for internal reviewers only**. Provide decision cues that help verify "
    "membership/boundaries (e.g., inclusion/exclusion heuristics, diagnostic hints). Avoid examples, collocations, or learner-facing tips; "
    "do not use English here.\n"
    "- ids / pos: Fill according to the schema.\n"
    "\n"
    "[CONFLICT RESOLUTION]\n"
    "- Assign a borderline sense to the cluster that maximizes contrast between clusters and reduces learner confusion.\n"
    "- If a sense spans multiple frames, prefer the more common and transferable frame for Chinese learners unless the input clearly indicates otherwise.\n"
    "- When collocations/arguments strongly indicate a cluster, treat that as primary evidence (present these cues in defs, not in chinese_criteria).\n"
    "\n"
    "[SELF-CHECK (DO NOT OUTPUT)]\n"
    "- All ids are covered exactly once; no cluster mixes POS.\n"
    "- Clusters are distinguishable by collocations/arguments/contexts (reflected in english_def/chinese_def), not just wording.\n"
    "- chinese_criteria is in Simplified Chinese, internal-audit-oriented, and free of examples/collocations/teaching tips.\n"
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
        "--labels",
        nargs="+",
        default=TARGET_WORD_LABELS,
        help=(
            "Word labels to process (kind fixed to 'word'). "
            "Default: orange apple significant elephant pornography versatile run go season design record"
        ),
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

    client = OpenAI()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    for label in args.labels:
        key = f"[word, {label}]"
        rows = data.get(key)
        if rows is None:
            print(f"!!! Skipping {key}: not found in input JSON.")
            continue

        prompt, rows_payload = build_prompt(key, rows)
        print(f"=== Prompt for {key} ===")
        print("Instructions:")
        print(INSTRUCTIONS)
        print()
        print("Input:")
        print(prompt)
        print()

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
            original_kind, original_label = parse_key(key)
            failure_record = {
                "kind": original_kind,
                "label": original_label,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "rows": rows_payload,
            }
            with args.failed_output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure_record, ensure_ascii=False))
                handle.write("\n")
            print(f"(Logged failure to {args.failed_output})")
            print()
            continue

        print("--- Model Response ---")
        original_kind, original_label = parse_key(key)
        enriched = {
            "kind": original_kind,
            "label": original_label,
            **parsed,
        }
        print(json.dumps(enriched, ensure_ascii=False, indent=2))
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, ensure_ascii=False))
            handle.write("\n")
        print(f"(Appended to {args.output})")
        print()


if __name__ == "__main__":
    main()
