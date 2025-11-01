#!/usr/bin/env python3
"""
Refine previously generated coarse-grained clusters with a second pass.

Reads coarse_senses.jsonl (or another JSONL file with coarse clusters), sends the
existing clusters to the model, and asks for a more aggressive consolidation that
still remains learner-friendly for Chinese learners.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from openai import APIError, APITimeoutError


JSON_SCHEMA = {
    "type": "json_schema",
    "name": "coarse_groupings_v2",
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
                    "required": [
                        "ids",
                        "pos",
                        "english_def",
                        "chinese_def",
                        "chinese_criteria",
                        "chinese_label",
                        "english_label",
                    ],
                },
            }
        },
        "required": ["coarse_senses"],
    },
}


INSTRUCTIONS = (
    "Goal: Aggressively consolidate coarse-grained clusters into fewer, clearer clusters for Chinese learners of English, "
    "while keeping coverage/exclusivity and never merging across parts of speech (POS).\n"
    "\n"
    "Input:\n"
    "- Header: 'Fine-grained key: {kind}:{label}'.\n"
    "- Fine senses: JSON array with id (string), pos (string), def (string).\n"
    "- Current clusters: JSON 'coarse_senses' using the target schema.\n"
    "\n"
    "Output:\n"
    "- Return JSON only, matching the existing schema exactly (no extra keys or text).\n"
    "\n"
    "Hard constraints:\n"
    "1) Coverage & exclusivity: Every input id appears exactly once in the final output; do not drop, duplicate, invent, or edit ids.\n"
    "2) No cross-POS merges: Each cluster has a single POS, which must occur among its member senses. If any cluster mixes POS, split by POS first.\n"
    "3) english_label: Output only the bare label phrase (no headword, parentheses, or extra punctuation).\n"
    "4) chinese_criteria: Write in Simplified Chinese for internal auditing only; provide diagnostic inclusion/exclusion cues. "
    "Do not put examples/collocations/teaching tips here—place usage cues in the definitions.\n"
    "\n"
    "Aggressive consolidation rules (merge whenever it stays teachable):\n"
    "- Same POS + near-identical chinese_label (same Chinese concept) ⇒ merge. Put minor nuances inside english_def/chinese_def.\n"
    "- Same POS + highly overlapping english_label (near-synonyms/slash variants) ⇒ merge.\n"
    "- Literal vs figurative: merge when collocations/syntax are effectively the same for learners; split only if stable differences in "
    "argument structure/prepositions would cause errors.\n"
    "- Domain uses (finance/legal/tech): if usage patterns match the general cluster, merge into the general cluster and note the domain in the defs; "
    "keep separate only when patterns are distinct and pedagogically relevant.\n"
    "- Process vs result; causative vs state; abstract vs concrete: merge if interchangeable in learner usage; split only when the difference drives different collocations/arguments.\n"
    "- Absorb micro-clusters that largely overlap a bigger cluster.\n"
    "\n"
    "Noun-specific guidance (typical pattern like 'record'):\n"
    "- If multiple noun clusters all mean “a stored account/evidence/history of facts or performance,” prefer a single umbrella cluster "
    "(e.g., records/archives/history) and explain sub-uses in the defs (official files; personal history/track record; sports/statistical records; legal/conviction records). "
    "Keep truly different senses (e.g., audio medium/record) separate.\n"
    "\n"
    "Do not merge when:\n"
    "- POS differs.\n"
    "- Collocations/argument structure/prepositions differ in a way that changes usage choice for learners.\n"
    "- Meanings have consistent opposite polarity/direction reflected in syntax.\n"
    "\n"
    "Field rewriting after merges:\n"
    "- ids/pos: take the union of ids; keep the POS.\n"
    "- chinese_label/english_label: choose a higher-level, general label aligned across languages (english_label is the bare phrase only).\n"
    "- english_def/chinese_def: learner-facing paraphrases (do not copy input defs). Fold merged sub-uses into the defs and include helpful usage cues "
    "(typical collocations, common objects, argument structure, scenarios). Keep english_def in English and chinese_def in Simplified Chinese.\n"
    "- chinese_criteria: internal Simplified Chinese notes for membership/boundaries only (no examples/collocations/teaching tips).\n"
    "- Ordering: core/high-frequency → common extensions → domain-specific/rare. Remove fully absorbed clusters and unify style.\n"
    "\n"
    "Conflict resolution:\n"
    "- When multiple merges are possible, prefer the option that reduces cluster count without increasing confusion. "
    "Prioritize the more common/transferable frame for Chinese learners. Use collocations/arguments as primary evidence and reflect them in the defs.\n"
)


def load_api_key(env_path: Path) -> Optional[str]:
    """Load OPENAI_API_KEY from a .env file if present."""
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


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc
    return entries


def build_prompt(entry: Dict[str, object]) -> Tuple[str, List[Dict[str, object]]]:
    kind = entry.get("kind", "")
    label = entry.get("label", "")
    coarse_senses = entry.get("coarse_senses", [])
    if not isinstance(coarse_senses, list):
        raise ValueError(f"Entry {kind}:{label} has invalid coarse_senses format")

    prompt_payload: List[Dict[str, object]] = []
    for cluster in coarse_senses:
        if not isinstance(cluster, dict):
            continue
        prompt_payload.append(
            {
                "ids": cluster.get("ids", []),
                "pos": cluster.get("pos", ""),
                "english_def": cluster.get("english_def", ""),
                "chinese_def": cluster.get("chinese_def", ""),
                "chinese_criteria": cluster.get("chinese_criteria", ""),
                "chinese_label": cluster.get("chinese_label", ""),
                "english_label": cluster.get("english_label", ""),
            }
        )

    payload_text = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    prompt = (
        f"Coarse key: {kind}:{label}\n"
        "Existing coarse clusters (JSON array):\n"
        f"{payload_text}\n"
    )
    return prompt, prompt_payload


def call_openai_with_retry(
    *,
    instructions: str,
    prompt: str,
    json_schema: Dict[str, object],
    timeout: Optional[float] = None,
    additional_retry: int = 1,
) -> Dict[str, object]:
    attempts = additional_retry + 1
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
    entry: Dict[str, object],
    *,
    instructions: str,
    json_schema: Dict[str, object],
    timeout: Optional[float],
    retries: int,
) -> Tuple[int, str, Dict[str, object]]:
    kind = entry.get("kind", "")
    label = entry.get("label", "")
    prompt, payload = build_prompt(entry)

    try:
        parsed = call_openai_with_retry(
            instructions=instructions,
            prompt=prompt,
            json_schema=json_schema,
            timeout=timeout,
            additional_retry=retries,
        )
        enriched: Dict[str, object] = {
            "kind": kind,
            "label": label,
            "version": entry.get("version", 1) + 1,
            "previous_version": entry.get("version", 1),
            **parsed,
        }
        return index, "ok", enriched
    except Exception as exc:
        failure = {
            "kind": kind,
            "label": label,
            "version": entry.get("version", 1),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "input_clusters": payload,
        }
        return index, "err", failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine coarse clusters with a second-pass AI consolidation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses.jsonl",
        help="Source JSONL of first-pass coarse clusters (default: fine-coarse/coarse_senses.jsonl)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env holding OPENAI_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional list of labels to process (kind fixed to 'word'); if omitted, process all entries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit to first N entries after filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses_stage2.jsonl",
        help="Append refined JSON lines (default: fine-coarse/coarse_senses_stage2.jsonl)",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses_stage2_failed.jsonl",
        help="Append failures for later inspection (default: fine-coarse/coarse_senses_stage2_failed.jsonl)",
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

    entries = read_jsonl(args.input)
    if not entries:
        print(f"No entries found in {args.input}")
        return

    label_set = {label.lower() for label in args.labels} if args.labels else None
    filtered: List[Dict[str, object]] = []
    for entry in entries:
        if entry.get("kind") != "word":
            continue
        label = str(entry.get("label", "")).lower()
        if label_set is not None and label not in label_set:
            continue
        filtered.append(entry)

    if not filtered:
        print("No matching entries found; exiting.")
        return

    if args.limit is not None:
        filtered = filtered[: args.limit]

    total = len(filtered)
    print(f"Preparing refinement for {total} entries...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as fout, args.failed_output.open(
        "a", encoding="utf-8"
    ) as ffail:
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for idx, entry in enumerate(filtered):
                fut = executor.submit(
                    process_entry,
                    idx,
                    entry,
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
        f"Done. Refined outputs appended to {args.output}, failures logged to {args.failed_output}."
    )


if __name__ == "__main__":
    main()
