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
    "你是一名资深词典编纂者（lexicographer）。你的任务是：把细粒度词典义项聚合成**面向学习者**的少量粗粒度义项簇。\n"
    "\n"
    "【输入说明】\n"
    "- 头部行提供词条：'Fine-grained key: {kind}:{label}'。\n"
    "- 你会收到一个细粒度释义的 JSON 数组：每项含 id（字符串）、pos（词性，字符串）、def（定义，字符串）。\n"
    "- 你的工作是把这些细义分组成若干**对学习最有帮助**的粗粒度簇，并且只按给定的 schema 输出 JSON。\n"
    "\n"
    "【必须遵守的硬性约束】\n"
    "1) 覆盖且互斥：每个输入 id 必须且只出现一次；不得遗漏、重复、伪造或修改 id。\n"
    "2) 每簇单一词性：同一簇内只允许一个 pos；簇的 pos 必须来自其成员实际词性（如 'verb'、'noun'、'adjective'、'adverb'）。\n"
    "3) 仅输出 JSON：严格匹配提供的 schema，不添加任何多余文字、解释或推理过程。\n"
    "\n"
    "【以学习为中心的分簇目标】\n"
    "- 目标不是复现词典细微差别，而是形成**少而清晰**的“可教用法”。\n"
    "- 优先保证：学习者能一眼区分各簇、能在真实表达中**直接套用**。\n"
    "- 粗粒度数量不设硬性上限或下限；但应避免不必要的细分，确保每簇都具有**明显的使用边界**。\n"
    "- 通过 coarse_senses 的**顺序**体现教学优先级：先列**核心/高频**，再列**常见扩展**，最后是**专业/少见**。\n"
    "\n"
    "【聚类原则（按优先级应用）】\n"
    "A) 语义框架与论元结构：优先把“同一事件/框架、相近论元与搭配”的细义归为一簇（例如典型宾语/施事、常见介词、谁对谁做了什么）。\n"
    "B) 字面与比喻：若搭配或句法差异明显，应将字面物理义与比喻/抽象义分簇；只有在学习场景下**用法几乎一致**时才合并。\n"
    "C) 领域专属：财务/法律/技术等专业用法应单列，方便识别和避坑。\n"
    "D) 用法同、差别小则合并：仅在程度/语域/文体上有差别、但**使用方式相同**的细义，应合并为一个更易学的簇。\n"
    "E) 过程 vs 结果、致使 vs 状态：若学习上可互相替换且搭配一致，可同簇；若会引起误用，应分开。\n"
    "\n"
    "【字段内容写作（自由度高，聚焦可教性）】\n"
    "- english_def / chinese_def：写成面向学习者的释义性改写，避免照抄输入定义。可自由融入对学习有帮助的线索（如典型搭配、常见宾语、常见情景）。\n"
    "- chinese_label：用便于记忆的中文标签概括该簇的核心用法，便于做卡片记忆。\n"
    "- chinese_criteria：自由格式，给出能帮助**判定成员归属**的线索；可写“包含/排除”条件、常见搭配、论元提示、典型替换词等，按需选择。\n"
    "- ids / pos：按 schema 正确填写即可（不强制排序方式）。\n"
    "\n"
    "【冲突裁决（当某一细义看似可归多簇）】\n"
    "- 选择能**最大化簇间区分度**、最利于学习者避免混淆的归属。\n"
    "- 若一个细义在多个框架之间摆动，优先归入**更常用、更可迁移**的框架（除非输入明示相反）。\n"
    "- 当搭配/论元与某簇强匹配时，以此为主要依据。\n"
    "\n"
    "【自检清单（只做内部检查，不输出）】\n"
    "- 所有 id 覆盖且互斥；无跨 POS 的混合簇。\n"
    "- 各簇可被学习者用“搭配/论元/场景”清楚区分。\n"
    "- 释义是为学习者改写的、可直接迁移到表达中的表述。\n"
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
