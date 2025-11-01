#!/usr/bin/env python3
"""
Prepare prompts for converting fine-grained dictionary entries into coarse-grained ones.

Now supports parallel processing with a thread pool while preserving output order.

Usage examples:
  # Process first 20 keys in parallel (default behaviour)
  python script.py --input fine-coarse/fine_unit_rows_multiple.json --output fine-coarse/coarse_senses.jsonl

  # Process only one key for testing
  python script.py --only-key "[phrase, abstract]"

  # Control workers / timeout / retries / limit
  python script.py --workers 8 --timeout 90 --retries 2 --limit 100 --only-key "[word, abandon]"
"""

#!/usr/bin/env python3
"""
使用 OpenAI Responses API 将细粒度词典条目聚合为粗粒度释义的多线程脚本。

功能要点：
1. 并行处理：通过线程池一次性提交多个 [kind, label] 组合，自动控制输出顺序避免错乱。
2. JSON Schema 约束：使用结构化输出模式强制模型返回符合课程需求的 JSON 数据。
3. 失败记录：若模型调用失败会自动重试，如仍失败则原始数据与错误信息会写入失败日志。
4. 可按需过滤：可通过 CLI 指定仅处理单个 key、限制数量、调节超时与重试次数等。
"""

import argparse  # 负责 CLI 参数解析，方便灵活控制脚本行为
import json      # 读写 JSON 文件以及处理模型返回的 JSON 字符串
import os        # 读取环境变量（如 OPENAI_API_KEY），或设置新的环境变量
import time      # 控制重试之间的休眠时间，实现简单退避策略
from pathlib import Path  # 处理跨平台文件路径
from typing import Iterable, List, Dict, Tuple, Optional  # 类型提示提升可读性与可维护性
from concurrent.futures import ThreadPoolExecutor, as_completed  # 简化并行执行与结果收集流程

from openai import OpenAI  # OpenAI Python SDK 客户端
from openai import APIError, APITimeoutError  # 常见错误类型，用于决定是否重试

# ---------------------------------------------------------------------------
# 结构化输出所需的 JSON Schema
# ---------------------------------------------------------------------------
# 说明：
# - type 指明响应整体是一个 JSON 对象；
# - coarse_senses 是核心字段，代表聚合后的粗粒度义项列表；
# - 每个义项必须包含 ids (string 数组)、pos、english_def、chinese_def、chinese_criteria、chinese_label；
# - additionalProperties 禁止出现未定义字段，避免模型输出多余信息。
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
# ---------------------------------------------------------------------------
# 发送给模型的系统级指令说明
# ---------------------------------------------------------------------------
# 说明：
# - 初段说明任务目标；
# - [INPUT]/[HARD CONSTRAINTS]/[LEARNER-CENTRIC GOALS] 等分区详述要求；
# - 明确约束：覆盖全部 id、每个聚合仅包含单一 POS、仅返回 JSON；
# - 提供聚合原则、释义撰写建议、冲突解决策略与自检要点；尽量避免模型输出偏离主题。
INSTRUCTIONS = (
    "You are a senior lexicographer. Aggregate fine-grained dictionary senses into a small set of "
    "learner-oriented, coarse-grained clusters for **Chinese learners of English** (Simplified Chinese context).\n"
    "\n"
    "[INPUT]\n"
    "- Header: 'Fine-grained key: {kind}:{label}'.\n"
    "- Fine senses: JSON array with ID, Part-of-speech, Definition.\n"
    "\n"
    "[HARD CONSTRAINTS]\n"
    "1) Coverage & exclusivity: Use every input id exactly once; no drops, duplicates, inventions, or edits.\n"
    "2) Single POS per cluster: One pos only, and it must occur among the cluster’s member senses (e.g., 'verb', 'noun', 'adjective', 'adverb'). "
    "If the headword spans multiple POS, make separate clusters per POS.\n"
    "3) JSON only: Return JSON that matches the provided schema; no extra text or reasoning.\n"
    "4) Consolidation for Chinese learners: If two candidate clusters share the same POS and would yield a near-identical chinese_label "
    "(i.e., the same Chinese concept), merge them. For example, \"design\" meaning “to plan an abstract solution” and “to craft a concrete artistic decoration” share the same Mandarin anchor 设计, so keep them together and use the definitions to mention both scenarios. "
    "Split only when collocations/argument patterns diverge enough to cause learner errors.\n"
    "\n"
    "[LEARNER-CENTRIC GOALS]\n"
    "- Aim for a few clear, teachable uses that learners can reliably distinguish in real usage.\n"
    "- Differences between clusters should be visible in collocations, argument patterns, and contexts (reflected in the definitions).\n"
    "- Order clusters by teaching priority: core/high-frequency → common extensions → domain-specific/rare.\n"
    "\n"
    "[CLUSTERING PRINCIPLES]\n"
    "A) Semantic frame & arguments first: group senses sharing the same event/frame and similar argument patterns (typical objects, prepositions, who-does-what).\n"
    "B) Literal vs figurative: split when collocations/syntax differ; merge only if learner usage is essentially the same.\n"
    "C) Domain-specific: finance/legal/technical uses should be dedicated clusters.\n"
    "D) Same usage, small differences (intensity/register/style): merge. Like a verb can be used for an abstract concept or a concrete thing, but both are closely related. Minor surface distinctions stay in one cluster; mention the scenarios inside english_def / chinese_def. Reserve splitting for cases that would mislead learners.\n"
    "E) Process vs result; causative vs state: merge if interchangeable for learners; split if the difference affects usage and error risk.\n"
    "\n"
    "[GUIDANCE FOR FILLING FIELDS]\n"
    "- english_def / chinese_def: Learner-facing paraphrases (do not copy input defs). Include useful usage cues (typical collocations, common objects, argument structure, scenarios). "
    "Keep english_def in English and chinese_def in Simplified Chinese.\n"
    "- chinese_label: A memorable Chinese literal translation capturing the cluster’s core use (flashcard-friendly).\n"
    "- english_label: Short English explanatory Phrases, near Synonyms, aligned with chinese_label.\n"
    "(e.g., \"operate/stop\", not \"go (operate/stop)\"; \"extend/lead\", not \"go (extend/lead)\").\n"
    "- chinese_criteria: Write in Simplified Chinese for **internal auditing only** (not learner-facing). Give decision cues for membership/boundaries "
    "(e.g., inclusion/exclusion heuristics, diagnostic hints). Do **not** include examples, collocations, or teaching tips here—put such cues in the defs.\n"
    "- ids / pos: Fill according to the schema.\n"
    "\n"
    "[CONFLICT RESOLUTION]\n"
    "- Assign borderline senses to the cluster that maximizes contrast and reduces learner confusion.\n"
    "- If a sense spans multiple frames, prefer the more common/transferable frame for Chinese learners unless the input indicates otherwise.\n"
    "- When collocations/arguments clearly signal a cluster, treat that as primary evidence (reflect these cues in the defs).\n"
)

def load_api_key(env_path: Path) -> Optional[str]:
    """从 .env 文件读取 OPENAI_API_KEY（若存在则写回环境变量）。"""
    if not env_path.exists():
        return None

    key = None
    with env_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            # 忽略空行以及注释行（以 # 开头），保持 .env 常规写法兼容
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if key:
        # 如果 .env 中定义了 key，将其写回环境变量，供 openai.OpenAI 默认读取
        os.environ.setdefault("OPENAI_API_KEY", key)
    return key


def parse_key(key: str) -> Tuple[str, str]:
    """解析 JSON 中的键，将字符串形式的 [kind, label] 拆为二元组。"""
    content = key.strip()[1:-1]
    kind, label = content.split(", ", 1)
    return kind, label


def build_prompt(key: str, rows: Iterable[dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    根据指定的 key 构造提示词，同时返回经过清洗的行数据。

    rows_payload 是给模型看的原子信息，prompt 则是最终发送的字符串。
    """
    kind, label = parse_key(key)

    # rows_payload：模型真正需要的细粒度条目列表。
    # 为什么要重新组织？
    #   - DictReader 读出的行可能带有多余字段，我们只保留 id/pos/def 三个核心信息。
    #   - 避免模型被无关字段干扰，同时也方便失败时把干净数据写回日志。
    rows_payload = [
        {
            "id": row.get("id", ""),
            "pos": row.get("pos", ""),
            "def": row.get("def", ""),
        }
        for row in rows
    ]

    # 将 rows_payload 序列化为易读的 JSON，直接嵌入提示词。
    # 由于模型会读取整段文本，保持缩进有助于 AI 解析，而 ensure_ascii=False 确保非 ASCII 字符（如中文）不被转义。
    rows_text = json.dumps(rows_payload, ensure_ascii=False, indent=2)
    prompt = (
        f"Fine-grained key: {kind}:{label}\n"
        "Each entry includes an id, part of speech, and definition.\n"
        f"{rows_text}\n"
    )
    return prompt, rows_payload


def call_openai_with_retry(
    *,
    instructions: str,
    prompt: str,
    json_schema: dict[str, object],
    timeout: Optional[float] = None,
    additional_retry: int = 1,
) -> dict[str, object]:
    """
    调用 OpenAI Responses API，并在 SDK 默认重试之外额外提供手动重试。
    为避免线程安全问题，每次调用都重新实例化 client。
    """
    attempts = additional_retry + 1
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            # 这里每次调用都创建独立的 OpenAI 客户端，避免线程之间共享会话导致的潜在竞态。
            client = OpenAI()
            response = client.responses.create(
                model="gpt-5-mini",
                instructions=instructions,
                input=prompt,
                text={"format": json_schema},
                service_tier="flex",
                timeout=timeout,
            )
            # 正常返回后解析字符串为 Python dict；
            # 如果模型因为某些原因返回了非 JSON（极少见），会被 json.loads 捕获并进入异常处理。
            return json.loads(response.output_text)
        except (APIError, APITimeoutError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            # 使用线性退避：第一次睡 1 秒，第二次睡 2 秒……以缓解短暂限流或网络抖动。
            time.sleep(1 + attempt)

    assert last_error is not None
    raise last_error


def process_one(
    index: int,
    key: str,
    rows: List[Dict[str, str]],
    *,
    instructions: str,
    json_schema: dict[str, object],
    timeout: Optional[float],
    retries: int,
) -> Tuple[int, str, Dict[str, object]]:
    """
    处理单个 [kind, label] 任务。
    返回值：
        (index, 'ok', enriched_dict)  表示成功，携带模型返回结果。
        (index, 'err', failure_record) 表示失败，携带原始行及错误信息。
    """
    prompt, rows_payload = build_prompt(key, rows)
    kind, label = parse_key(key)

    try:
        # 如果成功拿到模型输出，就拼装上原始 kind/label，形成最终结构化结果。
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
            **parsed,
        }
        return index, "ok", enriched
    except Exception as exc:
        # 捕获所有异常，将原始行与错误原因打包，便于后续人工排查。
        failure_record: Dict[str, object] = {
            "kind": kind,
            "label": label,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "rows": rows_payload,
        }
        return index, "err", failure_record


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数，统一管理线程数、重试次数、输入输出等配置。"""
    parser = argparse.ArgumentParser(
        description="Build coarse-grained sense clusters with parallel model calls (ordered writes)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fine-coarse") / "fine_unit_rows_multiple.json",
        help="Source JSON with multi-row entries (default: fine-coarse/fine_unit_rows_multiple.json)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env holding OPENAI_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--only-key",
        type=str,
        default=None,
        help="Process only this single key, e.g. \"[word, abstract]\". If omitted, process all keys.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit to the first N keys (after filtering by --only-key if provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses.jsonl",
        help="Append JSON lines with model outputs (default: fine-coarse/coarse_senses.jsonl)",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("fine-coarse") / "coarse_senses_failed.jsonl",
        help="Append JSON lines when calls fail (default: fine-coarse/coarse_senses_failed.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max number of parallel threads (default: 10). Tune to match API rate limits.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-call timeout in seconds passed to the API client (default: 90).",
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

    # 1. 读取 API key（若配置在 .env 中），保证后续线程能直接读取到凭据。
    load_api_key(args.env)

    # 2. 读取细粒度词典数据。该 JSON 结构通常为 { "[kind, label]": [ {id,...}, ... ], ... }。
    #    使用 UTF-8 打开，确保中文定义不会乱码。
    with args.input.open(encoding="utf-8") as handle:
        data: Dict[str, List[Dict[str, str]]] = json.load(handle)

    # 3. 根据用户参数决定要处理哪些 key：
    #    - 若指定 only_key，则只处理单个条目；
    #    - 否则保留 JSON 中原始 key 顺序，避免输出与输入错位。
    if args.only_key is not None:
        keys: List[str] = [args.only_key]
        if args.only_key not in data:
            print(f"No entry found for key {args.only_key!r}")
            return
    else:
        # dict 在 Python 3.7+ 默认保持插入顺序，直接 list() 即可
        keys = list(data.keys())
        # 只保留目标词列表中的 word 条目，保持原顺序
        target_set = {label.lower() for label in TARGET_WORD_LABELS}
        filtered_keys: List[str] = []
        for key in keys:
            try:
                kind, label = parse_key(key)
            except ValueError:
                continue
            if kind == "word" and label.lower() in target_set:
                filtered_keys.append(key)
        if not filtered_keys:
            print("No matching word entries found for the target label list.")
            return
        # 提醒哪些目标缺失
        existing_labels = {parse_key(k)[1].lower() for k in filtered_keys}
        missing = [label for label in TARGET_WORD_LABELS if label.lower() not in existing_labels]
        if missing:
            print("Warning: missing labels in input JSON:", ", ".join(missing))
        keys = filtered_keys

    # 4. limit 参数用于快速抽样或断点续跑，避免一次跑完整个数据集
    if args.limit is not None:
        keys = keys[: args.limit]

    total = len(keys)
    if total == 0:
        print("No keys to process.")
        return

    # 5. 准备输出文件夹；若文件夹不存在则自动创建。
    #    输出文件使用追加模式（append），可以多次运行脚本累积结果。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)
    # 使用 with 确保文件在异常时也能正确关闭
    with args.output.open("a", encoding="utf-8") as fout, args.failed_output.open("a", encoding="utf-8") as ffail:
        # 提交并行任务
        print(f"Submitting {total} tasks with {args.workers} workers...")
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for idx, key in enumerate(keys):
                rows = data[key]
                # submit 会立即返回 Future，方便后面异步收集；
                # 传入 index 用于保持原始顺序。
                fut = executor.submit(
                    process_one,
                    idx,
                    key,
                    rows,
                    instructions=INSTRUCTIONS,
                    json_schema=JSON_SCHEMA,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                futures.append(fut)

            # 依赖 buffer 保证写入顺序：只有当前索引的结果准备好时才输出
            next_to_write = 0
            buffer: Dict[int, Tuple[str, Dict[str, object]]] = {}

            completed = 0
            for fut in as_completed(futures):
                # as_completed 会在任一任务完成时返回对应 Future，顺序完全取决于任务耗时。
                idx, status, payload = fut.result()
                buffer[idx] = (status, payload)

                while next_to_write in buffer:
                    # 只有当“期望写入的下一个索引”已经在 buffer 里时，才真正落盘；
                    # 这样即便任务 A 比任务 B 慢，也会等待 B 写完前面的索引，输出顺序与原始 JSON 一致。
                    status2, payload2 = buffer.pop(next_to_write)
                    if status2 == "ok":
                        # 成功结果追加到目标 JSONL
                        fout.write(json.dumps(payload2, ensure_ascii=False))
                        fout.write("\n")
                        fout.flush()
                    else:
                        # 失败结果写入失败日志，后续可重新跑或人工审查
                        ffail.write(json.dumps(payload2, ensure_ascii=False))
                        ffail.write("\n")
                        ffail.flush()
                    completed += 1
                    print(f"[{completed}/{total}] wrote index {next_to_write + 1} ({'OK' if status2=='ok' else 'FAIL'})")
                    next_to_write += 1

    print(f"Done. Appended outputs to {args.output} and failures to {args.failed_output}.")


if __name__ == "__main__":
    main()
