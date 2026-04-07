#!/usr/bin/env python3
"""
Transcript 到学习结构化结果再到 coarse_unit 映射的主流程脚本。

这个文件是 `agent处理流程.md` 的可执行实现。

整体执行顺序：
1. 读取原始 transcript JSON，并判断是否需要续跑。
2. 按 3 句一批顺序处理。
3. 每个批次依次执行：
   - 阶段 0：清洗出给第一阶段 LLM 的最小输入
   - 阶段 1：由 LLM 生成学习导向的结构化 token
   - 阶段 2：代码侧校验 token 覆盖与顺序，并补上 token.index
   - 阶段 3：通过受控搜索把 token 映射到 `semantic.coarse_unit`
4. 把当前批次结果合并进累计结果。
5. 通过原子替换写出结果，避免中断时留下半截 JSON。

用法：
  python3 5agent-mapping.py <input_json> <output_json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError


DEFAULT_BATCH_SIZE = 3
DEFAULT_STAGE_ONE_MODEL = "gpt-5.4-mini"
DEFAULT_STAGE_ONE_REASONING_EFFORT = "medium"
DEFAULT_STAGE_THREE_MODEL = "gpt-5.4-nano"
DEFAULT_STAGE_THREE_REASONING_EFFORT = "high"
ROOT_DIR = Path(__file__).resolve().parent
QUERY_SCRIPT_PATH = ROOT_DIR / "supabase" / "query_coarse_units.py"
DIRECT_NO_MATCH_BASEFORMS = {
    "a", "an", "the",
    "and", "or", "but", "if",
    "as", "than", "that", "which", "who", "whom", "whose",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "can", "could", "may", "must",
    "not", "no", "yes",
    "he", "his", "him",
    "she", "her", "hers",
    "they", "their", "theirs", "them",
    "we", "our", "ours", "us",
    "you", "your", "yours",
    "it", "its",
    "i", "me", "my", "mine",
    "this", "these", "that", "those",
    "there", "here",
    "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "up", "down", "out", "off",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
}
DIRECT_NO_MATCH_REASON = (
    "当前 token 属于过于简单、极高频、低学习价值的基础功能词或代词，"
    "不进入 coarse_unit 映射，直接按 no_match 处理。"
)


def log_header(title: str) -> None:
    """打印一级标题，便于在长时间运行时快速定位当前阶段。"""

    print(f"\n=== {title} ===", flush=True)


def log_step(message: str, indent: int = 0) -> None:
    """打印带缩进的步骤日志。"""

    prefix = "  " * indent
    print(f"{prefix}{message}", flush=True)


def shorten_text(value: str, limit: int = 48) -> str:
    """缩短过长文本，避免命令行进度输出过宽。"""

    normalized = normalize_whitespace(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def count_tokens_in_batch(batch_output: dict[str, Any]) -> int:
    """统计一个批次中 token 总数，用于进度打印。"""

    return sum(len(sentence.get("tokens", [])) for sentence in batch_output.get("sentences", []))

STAGE_ONE_PROMPT = """
请将英文文本进行结构化分析，并严格按输入 token 顺序完成语言分片与解释生成。
你必须遵守以下规则：

1. 对输入批次中的每个句子都输出一个整句中文翻译 explanation。
2. 以输入 tokens 为唯一基础单位合并为有意义的语言元素分片; 只能将相邻 tokens 合并，不能改写文本。
3. 输出必须完整覆盖所有输入 tokens, 输出 token 之间不得重叠、不得跳词、不得打乱顺序。
4. 合并目标以英语学习为导向：
   - 固定短语、常见语法搭配的结构优先合并
   - 普通词组不要合并, 保持单个单词独立性
   - 以一个核心单词为主体的词语用法或者语法搭配（如"be addicted to"），应该合并为一个token，在explanation中同时解释短语含义和核心单词，semanticElement.baseForm应为核心词的原形（如"addicted"）。
6. 对每个输出 token：
   - 提供符合上下文语境的中文翻译和相关解释 explanation
   - 生成 semanticElement.baseForm 单词的原形（running -> run，studies -> study), 或者短语无时态形式(looking forward to -> look forward to)
   - 生成 semanticElement.dictionary 上下文无关词典释义
7. 你必须保持输出句子和 tokens 顺序与输入完全一致。

正面例子(建议合并)：
- "by the way" -> 固定短语
- "deal with" -> 固定搭配
- "at the same time" -> 固定短语
- "look forward to" -> 固定搭配
- "break down" -> 固定搭配
- "be addicted to" -> explanation: "对...上瘾；addicted表示沉迷的、上瘾的"，baseForm: "addicted"

反面例子(不要合并)：
- math class
- in the office
- software development
- brains and beauty
"""

STAGE_THREE_SYSTEM_PROMPT = """

任务：根据当前单词或短语在具体句子里的意思，从给定 coarse_unit 候选中选择最匹配的一个具体义项。

coarse_unit 的含义：
- 它可以对应单词, 也可以对应短语
- 它表示的是某个单词或短语在具体语境中的一个具体意思

核心原则：
- 重点不是匹配表面字符串，而是匹配“这个词或短语在这里到底是什么意思”
- 如果当前候选里已经有可靠匹配，就选出最合适的那个 coarse_unit
- 如果当前候选还不足以确认匹配，就给出下一步更合适的搜索词

单词义项示例：
同一个单词如果意思差别很大，通常对应不同 coarse_unit。
- light
  - Turn on the light. -> 灯
  - This bag is light. -> 轻的
- run
  - I run every morning. -> 跑步
  - She runs the company. -> 经营、管理

短语义项示例：
- take off
  - The plane took off. -> 起飞
  - He took off his jacket. -> 脱下
- work out
  - I work out at the gym. -> 锻炼
  - We need to work out the problem. -> 解决、想出办法

匹配边界：
- 匹配时不需要过度纠结极细的词典颗粒度；只要当前语境下的核心意思足够接近，就可以认为是可靠匹配
- 重点是避免明显的语义错绑，而不是追求过度苛刻的细粒度区分

正面例子：
- legend 在语境里表示“传奇人物”时，如果当前候选更接近“传奇 / 被广泛传颂的对象”这一核心意思，也可以接受
- 可以接受同一具体意思下、颗粒度略有粗细差别的匹配

反面例子：
- 这里不是做近义词联想匹配，也不是只要主题相关就可以匹配
- 不接受相关概念、主题相关或近义词替代式的匹配
- sexuality 不能匹配到 sex appeal
- economic policy 不能直接匹配到 economy
- angry 不能直接匹配到 upset
"""

STAGE_THREE_RULES_PROMPT = """
请按下面的逻辑完成判断。

你该做什么：
- 你只处理当前这一个 token，当前处理对象由 sentence.index + token.index 唯一确定
- 你要判断：当前 token 在这个具体语境里的意思，是否已经能可靠映射到某个 coarse_unit

什么情形下返回 `match`：
- 当前候选里已经有语义可靠、最贴合当前语境的 coarse_unit
- 你可以明确判断“这个词或短语在这里就是这个意思”
- 返回 `match` 时，只返回 coarse_id 和 reason
- 如果你认为有必要优化 token 的 explanation，也可以同时返回完整优化后的 explanation

什么情形下返回 `search`：
- 当前候选还不足以确认匹配
- 但你认为继续搜索仍然有意义，且当前 token 更像是一个需要拆开的词组或屈折形式，而不是一个已经很明确的单词

`search` 的用法：
- `search` 的作用是告诉系统：下一回应该继续精确查询哪些词
- 对大部分单词，第一次精确搜索就已经足够；可以直接 match/no_match，通常不应再返回 `search`
- 词组或变形表达，才适合继续搜索更短、更核心或更标准的精确查询词
- 新查询词应尽量朝“核心词 / 更标准写法 / 去掉不必要修饰”收缩，而不是保留整段原短语继续绕圈搜索
- 不要把整句里的额外修饰词一起带入搜索, 也不允许搜索近义词, 类似意思的短语

- 合理例子：
  - `be addicted to` -> 可以继续查 `addicted to`、`addicted`、`addict`
  - `setting up` -> 可以继续查 `set up`、`set`
- 不合理例子：
  - `sexuality` 第一次精确搜索通常就应直接 `match/no_match`，不应继续扩展成别的词
  - `economic` 第一次精确搜索通常也应直接 `match/no_match`
- `search.queries`
  - 应该是你建议系统继续做精确匹配的词或短语
  - 最多提供 4 个

什么情形下返回 `no_match`：
- 已经到最后一回，仍然没有语义可靠的候选
- 当前候选虽然表面相似，但你无法确认意思一致
- 继续搜索也不太可能得到更可靠结果
- 返回 `no_match` 的意思是：当前 token 不应该绑定到任何 coarse_unit

统一输出要求：
- 只输出一个 JSON 对象
- 不要输出 Markdown
- 不要输出额外解释
- 不要使用未要求字段
"""


class StageOneSemanticElement(BaseModel):
    """第一阶段产出的语义字段，此时还没有做数据库映射。"""

    baseForm: str
    dictionary: str


class StageOneToken(BaseModel):
    """第一阶段返回的学习导向 token。"""

    text: str
    explanation: str
    semanticElement: StageOneSemanticElement


class StageOneSentence(BaseModel):
    """第一阶段返回的一条句子结果。"""

    index: int
    text: str
    explanation: str
    tokens: list[StageOneToken]


class StageOneOutput(BaseModel):
    """第一阶段的结构化输出约束。"""

    sentences: list[StageOneSentence]


class StageThreeDecision(BaseModel):
    """
    第三阶段的结构化决策约束。

    LLM 在第三阶段只允许做三种事：
    - `match`：从当前数据库候选里选一个 coarse_id
    - `search`：给出下一回搜索建议
    - `no_match`：在继续搜索意义不大或非法重试后结束当前 token
    """

    action: Literal["match", "search", "no_match"]
    coarse_id: int | None = None
    queries: list[str] | None = None
    reason: str
    explanation: str | None = None


@dataclass
class SearchRoundRecord:
    """第三阶段中某个 token 的一回具体数据库搜索记录。"""

    round_no: int
    mode: str
    queries: list[str]
    results: dict[str, Any]

    @property
    def candidate_count(self) -> int:
        return len(flatten_candidates(self.results))


@dataclass
class TokenRuntime:
    """
    第三阶段当前正在处理的 token 的运行时视图。

    文档要求第三阶段一次对话只处理一个 token，并由
    `sentence.index + token.index` 唯一确定。这个对象就是控制器侧对
    该处理单元的表示。
    """

    sentence_index: int
    token_index: int
    sentence_text: str
    token: dict[str, Any]


class CoarseQueryRunner:
    """
    对 `supabase/query_coarse_units.py` 的轻量封装。

    文档明确要求数据库访问必须由脚本执行，而不是由 LLM 自行完成。
    这个类就是第三阶段控制器侧的查询执行器。
    """

    def __init__(self, repo_root: Path, script_path: Path) -> None:
        self.repo_root = repo_root
        self.script_path = script_path

    def run(self, mode: str, queries: list[str]) -> dict[str, Any]:
        """
        执行一回搜索。

        这里的一次调用就等于文档里的“一回搜索”。查询脚本本身已经处理了
        4 个参数上限和 `status=active` 过滤；这里先做去重，保证请求尽量
        干净且确定。
        """

        normalized_queries = dedupe_queries(queries)
        if not normalized_queries:
            return {"results": []}

        completed = subprocess.run(
            ["python3", str(self.script_path), mode, *normalized_queries],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or "results" not in payload:
            raise ValueError("query_coarse_units.py returned an invalid payload")
        return payload


class AuditLogger:
    """
    追加写入 token 级别的搜索审计记录。

    搜索审计日志保存在输出目录下的 `log/` 中，并在流程结束后保留。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transcript agent mapping pipeline.")
    parser.add_argument("input_json", type=Path, help="Path to the original transcript JSON.")
    parser.add_argument("output_json", type=Path, help="Path to the final output JSON.")
    parser.add_argument(
        "--env-path",
        type=Path,
        default=ROOT_DIR / ".env",
        help="Path to .env containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of sentences per batch. Default is 3.",
    )
    parser.add_argument(
        "--stage-one-model",
        default=DEFAULT_STAGE_ONE_MODEL,
        help="OpenAI model for stage one.",
    )
    parser.add_argument(
        "--stage-one-reasoning-effort",
        default=DEFAULT_STAGE_ONE_REASONING_EFFORT,
        help="Reasoning effort for stage one.",
    )
    parser.add_argument(
        "--stage-three-model",
        default=DEFAULT_STAGE_THREE_MODEL,
        help="OpenAI model for stage three.",
    )
    parser.add_argument(
        "--stage-three-reasoning-effort",
        default=DEFAULT_STAGE_THREE_REASONING_EFFORT,
        help="Reasoning effort for stage three.",
    )
    return parser.parse_args()


def load_openai_api_key(env_path: Path) -> str:
    """从 `.env` 读取 OpenAI API Key。"""

    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY not found in {env_path}")
    return api_key


def create_structured_llm(
    env_path: Path,
    model_name: str,
    reasoning_effort: str,
    schema: type[BaseModel],
) -> Any:
    """创建一个绑定结构化输出 schema 的 OpenAI 聊天模型。"""

    api_key = load_openai_api_key(env_path)

    llm = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        reasoning_effort=reasoning_effort,
    )
    return llm.with_structured_output(schema)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_whitespace(value: str) -> str:
    """压缩多余空白，保证 token 对齐和查询去重更稳定。"""

    return " ".join(value.split())


def normalize_lookup_text(value: str) -> str:
    """把 token.text / baseForm 归一化成便于停用词判断的键。"""

    lowered = normalize_whitespace(value).lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", lowered)


def dedupe_queries(queries: list[str]) -> list[str]:
    """按首次出现顺序对查询词去重。"""

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_query in queries:
        query = normalize_whitespace(raw_query.strip())
        if not query or query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    return deduped


def normalize_llm_search_query(value: str) -> str:
    """
    清洗 LLM 返回的 search query。

    用户要求仅对模型返回的 search query 做这一步：
    1. 全部转小写
    2. 如果首字符或尾字符不是字母/数字，则删除
    3. 再做空白压缩
    """

    normalized = normalize_whitespace(value).lower().strip()
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", normalized)


def normalize_llm_search_queries(queries: list[str]) -> list[str]:
    """按首次出现顺序清洗并去重 LLM 返回的 search queries。"""

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_query in queries:
        query = normalize_llm_search_query(raw_query)
        if not query or query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    return deduped


def chunk_sentences(sentences: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """把输入句子按顺序切成批次，默认每批 3 句。"""

    return [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]


def build_stage_zero_batch(batch_sentences: list[dict[str, Any]]) -> dict[str, Any]:
    """
    阶段 0：构造发给第一阶段 LLM 的最小 JSON。

    文档要求：
    - 保留 `sentence.index`
    - 保留 `sentence.text`
    - 只把 token 文本发给 LLM
    - 原始 token 顺序继续由代码保留，供后续校验使用
    """

    cleaned_sentences = []
    for sentence in batch_sentences:
        cleaned_sentences.append(
            {
                "index": sentence["index"],
                "text": sentence["text"],
                "tokens": [token["text"] for token in sentence.get("tokens", [])],
            }
        )
    return {"sentences": cleaned_sentences}


def invoke_stage_one(
    stage_one_llm: Any,
    cleaned_batch: dict[str, Any],
) -> dict[str, Any]:
    """
    阶段 1：对一个批次调用 LLM。

    发送给模型的内容只有：
    - 固定的第一阶段规则 prompt
    - 阶段 0 产出的清洗后批次 JSON

    返回结果会先通过 `StageOneOutput` 的结构化约束校验，再继续进入阶段 2。
    """

    messages = [
        SystemMessage(content=STAGE_ONE_PROMPT),
        HumanMessage(content=json.dumps(cleaned_batch, ensure_ascii=False, indent=2)),
    ]
    result = stage_one_llm.invoke(messages)
    if isinstance(result, StageOneOutput):
        return result.model_dump()
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return StageOneOutput.model_validate(result).model_dump()
    raise TypeError("Unexpected StageOneOutput result type")


def validate_and_index_batch(
    original_batch: list[dict[str, Any]],
    stage_one_output: dict[str, Any],
) -> dict[str, Any]:
    """
    阶段 2：在代码里校验第一阶段结果，并补上 token.index。

    这里实现的是文档要求的“严格从左到右对齐”：
    - 句子数量和顺序必须一致
    - sentence.text 和 sentence.index 必须一致
    - 输出 token 必须按顺序完整覆盖原始 token 列表
    - 不允许漏 token
    - 不允许重复 token
    - 不允许跳 token

    校验通过后，代码会从 0 开始补 `tokens[].index`。
    """

    output_sentences = stage_one_output.get("sentences", [])
    if len(output_sentences) != len(original_batch):
        raise ValueError("Stage one sentence count does not match input batch")

    validated_sentences: list[dict[str, Any]] = []
    for original_sentence, output_sentence in zip(original_batch, output_sentences, strict=True):
        if output_sentence.get("index") != original_sentence["index"]:
            raise ValueError("sentence.index mismatch after stage one")
        if output_sentence.get("text") != original_sentence["text"]:
            raise ValueError("sentence.text mismatch after stage one")

        # 这里刻意使用严格的游标式对齐，而不是模糊匹配。
        # 文档的目标是：根据第一阶段合并后的 token 文本，确定性地恢复
        # 它对应的原始 token 范围。
        original_tokens = [token["text"] for token in original_sentence.get("tokens", [])]
        cursor = 0
        validated_tokens: list[dict[str, Any]] = []

        for token_idx, output_token in enumerate(output_sentence.get("tokens", [])):
            token_text = normalize_whitespace(str(output_token.get("text", "")).strip())
            if not token_text:
                raise ValueError("stage one returned an empty token text")

            matched = False
            # 这里只允许连续的前向合并：original_tokens[cursor:end]。
            # 这样就能直接保证“不重叠 / 不跳词 / 不乱序”。
            for end in range(cursor + 1, len(original_tokens) + 1):
                candidate = normalize_whitespace(" ".join(original_tokens[cursor:end]))
                if candidate == token_text:
                    validated_token = json.loads(json.dumps(output_token, ensure_ascii=False))
                    validated_token["index"] = token_idx
                    matched = True
                    cursor = end
                    validated_tokens.append(validated_token)
                    break

            if not matched:
                raise ValueError(
                    f"cannot align output token '{token_text}' in sentence {original_sentence['index']}"
                )

        if cursor != len(original_tokens):
            raise ValueError(
                f"sentence {original_sentence['index']} does not fully cover original tokens"
            )

        validated_sentence = {
            "index": output_sentence["index"],
            "text": output_sentence["text"],
            "explanation": output_sentence["explanation"],
            "tokens": validated_tokens,
        }
        validated_sentences.append(validated_sentence)

    return {"sentences": validated_sentences}


def run_stage_one_with_retry(
    stage_one_llm: Any,
    original_batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    对一个批次执行阶段 0 + 1 + 2，并在失败时重试两次。

    文档要求：
    - 如果校验失败，第一阶段重试两次
    - 如果第一阶段 LLM 请求失败，也算一次失败
    - 如果仍失败，则整个流程终止
    - 失败批次不允许进入阶段 3
    """

    cleaned_batch = build_stage_zero_batch(original_batch)
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            stage_one_output = invoke_stage_one(stage_one_llm, cleaned_batch)
            return validate_and_index_batch(original_batch, stage_one_output)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                retry_index = attempt + 1
                log_step(f"[阶段 1/2] 第 {retry_index} 次执行失败，准备重试：{exc}", indent=1)
                continue
            raise

    raise RuntimeError(f"stage one failed unexpectedly: {last_error}")


def build_round_payload(record: SearchRoundRecord) -> dict[str, Any]:
    """把一回搜索记录转换成发给 LLM 的 JSON 载荷。"""

    return {
        "search_round": record.round_no,
        "search_mode": record.mode,
        "queries": record.queries,
        "candidates": record.results,
    }


def build_token_payload(token_runtime: TokenRuntime) -> dict[str, Any]:
    """构造第三阶段要求的“当前 token”明确载荷。"""

    return {
        "sentence_index": token_runtime.sentence_index,
        "token_index": token_runtime.token_index,
        "sentence_text": token_runtime.sentence_text,
        "token": token_runtime.token,
    }


def format_token_progress(
    token_runtime: TokenRuntime,
    token_position: int,
    total_tokens: int,
) -> str:
    """生成命令行中的 token 进度标签。"""

    token_text = shorten_text(str(token_runtime.token.get("text", "")))
    return (
        f"[Token {token_position}/{total_tokens}] "
        f"句子 {token_runtime.sentence_index} / token {token_runtime.token_index} / {token_text}"
    )


def build_current_round_instruction(round_no: int) -> str:
    """构造“当前这一轮是什么输入、允许返回什么”的动态说明。"""

    if round_no == 1:
        return """当前这一轮的输入：
- 你会看到当前 token
- 你会看到当前句子
- 你会看到系统已经自动执行的第 1 回搜索结果：token.text + baseForm

这一轮允许的输出：
- 如果当前候选已经足够可靠，返回 `match`
- 如果当前候选还不够，但继续搜索有意义，返回 `search`
- 如果你已经判断继续搜索意义不大，也可以直接返回 `no_match`

"""

    if round_no == 2:
        return """当前这一轮的输入：
- 你会看到当前 token
- 你会看到当前句子
- 你会看到第 1 回搜索历史
- 你会看到第 2 回搜索结果

这一轮允许的输出：
- 如果当前候选已经足够可靠，返回 `match`
- 如果当前候选还不够，但继续搜索有意义，返回 `search`
- 如果你已经判断继续搜索意义不大，也可以直接返回 `no_match`
"""

    return """当前这一轮的输入：
- 你会看到当前 token
- 你会看到当前句子
- 你会看到前两回搜索历史
- 你会看到第 3 回搜索结果

这一轮允许的输出：
- 如果当前候选已经足够可靠，返回 `match`
- 如果仍然无法可靠匹配，返回 `no_match`
- 不要返回 `search`
"""


def build_stage_three_messages(
    shared_context: dict[str, Any],
    token_runtime: TokenRuntime,
    rounds: list[SearchRoundRecord],
    retry_note: str | None = None,
) -> list[Any]:
    """
    为第三阶段 LLM 显式构造一次 `messages` 请求。

    这里就是文档中分支上下文策略的具体实现：
    - 固定前缀 `H`
    - 共享批次上下文 `A`
    - 当前 token 载荷
    - 仅属于当前 token 的历史轮次
    - 当前轮次载荷

    当控制器切换到下一个 token 时，会重新从同一个 `H + A` 开始构造，
    不会带上前一个 token 的历史。
    """

    history_messages = [
        HumanMessage(
            content="历史搜索回合：\n"
            + json.dumps(build_round_payload(record), ensure_ascii=False, indent=2)
        )
        for record in rounds[:-1]
    ]

    messages: list[Any] = [
        SystemMessage(content=STAGE_THREE_SYSTEM_PROMPT),
        HumanMessage(content=STAGE_THREE_RULES_PROMPT),
        HumanMessage(content=build_current_round_instruction(rounds[-1].round_no)),
        HumanMessage(
            content="第二阶段当前批次 JSON：\n"
            + json.dumps(shared_context, ensure_ascii=False, indent=2)
        ),
        HumanMessage(
            content="当前处理 token：\n"
            + json.dumps(build_token_payload(token_runtime), ensure_ascii=False, indent=2)
        ),
        *history_messages,
        HumanMessage(
            content="当前搜索回合：\n"
            + json.dumps(build_round_payload(rounds[-1]), ensure_ascii=False, indent=2)
        ),
    ]

    if retry_note:
        messages.append(
            HumanMessage(
                content=f"上一条响应无效：{retry_note}\n请基于相同上下文重新返回合法结构化结果。"
            )
        )

    return messages


def flatten_candidates(results_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """
    把查询脚本返回结果拍平成以 coarse_id 为键的候选表。

    第三阶段要求 LLM 只能从当前轮候选里选 id，所以控制器先把候选整理成
    便于 O(1) 查找的字典。
    """

    candidates_by_id: dict[int, dict[str, Any]] = {}
    for result_entry in results_payload.get("results", []):
        for row in result_entry.get("rows", []):
            row_id = row.get("id")
            if isinstance(row_id, int):
                candidates_by_id[row_id] = row
    return candidates_by_id


def invoke_stage_three_decision(
    stage_three_llm: Any,
    shared_context: dict[str, Any],
    token_runtime: TokenRuntime,
    rounds: list[SearchRoundRecord],
    retry_note: str | None = None,
) -> StageThreeDecision:
    """针对当前 token 和当前轮次调用第三阶段决策 LLM。"""

    messages = build_stage_three_messages(shared_context, token_runtime, rounds, retry_note)
    result = stage_three_llm.invoke(messages)
    if isinstance(result, StageThreeDecision):
        return result
    if isinstance(result, BaseModel):
        return StageThreeDecision.model_validate(result.model_dump())
    return StageThreeDecision.model_validate(result)


def validate_stage_three_decision(
    decision: StageThreeDecision,
    rounds: list[SearchRoundRecord],
) -> tuple[bool, str]:
    """
    按工作流规则校验一条第三阶段决策。

    关键校验点：
    - `reason` 必须非空
    - `match` 的 id 必须存在于当前候选中
    - `search` 不再区分 mode，只允许继续给出精确查询词
    - `no_match` 允许提前出现，用于简单高频且无需映射的 token
    """

    current_round = rounds[-1]
    current_candidates = flatten_candidates(current_round.results)
    reason = decision.reason.strip()

    if not reason:
        return False, "reason must be a non-empty string"

    if decision.action == "match":
        if decision.coarse_id is None:
            return False, "match requires coarse_id"
        if decision.coarse_id not in current_candidates:
            return False, "match coarse_id is not present in current round candidates"
        return True, ""

    if decision.action == "search":
        if current_round.round_no not in {1, 2}:
            return False, "search is not allowed after round 3"

        normalized_queries = normalize_llm_search_queries(decision.queries or [])
        if not normalized_queries:
            return False, "search requires at least one non-empty query"
        if len(normalized_queries) > 4:
            return False, "search query count must be at most 4 after dedupe"
        decision.queries = normalized_queries
        return True, ""

    if decision.action == "no_match":
        return True, ""

    return False, "unsupported action"


def ask_stage_three_with_retry(
    stage_three_llm: Any,
    shared_context: dict[str, Any],
    token_runtime: TokenRuntime,
    rounds: list[SearchRoundRecord],
) -> StageThreeDecision:
    """
    调用第三阶段 LLM；如果响应非法，则当前轮重试一次。

    这对应文档里的要求：
    - 结构化输出非法时重试一次
    - `coarse_id` 非法时重试一次
    - LLM 请求失败时也重试一次，并同样算一次失败
    - 若仍非法，则按失败流程走 `no_match`
    """

    retry_note: str | None = None
    last_error = "unknown decision error"

    for _ in range(2):
        try:
            decision = invoke_stage_three_decision(
                stage_three_llm,
                shared_context,
                token_runtime,
                rounds,
                retry_note=retry_note,
            )
        except Exception as exc:
            # LLM 请求失败也按当前轮的一次失败处理，并只再重试一次。
            last_error = f"当前轮 LLM 请求或结构化解析失败：{exc}"
            retry_note = last_error
            continue

        is_valid, error = validate_stage_three_decision(decision, rounds)
        if is_valid:
            return decision

        last_error = error
        retry_note = error

    return StageThreeDecision(
        action="no_match",
        reason=f"当前轮次 agent 输出非法，按失败流程处理：{last_error}",
    )


def ensure_semantic_element(token: dict[str, Any]) -> dict[str, Any]:
    """确保输出 token 上存在可修改的 semanticElement 对象。"""

    semantic_element = token.setdefault("semanticElement", {})
    if not isinstance(semantic_element, dict):
        raise ValueError("semanticElement must be an object")
    return semantic_element


def finalize_match(
    token_runtime: TokenRuntime,
    decision: StageThreeDecision,
    rounds: list[SearchRoundRecord],
) -> tuple[str, int | None, str]:
    """
    完成一次第三阶段成功匹配的回填。

    关键点：最终数据库真值不是由 LLM 自己写，而是由控制器读取当前匹配到的
    数据库候选后回填：
    - coarse_id <- candidate.id
    - baseForm <- candidate.label
    - dictionary <- candidate.chinese_def
    - reason <- decision.reason

    如果 LLM 同时返回了完整 explanation，则这里直接覆盖 token.explanation。
    """

    current_candidates = flatten_candidates(rounds[-1].results)
    matched = current_candidates[decision.coarse_id]
    semantic_element = ensure_semantic_element(token_runtime.token)
    semantic_element["coarse_id"] = matched["id"]
    semantic_element["baseForm"] = matched["label"]
    semantic_element["dictionary"] = matched.get("chinese_def") or semantic_element.get("dictionary", "")
    semantic_element["reason"] = decision.reason.strip()

    if decision.explanation and decision.explanation.strip():
        token_runtime.token["explanation"] = decision.explanation.strip()

    return "match", matched["id"], semantic_element["reason"]


def finalize_no_match(
    token_runtime: TokenRuntime,
    reason: str,
) -> tuple[str, int | None, str]:
    """
    完成一次第三阶段失败映射的回填。

    按文档要求：
    - coarse_id 设为 null
    - 第一阶段的 baseForm 和 dictionary 保持不变
    - reason 仍然必须写入
    """

    semantic_element = ensure_semantic_element(token_runtime.token)
    semantic_element["coarse_id"] = None
    semantic_element["reason"] = reason.strip()
    return "no_match", None, semantic_element["reason"]


def process_single_token(
    stage_three_llm: Any,
    query_runner: CoarseQueryRunner,
    audit_logger: AuditLogger,
    shared_context: dict[str, Any],
    token_runtime: TokenRuntime,
    token_position: int,
    total_tokens: int,
) -> None:
    """
    对单个 token 执行完整的第三阶段流程。

    轮次顺序：
    1. 自动 exact(token.text, baseForm)
    2. 如有需要，使用 agent 给出的 exact 查询词
    3. 如仍有需要，继续使用 agent 给出的 exact 查询词

    当前 token 结束后：
    - 写入一条审计记录
    - 返回上层，由上层切回新的 `H + A` 分支再处理下一个 token
    """

    token_progress = format_token_progress(token_runtime, token_position, total_tokens)
    log_step(token_progress, indent=1)

    semantic_element = ensure_semantic_element(token_runtime.token)
    lookup_values = {
        normalize_lookup_text(str(token_runtime.token.get("text", ""))),
        normalize_lookup_text(str(semantic_element.get("baseForm", ""))),
    }
    lookup_values.discard("")

    # 对于固定的一小批基础功能词 / 代词，不进入数据库查询和 LLM 判断，
    # 直接按 no_match 处理，减少无意义搜索。
    if lookup_values & DIRECT_NO_MATCH_BASEFORMS:
        log_step("[阶段 3] 命中直接 no_match 词表，跳过数据库查询和 LLM。", indent=2)
        final_action, final_coarse_id, final_reason = finalize_no_match(
            token_runtime,
            DIRECT_NO_MATCH_REASON,
        )
        audit_logger.write(
            {
                "sentence_index": token_runtime.sentence_index,
                "token_index": token_runtime.token_index,
                "token_text": token_runtime.token.get("text"),
                "final_action": final_action,
                "final_coarse_id": final_coarse_id,
                "final_reason": final_reason,
                "rounds": [],
            }
        )
        log_step(f"[完成] {final_action} | coarse_id=None", indent=2)
        return

    rounds: list[SearchRoundRecord] = []

    # 第一回永远是自动搜索。即使同时查 token.text 和 baseForm，
    # 在文档语义里也只算一回搜索。
    round1_queries = dedupe_queries(
        [
            str(token_runtime.token.get("text", "")),
            str(token_runtime.token.get("semanticElement", {}).get("baseForm", "")),
        ]
    )
    log_step(f"[搜索 1/3] exact -> {round1_queries}", indent=2)
    round1_record = SearchRoundRecord(
        round_no=1,
        mode="exact",
        queries=round1_queries,
        results=query_runner.run("exact", round1_queries),
    )
    rounds.append(round1_record)
    log_step(f"[搜索 1/3] 候选数：{round1_record.candidate_count}", indent=2)
    decision = ask_stage_three_with_retry(stage_three_llm, shared_context, token_runtime, rounds)
    log_step(f"[决策 1/3] {decision.action}", indent=2)

    if decision.action == "match":
        final_action, final_coarse_id, final_reason = finalize_match(token_runtime, decision, rounds)
    else:
        if decision.action == "search":
            log_step(f"[搜索 2/3] exact -> {decision.queries or []}", indent=2)
            round2_record = SearchRoundRecord(
                round_no=2,
                mode="exact",
                queries=decision.queries or [],
                results=query_runner.run("exact", decision.queries or []),
            )
            rounds.append(round2_record)
            log_step(f"[搜索 2/3] 候选数：{round2_record.candidate_count}", indent=2)
            decision = ask_stage_three_with_retry(stage_three_llm, shared_context, token_runtime, rounds)
            log_step(f"[决策 2/3] {decision.action}", indent=2)

        if decision.action == "match":
            final_action, final_coarse_id, final_reason = finalize_match(token_runtime, decision, rounds)
        else:
            if decision.action == "search":
                log_step(f"[搜索 3/3] exact -> {decision.queries or []}", indent=2)
                round3_record = SearchRoundRecord(
                    round_no=3,
                    mode="exact",
                    queries=decision.queries or [],
                    results=query_runner.run("exact", decision.queries or []),
                )
                rounds.append(round3_record)
                log_step(f"[搜索 3/3] 候选数：{round3_record.candidate_count}", indent=2)
                decision = ask_stage_three_with_retry(stage_three_llm, shared_context, token_runtime, rounds)
                log_step(f"[决策 3/3] {decision.action}", indent=2)

            if decision.action == "match":
                final_action, final_coarse_id, final_reason = finalize_match(token_runtime, decision, rounds)
            else:
                no_match_reason = decision.reason.strip()
                if not no_match_reason:
                    no_match_reason = "三回搜索后未得到可靠匹配。"
                final_action, final_coarse_id, final_reason = finalize_no_match(token_runtime, no_match_reason)

    log_step(
        f"[完成] {final_action} | coarse_id={final_coarse_id if final_coarse_id is not None else 'None'}",
        indent=2,
    )

    # 审计文件是追加写入，并且按文档要求在流程结束后保留。
    audit_logger.write(
        {
            "sentence_index": token_runtime.sentence_index,
            "token_index": token_runtime.token_index,
            "token_text": token_runtime.token.get("text"),
            "final_action": final_action,
            "final_coarse_id": final_coarse_id,
            "final_reason": final_reason,
            "rounds": [
                {
                    "round_no": record.round_no,
                    "mode": record.mode,
                    "queries": record.queries,
                    "candidate_count": record.candidate_count,
                    "results": record.results.get("results", []),
                }
                for record in rounds
            ],
        }
    )


def process_stage_three_batch(
    stage_three_llm: Any,
    query_runner: CoarseQueryRunner,
    audit_logger: AuditLogger,
    batch_output: dict[str, Any],
) -> dict[str, Any]:
    """
    对一个批次执行完整第三阶段。

    当前批次的 JSON 就是共享上下文 `A`。每个 token 都从同一个 `A`
    开始，但只继承属于自己这个 token 的轮次历史。
    """

    shared_context = json.loads(json.dumps(batch_output, ensure_ascii=False))
    total_tokens = count_tokens_in_batch(batch_output)
    processed_tokens = 0

    log_step(
        f"[阶段 3] 开始 coarse_unit 映射：{len(batch_output.get('sentences', []))} 句，{total_tokens} 个 token",
        indent=1,
    )

    for sentence in batch_output.get("sentences", []):
        log_step(
            f"[句子] index={sentence['index']} | {shorten_text(sentence['text'], limit=72)}",
            indent=1,
        )
        for token in sentence.get("tokens", []):
            processed_tokens += 1
            token_runtime = TokenRuntime(
                sentence_index=sentence["index"],
                token_index=token["index"],
                sentence_text=sentence["text"],
                token=token,
            )
            process_single_token(
                stage_three_llm=stage_three_llm,
                query_runner=query_runner,
                audit_logger=audit_logger,
                shared_context=shared_context,
                token_runtime=token_runtime,
                token_position=processed_tokens,
                total_tokens=total_tokens,
            )
    return batch_output


def ensure_output_dirs(output_path: Path) -> tuple[Path, Path]:
    """准备输出目录及其 `temp/`、`log/` 子目录。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "temp"
    log_dir = output_path.parent / "log"
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, log_dir


def get_intermediate_output_path(output_path: Path) -> Path:
    """返回批次级中间文件的固定保存路径。"""

    return output_path.parent / "temp" / output_path.name


def atomic_write_json(target_path: Path, temp_dir: Path, input_path: Path, payload: dict[str, Any]) -> None:
    """
    用原子替换方式写入任意 JSON 文件。

    - 中间文件：target_path 位于 temp/
    - 最终文件：target_path 位于目标目录
    - 原子写入时使用随机后缀临时文件，写完后再 replace
    """

    temp_path = temp_dir / f"{input_path.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_existing_output(output_path: Path) -> tuple[list[dict[str, Any]], int, str]:
    """
    读取已有输出，用于续跑。

    优先级：
    1. 正式输出文件
    2. temp/ 中的中间文件
    """

    intermediate_path = get_intermediate_output_path(output_path)

    source_path: Path | None = None
    source_label = "none"
    if output_path.exists():
        source_path = output_path
        source_label = "final"
    elif intermediate_path.exists():
        source_path = intermediate_path
        source_label = "intermediate"

    if source_path is None:
        return [], -1, source_label

    payload = load_json(source_path)
    sentences = payload.get("sentences", [])
    if not isinstance(sentences, list):
        raise ValueError("Existing output JSON does not contain a valid sentences array")
    if not sentences:
        return [], -1, source_label
    last_index = sentences[-1]["index"]
    return sentences, last_index, source_label


def validate_input_payload(input_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """在开始处理前校验输入 JSON 的最小必需结构。"""

    sentences = input_payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError("Input JSON must contain a sentences array")

    for sentence in sentences:
        if "index" not in sentence or "text" not in sentence or "tokens" not in sentence:
            raise ValueError("Each input sentence must contain index, text, and tokens")
        if not isinstance(sentence["tokens"], list):
            raise ValueError("sentence.tokens must be a list")
        for token in sentence["tokens"]:
            if "text" not in token:
                raise ValueError("Each input token must contain text")

    return sentences


def attach_timing_info(
    input_sentences: list[dict[str, Any]],
    output_sentences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    在正式输出前，按原始输入回填句子与 token 的 start/end 时间。

    规则：
    - 句子的 start/end 直接沿用原始输入句子
    - token 的 start/end 通过最终 token.text 再次顺序对齐原始 token 列表
    - 如果某个句子里某个 token 无法对齐，则当前句子中该 token 以及其后的 token 都写入空时间
    - 只在全部批次完成后的正式输出阶段执行，中间文件不写入这些时间字段
    """

    input_by_index = {sentence["index"]: sentence for sentence in input_sentences}
    timed_sentences: list[dict[str, Any]] = []

    for output_sentence in output_sentences:
        sentence_index = output_sentence["index"]
        input_sentence = input_by_index.get(sentence_index)
        if input_sentence is None:
            raise ValueError(f"Cannot find source sentence for index={sentence_index}")

        timed_sentence = json.loads(json.dumps(output_sentence, ensure_ascii=False))
        if "start" in input_sentence:
            timed_sentence["start"] = input_sentence["start"]
        if "end" in input_sentence:
            timed_sentence["end"] = input_sentence["end"]

        source_tokens = input_sentence.get("tokens", [])
        cursor = 0
        timed_tokens: list[dict[str, Any]] = []
        alignment_failed = False

        for output_token in timed_sentence.get("tokens", []):
            if alignment_failed:
                output_token["start"] = None
                output_token["end"] = None
                timed_tokens.append(output_token)
                continue

            token_text = normalize_whitespace(str(output_token.get("text", "")).strip())
            if not token_text:
                output_token["start"] = None
                output_token["end"] = None
                timed_tokens.append(output_token)
                alignment_failed = True
                continue

            matched = False
            for end in range(cursor + 1, len(source_tokens) + 1):
                candidate = normalize_whitespace(" ".join(token["text"] for token in source_tokens[cursor:end]))
                if candidate == token_text:
                    first_source = source_tokens[cursor]
                    last_source = source_tokens[end - 1]
                    if "start" in first_source:
                        output_token["start"] = first_source["start"]
                    if "end" in last_source:
                        output_token["end"] = last_source["end"]
                    timed_tokens.append(output_token)
                    cursor = end
                    matched = True
                    break

            if not matched:
                output_token["start"] = None
                output_token["end"] = None
                timed_tokens.append(output_token)
                alignment_failed = True

        timed_sentence["tokens"] = timed_tokens
        timed_sentences.append(timed_sentence)

    return timed_sentences


def create_final_payload(sentences: list[dict[str, Any]]) -> dict[str, Any]:
    """把累计句子列表包装成最终输出 JSON 结构。"""

    return {"sentences": sentences}


def main() -> None:
    """
    程序入口。

    这个函数把整份文档的流程串起来：
    - 加载环境变量与模型
    - 读取输入与已有输出
    - 计算剩余批次
    - 逐批执行阶段 0/1/2/3
    - 每个成功批次结束后合并并原子保存
    """

    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    stage_one_llm = create_structured_llm(
        env_path=args.env_path,
        model_name=args.stage_one_model,
        reasoning_effort=args.stage_one_reasoning_effort,
        schema=StageOneOutput,
    )
    stage_three_llm = create_structured_llm(
        env_path=args.env_path,
        model_name=args.stage_three_model,
        reasoning_effort=args.stage_three_reasoning_effort,
        schema=StageThreeDecision,
    )
    query_runner = CoarseQueryRunner(ROOT_DIR, QUERY_SCRIPT_PATH)

    input_payload = load_json(args.input_json)
    input_sentences = validate_input_payload(input_payload)

    temp_dir, log_dir = ensure_output_dirs(args.output_json)
    intermediate_output_path = get_intermediate_output_path(args.output_json)

    # 续跑逻辑：
    # - 优先使用正式输出
    # - 若正式输出不存在，则使用 temp/ 下的中间文件
    accumulated_sentences, last_completed_index, existing_output_source = load_existing_output(args.output_json)
    remaining_sentences = [sentence for sentence in input_sentences if sentence["index"] > last_completed_index]

    if not remaining_sentences:
        log_header("执行完成")
        log_step("没有剩余句子需要处理。")
        return

    # 审计文件名由文档固定，并且流程结束后不清理。
    audit_path = log_dir / f"{args.input_json.name}.search_audit.jsonl"
    audit_logger = AuditLogger(audit_path)

    batches = chunk_sentences(remaining_sentences, args.batch_size)
    total_batches = len(batches)

    log_header("启动信息")
    log_step(f"输入文件：{args.input_json}")
    log_step(f"输出文件：{args.output_json}")
    log_step(f"中间文件：{intermediate_output_path}")
    log_step(f"审计文件：{audit_path}")
    log_step(f"续跑来源：{existing_output_source}")
    log_step(f"阶段 1 模型：{args.stage_one_model}")
    log_step(f"阶段 1 reasoning_effort：{args.stage_one_reasoning_effort}")
    log_step(f"阶段 3 模型：{args.stage_three_model}")
    log_step(f"阶段 3 reasoning_effort：{args.stage_three_reasoning_effort}")
    log_step(f"总句子数：{len(input_sentences)}")
    log_step(f"已完成到 sentence.index：{last_completed_index}")
    log_step(f"剩余句子数：{len(remaining_sentences)}")
    log_step(f"批次数：{total_batches}")
    log_step(f"批次大小：{args.batch_size}")

    for batch_idx, batch in enumerate(batches, start=1):
        sentence_range = f"{batch[0]['index']}..{batch[-1]['index']}"
        log_header(f"批次 {batch_idx}/{total_batches}")
        log_step(f"句子范围：{sentence_range}")
        log_step(f"句子数量：{len(batch)}")
        log_step(f"[阶段 0] 清洗输入并准备第一阶段请求", indent=1)

        # 先跑阶段 0 + 1 + 2。如果两次校验都失败，异常会直接终止程序，
        # 当前批次不会被写入输出文件。
        log_step("[阶段 1] 调用 LLM 生成结构化分片", indent=1)
        log_step("[阶段 2] 代码侧校验并补 token.index", indent=1)
        batch_stage_two_output = run_stage_one_with_retry(stage_one_llm, batch)
        log_step("[阶段 1/2] 完成", indent=1)

        # 再跑阶段 3。当前批次里的每个 token 都有自己独立的 1->2->3 搜索流程，
        # 同时各自写一条审计记录。
        batch_final_output = process_stage_three_batch(
            stage_three_llm=stage_three_llm,
            query_runner=query_runner,
            audit_logger=audit_logger,
            batch_output=batch_stage_two_output,
        )

        # 只有整个批次都成功后，才会把结果合并进累计输出，
        # 并把最新快照写到 temp/ 下，作为可续跑的中间文件。
        accumulated_sentences.extend(batch_final_output["sentences"])
        atomic_write_json(
            target_path=intermediate_output_path,
            temp_dir=temp_dir,
            input_path=args.input_json,
            payload=create_final_payload(accumulated_sentences),
        )

        log_step(
            f"[保存] 已写入批次 {batch_idx}/{total_batches}，累计句子数：{len(accumulated_sentences)}",
            indent=1,
        )
        log_step(f"[保存] 中间文件：{intermediate_output_path}", indent=1)

    if len(accumulated_sentences) != len(input_sentences):
        raise ValueError(
            "All batches completed but final sentence count does not match input; "
            "treating run as failed and not saving final output"
        )

    timed_sentences = attach_timing_info(input_sentences, accumulated_sentences)
    final_payload = create_final_payload(timed_sentences)
    atomic_write_json(
        target_path=args.output_json,
        temp_dir=args.output_json.parent,
        input_path=args.input_json,
        payload=final_payload,
    )

    if intermediate_output_path.exists():
        intermediate_output_path.unlink()

    log_header("执行完成")
    log_step(f"累计输出句子数：{len(accumulated_sentences)}")
    log_step(f"最终输出：{args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print("\n=== 执行失败 ===", file=sys.stderr, flush=True)
        print(f"错误信息：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
