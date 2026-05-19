#!/usr/bin/env python3
"""
Generate video-context quiz questions from mapped transcript JSON.

This script reads mapped transcript JSON, filters mapped semantic tokens
globally, asks Gemini for strict structured outputs, then fills stable
pre-ingest question metadata in code.

Usage:
  python 6question-generation-gemini.py <mapped_json> <output_questions_json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    CreateCachedContentConfig,
    GenerateContentConfig,
    HttpOptions,
    Part,
    ThinkingConfig,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_SELECTION_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_SELECTION_THINKING_LEVEL = "high"
DEFAULT_QUESTION_MODEL = "gemini-3.1-pro-preview"
DEFAULT_QUESTION_THINKING_LEVEL = "high"
ALLOWED_GEMINI_THINKING_LEVELS = {"low", "medium", "high"}
DEFAULT_SELECTION_TOP_K = 6
DEFAULT_SELECTION_BATCH_SIZE = 12
DEFAULT_SELECTION_MAX_WORKERS = 4
DEFAULT_CANDIDATE_SCORE_THRESHOLD = 6
DEFAULT_CACHE_TTL_SECONDS = 30 * 60
DEFAULT_VIDEO_MIME_TYPE = "video/mp4"
CHECKPOINT_VERSION = 3
DEFAULT_BATCH_SIZE = 12
SUPPORTED_QUESTION_TYPES = {"context_meaning_choice", "context_cloze_choice"}
EXPECTED_OPTION_IDS = ["correct", "wrong_1", "wrong_2", "wrong_3"]
CANDIDATE_SCORE_WEIGHTS = {
    "visual_context": 0.10,
    "context_clarity": 0.35,
    "learning_value": 0.25,
    "question_fit": 0.30,
}


SELECTION_SYSTEM_PROMPT = """
你是英语学习题目的上下文选择器。你的任务不是出题，也不是筛掉单词，而是为每个 coarse unit 在它出现过的句子中选择一个最适合后续学习和出题的 ref，并给出可用于脚本筛选的评分。

输入说明：
- FULL_VIDEO_TRANSCRIPT 是完整字幕；如果调用方提供了视频缓存，你还可以同时参考完整视频画面、声音、说话人语气和场景变化。
- CURRENT_CONTEXT_SELECTION_GROUPS 中的每个 group 代表一个 coarse unit。
- 每个 group 内有多个 sentence_candidates，它们都是该 coarse unit 在视频中真实出现过的位置。

硬性规则：
- 每个 group 必须选择且只能选择一个 sentence_candidate_id。
- sentence_candidate_id 必须来自当前 group 的 sentence_candidates，group_id 必须原样返回。
- 不要拒绝、不要跳过、不要合并 group，也不要为同一个 group 返回多个选择。
- 即使某个词不适合出题、画面帮助很弱、句子很普通，也必须选择一个 ref；这些问题只通过 scores 体现，后续脚本会决定是否成为 Candidate。
- 不要改写目标词、不要自行创造句子、不要输出 schema 之外的字段。

选择标准：
- 优先选择目标词或短语含义最清楚、上下文最完整、句子本身可独立理解的一次出现。
- 如果多个句子都可用，优先选择说话目的明确、上下文不依赖大量前情、画面或语气能辅助理解的句子。
- 如果目标词在某句里只是人名、专名、寒暄、语气词、歌词或难以形成学习点，也仍然选最清楚的一句，但给较低分。
- 选择时以 transcript 里的候选句和时间为准；视频只用于判断画面、声音和上下文是否帮助理解，不要根据视频改动候选边界。

评分规则：
- scores 中四个字段都必须是 0 到 10 的整数，0 表示完全不适合，10 表示非常适合。
- visual_context：视频画面、动作、表情、物件、声音或语气是否直接帮助理解目标词。纯对白且画面无帮助通常较低；画面能直接展示含义或情绪时较高。
- context_clarity：当前句子及邻近上下文是否足以让学习者理解目标词在此处的意思。上下文越独立、指代越少、语义越明确，分数越高。
- learning_value：该词或短语是否值得中文学习者学习。常见但有用的表达、地道搭配、语境化含义、可迁移用法分数更高；专名、一次性梗、无稳定学习价值的内容分数更低。
- question_fit：该 ref 是否适合后续生成唯一正确答案的选择题。正确答案清楚、错误项容易设计且不会出现多个可接受答案时分数更高。
- reason 必须使用中文，简短说明为什么选择该句，以及必要时说明主要扣分点。
""".strip()

QUESTION_SYSTEM_PROMPT = """
你是英语学习题目生成器。你的任务是根据输入的 candidates 生成适合中文学习者的视频上下文选择题；如果某个 candidate 不适合生成高质量题目，可以主动拒绝。

输出边界：
- 当前 batch 中的每个 candidate 最多生成一道题；不要为同一个 candidate 同时生成 result 和 rejection。
- 如果能生成高质量题目，把它放入 results；如果不能，放入 rejections，并用中文给出简短原因。
- candidate_id 必须原样返回，不要改写。不要凭空添加 candidates 之外的目标词，也不要使用 batch 外的 candidate_id。

语言和字段规则：
- question 必须使用中文，直接问学习者要判断的内容。
- explanation 必须使用中文，只解释为什么 correct 是正确答案；不要逐项解释所有错误项。
- context_text 必须保留英文上下文。context_meaning_choice 使用英文原句或足够完整的英文片段；context_cloze_choice 使用英文挖空句。
- context_meaning_choice 的 options.text 必须是中文释义或中文解释。
- context_cloze_choice 的 options.text 必须是英文单词或短语。
- options 必须恰好四个，id 顺序必须是 correct、wrong_1、wrong_2、wrong_3。

题型选择：
- context_meaning_choice 适合考察目标词或短语在当前语境里的含义，尤其是多义词、短语动词、地道表达、比喻用法。
- context_cloze_choice 适合考察固定搭配、短语动词、介词搭配、常见表达，以及能自然挖空且不会泄露答案的句子。
- 如果两种题型都可行，选择更能体现当前视频语境的一种。
- 如果 allowed_question_types 只允许一种题型，只能使用该题型；如果该题型不适合，就拒绝该 candidate。

出题质量要求：
- 题目必须有唯一正确答案。错误项不能和正确答案在当前语境中同样成立，也不能只是正确答案的近义改写。
- 错误项应有迷惑性，但必须可以被当前上下文排除；不要使用明显荒谬、语法完全不匹配、长度风格极不协调的选项。
- 不要用 candidate 的中文释义直接泄露答案；question 不要写成“请选择 xxx 的正确释义”这种只考字典记忆的形式，应该结合 context_text。
- 对 context_cloze_choice，context_text 必须包含 ____，并且不能保留目标词或目标短语本身。
- 对 context_cloze_choice，错误项放回原句后应语义或搭配不自然；不要选择放回去也基本正确的近义表达。
- 对 context_meaning_choice，正确选项应贴合目标词在当前句子里的具体意思，不要只给过宽泛的字典释义。

主动拒绝标准：
- 当前上下文不足以判断目标词含义。
- 目标词在该句里主要是人名、地名、品牌名、影视角色名或其他专有名词。
- 目标词只是语气词、填充词、无稳定学习价值的口头碎片。
- 无法设计三个在当前语境下明确错误且不误导的选项。
- 挖空后会泄露答案，或不挖空就无法形成自然英文句子。

内容示例：
- 正面例子 context_meaning_choice：如果目标词是 sacred，当前句子是 "The most sacred thing I do is care and provide for my workers."，题目可以问“这里的 sacred 最接近什么意思？”。正确选项可表达“神圣、非常重要”。错误项可以是“普通、随便”“昂贵、奢侈”“快速、临时”，因为它们和当前语境下 sacred 的“重要、不可轻视”不一致。解释只说明 sacred 在这里是比喻用法，表示说话人认为这件事非常重要、不可轻视。
- 反面例子 context_meaning_choice：不要把“重要、有意义”作为 sacred 的错误项，因为它和正确含义过近，放在当前语境里也能成立；这类错误项会造成多个可接受答案。
- 正面例子 context_cloze_choice：如果目标短语是 provide for，当前句子是 "The most sacred thing I do is care and provide for my workers."，上下文应改写为 "The most sacred thing I do is care and ____ my workers."。正确选项应是 "provide for"。错误项可以是 "take off"、"work out"、"look up"，因为它们放回原句后语义或语法不自然。
- 反面例子 context_cloze_choice：不要把 "look after" 作为 provide for 的错误项；在 "care and ____ my workers" 里，它在语境和语法上也可能成立，因此不是合格错误项。
""".strip()


class AIOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str


class AIContentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    context_text: str | None = None
    options: list[AIOption]
    explanation: str | None = None


class AIQuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    question_type: Literal["context_meaning_choice", "context_cloze_choice"]
    content_payload: AIContentPayload


class AIRejection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    reason: str


class AIQuestionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[AIQuestionResult]
    rejections: list[AIRejection]


class AIContextSelectionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_context: int = Field(ge=0, le=10)
    context_clarity: int = Field(ge=0, le=10)
    learning_value: int = Field(ge=0, le=10)
    question_fit: int = Field(ge=0, le=10)


class AIContextSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    sentence_candidate_id: str
    scores: AIContextSelectionScores
    reason: str


class AIContextSelectionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: list[AIContextSelection]


class FinalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_type: Literal["video_unit"]
    question_type: Literal["context_meaning_choice", "context_cloze_choice"]
    coarse_unit_id: int
    target_text: str
    context_sentence_index: int
    context_span_index: int
    context_start_ms: int
    context_end_ms: int
    content_payload: dict[str, Any]
    status: Literal["draft"]


class FinalSource(BaseModel):
    mapped_json: str
    model: str


class FinalAudit(BaseModel):
    ref_count: int
    candidate_count: int
    candidate_filtered_count: int
    generated_count: int
    rejected_count: int


class RefScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_context: int
    context_clarity: int
    learning_value: int
    question_fit: int


class QuestionCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    target_text: str
    raw_target_text: str
    base_form: str
    coarse_unit_id: int
    coarse_label: str
    kind: str
    pos: str
    sentence_index: int
    sentence_text: str
    sentence_start_ms: int
    sentence_end_ms: int
    token_index: int
    token_explanation: str
    score: int
    scores: RefScores
    candidate_score: float
    selection_reason: str


class CandidateCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    selection_model: str
    selection_top_k: int
    candidate_score_threshold: float
    allowed_question_types: list[str]
    candidates: list[QuestionCandidatePayload]


class SelectedCoarseUnitRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coarse_unit_id: int
    target_text: str
    sentence_index: int
    token_index: int
    scores: RefScores
    candidate_score: float
    question_reject_reason: str | None
    selection_reason: str


class SelectedCoarseUnitRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    selection_model: str
    selection_top_k: int
    candidate_score_threshold: float
    score_weights: dict[str, float]
    allowed_question_types: list[str]
    refs: list[SelectedCoarseUnitRef]


class FinalOutput(BaseModel):
    source: FinalSource
    questions: list[FinalQuestion]
    audit: FinalAudit
    selected_coarse_unit_refs: SelectedCoarseUnitRefs


@dataclass(frozen=True)
class QuestionOccurrence:
    target_text: str
    raw_target_text: str
    base_form: str
    coarse_unit_id: int
    coarse_label: str
    kind: str
    pos: str
    sentence_index: int
    sentence_text: str
    sentence_start_ms: int
    sentence_end_ms: int
    token_index: int
    token_explanation: str
    score: int


@dataclass(frozen=True)
class SentenceCandidate:
    sentence_candidate_id: str
    target_text: str
    raw_target_text: str
    base_form: str
    coarse_unit_id: int
    coarse_label: str
    kind: str
    pos: str
    sentence_index: int
    sentence_text: str
    sentence_start_ms: int
    sentence_end_ms: int
    token_index: int
    token_explanation: str
    score: int

    def to_selection_payload(self) -> dict[str, Any]:
        return {
            "sentence_candidate_id": self.sentence_candidate_id,
            "target_text": self.target_text,
            "base_form": self.base_form,
            "sentence_text": self.sentence_text,
            "token_explanation": self.token_explanation,
            "score": self.score,
        }


@dataclass(frozen=True)
class ContextSelectionGroup:
    group_id: str
    coarse_unit_id: int
    coarse_label: str
    kind: str
    pos: str
    sentence_candidates: list[SentenceCandidate]

    def to_selection_payload(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "coarse_unit_id": self.coarse_unit_id,
            "coarse_label": self.coarse_label,
            "kind": self.kind,
            "pos": self.pos,
            "sentence_candidates": [
                candidate.to_selection_payload() for candidate in self.sentence_candidates
            ],
        }


@dataclass(frozen=True)
class QuestionCandidate(QuestionOccurrence):
    candidate_id: str
    scores: RefScores
    candidate_score: float
    selection_reason: str

    def to_ai_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_text": self.target_text,
            "base_form": self.base_form,
            "coarse_label": self.coarse_label,
            "kind": self.kind,
            "pos": self.pos,
            "sentence_text": self.sentence_text,
            "token_explanation": self.token_explanation,
        }


@dataclass(frozen=True)
class CandidateReject:
    candidate_id: str | None
    sentence_index: int | None
    token_index: int | None
    target_text: str | None
    reason: str


class GeminiStructuredLLM:
    """通过 Vertex AI Gemini 请求 Pydantic schema 约束的结构化输出。"""

    def __init__(
        self,
        client: Any,
        model_name: str,
        thinking_level: str,
        schema: type[BaseModel],
    ) -> None:
        if thinking_level not in ALLOWED_GEMINI_THINKING_LEVELS:
            raise ValueError("thinking_level must be one of: low, medium, high")
        self.client = client
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.schema = schema

    def invoke(
        self,
        messages: list[dict[str, str]],
        cached_content_name: str | None = None,
    ) -> BaseModel:
        system_instruction, contents = split_gemini_messages(messages)
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": self.schema,
            "thinking_config": ThinkingConfig(
                thinking_level=self.thinking_level,
            ),
        }
        if cached_content_name:
            config_kwargs["cached_content"] = cached_content_name
        else:
            config_kwargs["system_instruction"] = system_instruction
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(**config_kwargs),
        )
        print_usage_metadata(f"🤖 Gemini {self.model_name}", response.usage_metadata)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, self.schema):
            return parsed
        if isinstance(parsed, BaseModel):
            return self.schema.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return self.schema.model_validate(parsed)
        content = response.text
        if not content:
            raise ValueError("Gemini returned an empty structured response")
        return self.schema.model_validate_json(content)


class GeminiQuestionLLM:
    def __init__(self, client: Any, model_name: str, thinking_level: str) -> None:
        self.structured_llm = GeminiStructuredLLM(
            client=client,
            model_name=model_name,
            thinking_level=thinking_level,
            schema=AIQuestionBatchOutput,
        )

    def invoke_question_batch(self, messages: list[dict[str, str]]) -> AIQuestionBatchOutput:
        result = self.structured_llm.invoke(messages)
        if isinstance(result, AIQuestionBatchOutput):
            return result
        return AIQuestionBatchOutput.model_validate(result.model_dump())


class GeminiContextSelectionLLM:
    def __init__(self, client: Any, model_name: str, thinking_level: str) -> None:
        self.structured_llm = GeminiStructuredLLM(
            client=client,
            model_name=model_name,
            thinking_level=thinking_level,
            schema=AIContextSelectionBatchOutput,
        )

    def invoke_context_selection_batch(
        self,
        messages: list[dict[str, str]],
        cached_content_name: str | None = None,
    ) -> AIContextSelectionBatchOutput:
        result = self.structured_llm.invoke(
            messages,
            cached_content_name=cached_content_name,
        )
        if isinstance(result, AIContextSelectionBatchOutput):
            return result
        return AIContextSelectionBatchOutput.model_validate(result.model_dump())


def split_gemini_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    system_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "system"
    ]
    non_system_messages = [
        message
        for message in messages
        if message.get("role") != "system"
    ]
    contents = "\n\n".join(
        message.get("content", "")
        for message in non_system_messages
    )
    return "\n\n".join(system_messages) or None, contents


def print_usage_metadata(prefix: str, usage_metadata: object | None) -> None:
    if usage_metadata is None:
        return
    print(f"{prefix} usage_metadata: {usage_metadata}", flush=True)


def validate_video_gcs_uri(video_gcs_uri: str) -> None:
    if not video_gcs_uri:
        raise ValueError("--video-gcs-uri 是必填参数")
    if not video_gcs_uri.startswith("gs://"):
        raise ValueError("--video-gcs-uri 必须是 gs:// 开头的 Cloud Storage 对象地址")

    try:
        completed = subprocess.run(
            ["gcloud", "storage", "objects", "describe", video_gcs_uri],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("需要安装 gcloud CLI 才能校验 --video-gcs-uri") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FileNotFoundError(
            f"--video-gcs-uri 不存在、无权限访问或无法读取: {video_gcs_uri}"
            + (f"\n{detail}" if detail else "")
        )


def build_cache_display_name(video_gcs_uri: str) -> str:
    digest = hashlib.sha1(video_gcs_uri.encode("utf-8")).hexdigest()[:12]
    return f"question-selection-{digest}"


def create_video_context_cache(
    client: Any,
    model_name: str,
    video_gcs_uri: str,
    full_transcript_text: str,
    video_mime_type: str = DEFAULT_VIDEO_MIME_TYPE,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> str:
    if not video_gcs_uri.startswith("gs://"):
        raise ValueError("--video-gcs-uri must be a gs:// Cloud Storage URI")

    print(f"🎞️  创建 question selection 视频 explicit context cache: {video_gcs_uri}", flush=True)
    cache_display_name = build_cache_display_name(video_gcs_uri)
    cache = client.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            display_name=cache_display_name,
            ttl=f"{cache_ttl_seconds}s",
            system_instruction=SELECTION_SYSTEM_PROMPT,
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_uri(
                            file_uri=video_gcs_uri,
                            mime_type=video_mime_type,
                        ),
                        Part.from_text(
                            text=(
                                "FULL_VIDEO_TRANSCRIPT:\n"
                                f"{full_transcript_text}"
                            ),
                        ),
                    ],
                )
            ],
        ),
    )
    print(f"🎞️  cache display_name: {cache_display_name}", flush=True)
    print(f"🎞️  cache name: {cache.name}", flush=True)
    print_usage_metadata("🎞️  cache", cache.usage_metadata)
    return cache.name


def delete_context_cache(client: Any, cache_name: str) -> None:
    print(f"🧹 删除 explicit context cache: {cache_name}", flush=True)
    client.caches.delete(name=cache_name)


def log_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def log_step(message: str, indent: int = 0) -> None:
    print(f"{'  ' * indent}{message}", flush=True)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalize_lookup_text(value: str) -> str:
    normalized = normalize_whitespace(value).lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", normalized)


def normalize_candidate_key(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_lookup_text(value))


def strip_target_edges(value: str) -> str:
    normalized = normalize_whitespace(value)
    return re.sub(r"^[^\w]+|[^\w]+$", "", normalized)


def clean_target_text(value: str) -> str:
    return strip_target_edges(value).lower()


def apply_base_form_casing(target_text: str, base_form: str) -> str:
    if target_text and base_form[:1].isupper():
        return target_text[:1].upper() + target_text[1:]
    return target_text


def normalize_target_text_for_output(raw_target_text: str, base_form: str) -> str:
    return apply_base_form_casing(clean_target_text(raw_target_text), base_form)


def looks_like_proper_name(raw_target_text: str, base_form: str) -> bool:
    raw_cleaned = strip_target_edges(raw_target_text)
    if not raw_cleaned or not base_form:
        return False
    return raw_cleaned[:1].isupper() and base_form[:1].isupper()


def calculate_candidate_score(scores: RefScores | AIContextSelectionScores) -> float:
    return round(
        scores.visual_context * CANDIDATE_SCORE_WEIGHTS["visual_context"]
        + scores.context_clarity * CANDIDATE_SCORE_WEIGHTS["context_clarity"]
        + scores.learning_value * CANDIDATE_SCORE_WEIGHTS["learning_value"]
        + scores.question_fit * CANDIDATE_SCORE_WEIGHTS["question_fit"],
        2,
    )


def default_selection_scores() -> RefScores:
    return RefScores(
        visual_context=5,
        context_clarity=5,
        learning_value=5,
        question_fit=5,
    )


def score_candidate(kind: str, pos: str, target_text: str, sentence_text: str) -> int:
    score = 0
    if kind == "phrase":
        score += 80
    score += {
        "adjective": 60,
        "adverb": 55,
        "verb": 50,
        "noun": 40,
    }.get(pos, 10)
    if len(target_text.split()) > 1:
        score += 30
    word_count = len(sentence_text.split())
    if 4 <= word_count <= 22:
        score += 20
    elif word_count < 4:
        score -= 30
    elif word_count > 28:
        score -= 15
    return score


def reject_record(
    reason: str,
    sentence: dict[str, Any] | None = None,
    token: dict[str, Any] | None = None,
    target_text: str | None = None,
) -> CandidateReject:
    return CandidateReject(
        candidate_id=None,
        sentence_index=sentence.get("index") if isinstance(sentence, dict) else None,
        token_index=token.get("index") if isinstance(token, dict) else None,
        target_text=target_text,
        reason=reason,
    )


def extract_question_occurrences(
    mapped_payload: dict[str, Any],
    allowed_question_types: list[str],
) -> tuple[list[QuestionOccurrence], list[CandidateReject]]:
    if not set(allowed_question_types).issubset(SUPPORTED_QUESTION_TYPES):
        unsupported = sorted(set(allowed_question_types) - SUPPORTED_QUESTION_TYPES)
        raise ValueError(f"Unsupported question types: {unsupported}")

    sentences = mapped_payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError("Mapped JSON must contain a sentences array")

    rejects: list[CandidateReject] = []
    occurrences: list[QuestionOccurrence] = []

    for sentence in sentences:
        if not isinstance(sentence, dict):
            rejects.append(reject_record("sentence is not an object"))
            continue

        sentence_required = {"index", "text", "translation", "start", "end"}
        if not sentence_required.issubset(sentence):
            for token in sentence.get("tokens", []) if isinstance(sentence.get("tokens"), list) else []:
                rejects.append(reject_record("sentence timing or text is missing", sentence, token))
            continue

        tokens = sentence.get("tokens")
        if not isinstance(tokens, list):
            rejects.append(reject_record("sentence tokens is not a list", sentence))
            continue

        for token in tokens:
            if not isinstance(token, dict):
                rejects.append(reject_record("token is not an object", sentence))
                continue

            raw_target_text = strip_target_edges(str(token.get("text", "")))
            semantic = token.get("semantic_element")
            if not isinstance(semantic, dict):
                rejects.append(reject_record("semantic_element is missing", sentence, token, raw_target_text))
                continue

            coarse_id = semantic.get("coarse_id")
            if coarse_id is None:
                rejects.append(reject_record("coarse_id is null", sentence, token, raw_target_text))
                continue
            if not isinstance(coarse_id, int):
                rejects.append(reject_record("coarse_id is not an integer", sentence, token, raw_target_text))
                continue

            if not {"index", "text", "start", "end"}.issubset(token):
                rejects.append(reject_record("token timing is missing", sentence, token, raw_target_text))
                continue

            if not raw_target_text:
                rejects.append(reject_record("target text is empty", sentence, token, raw_target_text))
                continue

            base_form = normalize_whitespace(str(semantic.get("base_form", "")).strip())
            target_text = normalize_target_text_for_output(raw_target_text, base_form)
            translation = normalize_whitespace(str(semantic.get("translation", "")).strip())
            kind = normalize_lookup_text(str(semantic.get("kind", "")))
            pos = normalize_lookup_text(str(semantic.get("pos", "")))

            occurrences.append(
                QuestionOccurrence(
                    target_text=target_text,
                    raw_target_text=raw_target_text,
                    base_form=base_form,
                    coarse_unit_id=coarse_id,
                    coarse_label=translation,
                    kind=kind,
                    pos=pos,
                    sentence_index=int(sentence["index"]),
                    sentence_text=normalize_whitespace(str(sentence["text"])),
                    sentence_start_ms=int(sentence["start"]),
                    sentence_end_ms=int(sentence["end"]),
                    token_index=int(token["index"]),
                    token_explanation=normalize_whitespace(str(token.get("explanation", ""))),
                    score=score_candidate(kind, pos, target_text, str(sentence["text"])),
                )
            )

    occurrences.sort(
        key=lambda item: (
            -item.score,
            item.sentence_index,
            item.token_index,
            item.target_text.lower(),
        )
    )
    return occurrences, rejects


def build_context_selection_groups(
    occurrences: list[QuestionOccurrence],
    top_k: int,
) -> list[ContextSelectionGroup]:
    sentence_candidates_by_key: dict[tuple[int, int], QuestionOccurrence] = {}
    for occurrence in occurrences:
        key = (occurrence.coarse_unit_id, occurrence.sentence_index)
        existing = sentence_candidates_by_key.get(key)
        if existing is None or occurrence.token_index < existing.token_index:
            sentence_candidates_by_key[key] = occurrence

    grouped: dict[int, list[QuestionOccurrence]] = {}
    for occurrence in sentence_candidates_by_key.values():
        grouped.setdefault(occurrence.coarse_unit_id, []).append(occurrence)

    group_items = []
    for coarse_id, coarse_occurrences in grouped.items():
        coarse_occurrences.sort(
            key=lambda item: (
                -item.score,
                item.sentence_index,
                item.token_index,
                item.target_text.lower(),
            )
        )
        group_items.append((coarse_occurrences[0], coarse_occurrences[:top_k]))

    group_items.sort(
        key=lambda item: (
            -item[0].score,
            item[0].sentence_index,
            item[0].token_index,
            item[0].target_text.lower(),
        )
    )

    groups: list[ContextSelectionGroup] = []
    sentence_candidate_counter = 0
    for group_index, (best_occurrence, group_occurrences) in enumerate(group_items, start=1):
        sentence_candidates: list[SentenceCandidate] = []
        for occurrence in group_occurrences:
            sentence_candidate_counter += 1
            sentence_candidates.append(
                SentenceCandidate(
                    sentence_candidate_id=f"s_{sentence_candidate_counter:06d}",
                    **asdict(occurrence),
                )
            )
        groups.append(
            ContextSelectionGroup(
                group_id=f"g_{group_index:06d}",
                coarse_unit_id=best_occurrence.coarse_unit_id,
                coarse_label=best_occurrence.coarse_label,
                kind=best_occurrence.kind,
                pos=best_occurrence.pos,
                sentence_candidates=sentence_candidates,
            )
        )
    return groups


def apply_context_selections(
    groups: list[ContextSelectionGroup],
    selection_output: AIContextSelectionBatchOutput,
) -> list[QuestionCandidate]:
    groups_by_id = {group.group_id: group for group in groups}
    seen_group_ids: set[str] = set()
    selected: list[QuestionCandidate] = []

    for selection in selection_output.selections:
        if selection.group_id not in groups_by_id:
            raise ValueError(f"unknown group_id in context selection: {selection.group_id}")
        if selection.group_id in seen_group_ids:
            raise ValueError(f"duplicate selection group_id: {selection.group_id}")
        seen_group_ids.add(selection.group_id)

        group = groups_by_id[selection.group_id]
        candidates_by_id = {
            candidate.sentence_candidate_id: candidate for candidate in group.sentence_candidates
        }
        sentence_candidate = candidates_by_id.get(selection.sentence_candidate_id)
        if sentence_candidate is None:
            raise ValueError(
                f"unknown sentence_candidate_id for {selection.group_id}: {selection.sentence_candidate_id}"
            )
        scores = RefScores.model_validate(selection.scores.model_dump())
        selected.append(
            QuestionCandidate(
                candidate_id=f"c_{len(selected) + 1:06d}",
                target_text=sentence_candidate.target_text,
                raw_target_text=sentence_candidate.raw_target_text,
                base_form=sentence_candidate.base_form,
                coarse_unit_id=sentence_candidate.coarse_unit_id,
                coarse_label=sentence_candidate.coarse_label,
                kind=sentence_candidate.kind,
                pos=sentence_candidate.pos,
                sentence_index=sentence_candidate.sentence_index,
                sentence_text=sentence_candidate.sentence_text,
                sentence_start_ms=sentence_candidate.sentence_start_ms,
                sentence_end_ms=sentence_candidate.sentence_end_ms,
                token_index=sentence_candidate.token_index,
                token_explanation=sentence_candidate.token_explanation,
                score=sentence_candidate.score,
                scores=scores,
                candidate_score=calculate_candidate_score(scores),
                selection_reason=selection.reason,
            )
        )

    missing_group_ids = sorted(set(groups_by_id) - seen_group_ids)
    if missing_group_ids:
        raise ValueError(f"missing selections for groups: {missing_group_ids}")
    return selected


def extract_question_candidates(
    mapped_payload: dict[str, Any],
    allowed_question_types: list[str],
) -> tuple[list[QuestionCandidate], list[CandidateReject]]:
    occurrences, rejects = extract_question_occurrences(
        mapped_payload,
        allowed_question_types=allowed_question_types,
    )
    groups = build_context_selection_groups(
        occurrences,
        top_k=1,
    )
    selection_output = AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": group.group_id,
                    "sentence_candidate_id": group.sentence_candidates[0].sentence_candidate_id,
                    "scores": default_selection_scores().model_dump(),
                    "reason": "legacy auto selection",
                }
                for group in groups
            ]
        }
    )
    return apply_context_selections(groups, selection_output), rejects


def select_context_groups_with_llm(
    groups: list[ContextSelectionGroup],
    selection_llm: Any,
    full_transcript_text: str,
    selection_batch_size: int,
    selection_max_workers: int,
    cached_content_name: str | None = None,
) -> AIContextSelectionBatchOutput:
    selections_by_group_id: dict[str, AIContextSelection] = {}
    if selection_llm is None:
        for group in groups:
            selections_by_group_id[group.group_id] = AIContextSelection(
                group_id=group.group_id,
                sentence_candidate_id=group.sentence_candidates[0].sentence_candidate_id,
                scores=AIContextSelectionScores.model_validate(default_selection_scores().model_dump()),
                reason="only one sentence candidate",
            )
        return AIContextSelectionBatchOutput(
            selections=[selections_by_group_id[group.group_id] for group in groups]
        )

    llm_groups = list(groups)

    def invoke_batch(batch: list[ContextSelectionGroup]) -> AIContextSelectionBatchOutput:
        messages = build_context_selection_messages(
            batch,
            full_transcript_text,
            include_full_transcript=cached_content_name is None,
        )
        try:
            batch_output = selection_llm.invoke_context_selection_batch(
                messages,
                cached_content_name=cached_content_name,
            )
        except TypeError:
            batch_output = selection_llm.invoke_context_selection_batch(messages)
        apply_context_selections(batch, batch_output)
        return batch_output

    llm_batches = chunk_items(llm_groups, selection_batch_size)
    if llm_batches:
        worker_count = min(selection_max_workers, len(llm_batches))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_batch = {
                executor.submit(invoke_batch, batch): batch for batch in llm_batches
            }
            for future in as_completed(future_to_batch):
                batch_output = future.result()
                for selection in batch_output.selections:
                    selections_by_group_id[selection.group_id] = selection

    return AIContextSelectionBatchOutput(
        selections=[selections_by_group_id[group.group_id] for group in groups]
    )


def validate_content_payload(
    payload: AIContentPayload,
    question_type: str,
    target_text: str,
) -> None:
    question = normalize_whitespace(payload.question)
    if not question:
        raise ValueError("question must be non-empty")
    if len(payload.options) != 4:
        raise ValueError("content_payload.options must contain exactly 4 options")

    option_ids = [option.id for option in payload.options]
    if option_ids != EXPECTED_OPTION_IDS:
        raise ValueError(f"options must use ids {EXPECTED_OPTION_IDS}")

    option_texts = [normalize_whitespace(option.text) for option in payload.options]
    if any(not text for text in option_texts):
        raise ValueError("option text must be non-empty")
    if len({text.lower() for text in option_texts}) != len(option_texts):
        raise ValueError("option texts must be unique")

    if payload.context_text is None or not normalize_whitespace(payload.context_text):
        raise ValueError("content_payload.context_text must be non-empty")

    if question_type == "context_cloze_choice":
        target_lookup = normalize_candidate_key(target_text)
        context_lookup = normalize_candidate_key(payload.context_text)
        if target_lookup and re.search(rf"\b{re.escape(target_lookup)}\b", context_lookup):
            raise ValueError("cloze context_text leaks target text")
        if "____" not in payload.context_text:
            raise ValueError("cloze context_text must contain ____")


def merge_ai_result(
    candidate: QuestionCandidate,
    ai_result: AIQuestionResult,
) -> FinalQuestion:
    validate_content_payload(
        ai_result.content_payload,
        question_type=ai_result.question_type,
        target_text=candidate.target_text,
    )
    return FinalQuestion(
        scope_type="video_unit",
        question_type=ai_result.question_type,
        coarse_unit_id=candidate.coarse_unit_id,
        target_text=candidate.target_text,
        context_sentence_index=candidate.sentence_index,
        context_span_index=candidate.token_index,
        context_start_ms=candidate.sentence_start_ms,
        context_end_ms=candidate.sentence_end_ms,
        content_payload=ai_result.content_payload.model_dump(),
        status="draft",
    )


def build_ai_messages(
    candidates: list[QuestionCandidate],
    allowed_question_types: list[str],
    full_transcript_text: str,
) -> list[dict[str, str]]:
    batch_payload = {
        "candidates": [candidate.to_ai_payload() for candidate in candidates],
        "allowed_question_types": allowed_question_types,
    }
    return [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "FULL_VIDEO_TRANSCRIPT:\n"
                f"{full_transcript_text}\n\n"
                "CURRENT_CANDIDATE_BATCH:\n"
                + json.dumps(batch_payload, ensure_ascii=False, indent=2)
            ),
        },
    ]


def build_context_selection_messages(
    groups: list[ContextSelectionGroup],
    full_transcript_text: str,
    include_full_transcript: bool = True,
) -> list[dict[str, str]]:
    batch_payload = {
        "groups": [group.to_selection_payload() for group in groups],
    }
    content_parts = []
    if include_full_transcript:
        content_parts.append(
            "FULL_VIDEO_TRANSCRIPT:\n"
            f"{full_transcript_text}"
        )
    content_parts.append(
        "CURRENT_CONTEXT_SELECTION_GROUPS:\n"
        + json.dumps(batch_payload, ensure_ascii=False, indent=2)
    )
    return [
        {
            "role": "system",
            "content": SELECTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "\n\n".join(content_parts),
        },
    ]


def chunk_items(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def chunk_candidates(candidates: list[QuestionCandidate], batch_size: int) -> list[list[QuestionCandidate]]:
    return [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]


def count_candidate_batches(candidate_count: int, batch_size: int) -> int:
    if candidate_count <= 0:
        return 0
    return (candidate_count + batch_size - 1) // batch_size


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON file must contain an object")
    return payload


def build_full_transcript_text(mapped_payload: dict[str, Any]) -> str:
    sentences = mapped_payload.get("sentences")
    if not isinstance(sentences, list):
        return ""
    lines: list[str] = []
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        sentence_text = normalize_whitespace(str(sentence.get("text", "")).strip())
        if sentence_text:
            lines.append(sentence_text)
    return "\n".join(lines)


def atomic_write_json(target_path: Path, temp_dir: Path, input_path: Path, payload: dict[str, Any]) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{input_path.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_intermediate_output_path(output_path: Path) -> Path:
    return output_path.parent / "temp" / output_path.name


def ensure_output_dirs(output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "temp"
    log_dir = output_path.parent / "log"
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, log_dir


def question_resume_key(question: FinalQuestion) -> tuple[int, str]:
    return question.coarse_unit_id, normalize_candidate_key(question.target_text)


def candidate_resume_key(candidate: QuestionCandidate) -> tuple[int, str]:
    return candidate.coarse_unit_id, normalize_candidate_key(candidate.target_text)


def candidate_to_payload(candidate: QuestionCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["scores"] = candidate.scores.model_dump()
    return payload


def candidate_from_payload(payload: QuestionCandidatePayload) -> QuestionCandidate:
    data = payload.model_dump()
    data["scores"] = RefScores.model_validate(data["scores"])
    return QuestionCandidate(**data)


def build_candidate_checkpoint(
    candidates: list[QuestionCandidate],
    selection_model_name: str,
    selection_top_k: int,
    allowed_question_types: list[str],
    candidate_score_threshold: float,
) -> CandidateCheckpoint:
    return CandidateCheckpoint(
        version=CHECKPOINT_VERSION,
        selection_model=selection_model_name,
        selection_top_k=selection_top_k,
        candidate_score_threshold=candidate_score_threshold,
        allowed_question_types=list(allowed_question_types),
        candidates=[
            QuestionCandidatePayload.model_validate(candidate_to_payload(candidate))
            for candidate in candidates
        ],
    )


def build_selected_ref_checkpoint(
    refs: list[QuestionCandidate],
    selection_model_name: str,
    selection_top_k: int,
    allowed_question_types: list[str],
    candidate_score_threshold: float,
) -> CandidateCheckpoint:
    return build_candidate_checkpoint(
        candidates=refs,
        selection_model_name=selection_model_name,
        selection_top_k=selection_top_k,
        allowed_question_types=allowed_question_types,
        candidate_score_threshold=candidate_score_threshold,
    )


def validate_candidate_checkpoint(
    checkpoint: CandidateCheckpoint,
    selection_model_name: str,
    selection_top_k: int,
    allowed_question_types: list[str],
    candidate_score_threshold: float,
) -> None:
    if checkpoint.version != CHECKPOINT_VERSION:
        raise ValueError(f"candidate checkpoint version mismatch: {checkpoint.version}")
    if checkpoint.selection_model != selection_model_name:
        raise ValueError(f"candidate checkpoint selection model mismatch: {checkpoint.selection_model}")
    if checkpoint.selection_top_k != selection_top_k:
        raise ValueError(f"candidate checkpoint selection_top_k mismatch: {checkpoint.selection_top_k}")
    if checkpoint.candidate_score_threshold != candidate_score_threshold:
        raise ValueError(
            f"candidate checkpoint candidate_score_threshold mismatch: {checkpoint.candidate_score_threshold}"
        )
    if checkpoint.allowed_question_types != list(allowed_question_types):
        raise ValueError("candidate checkpoint allowed_question_types mismatch")

    seen_candidate_ids: set[str] = set()
    for candidate in checkpoint.candidates:
        if candidate.candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate candidate_id in checkpoint: {candidate.candidate_id}")
        seen_candidate_ids.add(candidate.candidate_id)


def build_selected_coarse_unit_refs(
    candidate_checkpoint: CandidateCheckpoint,
    question_reject_reasons: dict[str, str],
) -> SelectedCoarseUnitRefs:
    return SelectedCoarseUnitRefs(
        version=candidate_checkpoint.version,
        selection_model=candidate_checkpoint.selection_model,
        selection_top_k=candidate_checkpoint.selection_top_k,
        candidate_score_threshold=candidate_checkpoint.candidate_score_threshold,
        score_weights=dict(CANDIDATE_SCORE_WEIGHTS),
        allowed_question_types=list(candidate_checkpoint.allowed_question_types),
        refs=[
            SelectedCoarseUnitRef(
                coarse_unit_id=candidate.coarse_unit_id,
                target_text=candidate.target_text,
                sentence_index=candidate.sentence_index,
                token_index=candidate.token_index,
                scores=candidate.scores,
                candidate_score=candidate.candidate_score,
                question_reject_reason=question_reject_reasons.get(candidate.candidate_id),
                selection_reason=candidate.selection_reason,
            )
            for candidate in candidate_checkpoint.candidates
        ],
    )


def load_existing_question_output(
    output_path: Path,
    mapped_json: Path,
    model_name: str,
    allowed_question_types: list[str],
    selection_model_name: str,
    selection_top_k: int,
    candidate_score_threshold: float,
) -> tuple[
    list[FinalQuestion],
    set[str],
    int,
    str,
    CandidateCheckpoint | None,
    dict[tuple[int, str], str],
]:
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
        return [], set(), 0, source_label, None, {}

    payload = load_json(source_path)
    source = payload.get("source", {})
    if isinstance(source, dict):
        existing_mapped_json = source.get("mapped_json")
        if existing_mapped_json and existing_mapped_json != str(mapped_json):
            raise ValueError(f"Existing question output mapped_json mismatch: {existing_mapped_json}")
        existing_model = source.get("model")
        if existing_model and existing_model != model_name:
            raise ValueError(f"Existing question output model mismatch: {existing_model}")

    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ValueError("Existing question output JSON does not contain a valid questions array")

    questions = [FinalQuestion.model_validate(item) for item in raw_questions]
    unsupported_types = sorted({question.question_type for question in questions} - set(allowed_question_types))
    if unsupported_types:
        raise ValueError(f"Existing question output contains unsupported question types: {unsupported_types}")

    audit = payload.get("audit", {})
    rejected_count = 0
    processed_candidate_ids: set[str] = set()
    if isinstance(audit, dict):
        raw_rejected_count = audit.get("rejected_count", 0)
        if isinstance(raw_rejected_count, int):
            rejected_count = raw_rejected_count
        raw_processed_candidate_ids = audit.get("processed_candidate_ids", [])
        if isinstance(raw_processed_candidate_ids, list):
            processed_candidate_ids = {
                item for item in raw_processed_candidate_ids if isinstance(item, str) and item
            }

    checkpoint: CandidateCheckpoint | None = None
    raw_checkpoint = payload.get("candidate_checkpoint")
    if raw_checkpoint is not None:
        checkpoint = CandidateCheckpoint.model_validate(raw_checkpoint)
        validate_candidate_checkpoint(
            checkpoint,
            selection_model_name=selection_model_name,
            selection_top_k=selection_top_k,
            allowed_question_types=allowed_question_types,
            candidate_score_threshold=candidate_score_threshold,
        )

    existing_ref_reject_reasons: dict[tuple[int, str], str] = {}
    raw_selected_refs = payload.get("selected_coarse_unit_refs", {})
    if isinstance(raw_selected_refs, dict):
        raw_refs = raw_selected_refs.get("refs", [])
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                reason = ref.get("question_reject_reason")
                coarse_unit_id = ref.get("coarse_unit_id")
                target_text = ref.get("target_text")
                if isinstance(reason, str) and reason and isinstance(coarse_unit_id, int) and isinstance(target_text, str):
                    existing_ref_reject_reasons[(coarse_unit_id, normalize_candidate_key(target_text))] = reason

    return (
        questions,
        processed_candidate_ids,
        rejected_count,
        source_label,
        checkpoint,
        existing_ref_reject_reasons,
    )


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_hard_filter_rejects(audit_logger: AuditLogger, rejects: list[CandidateReject]) -> None:
    for reject in rejects:
        audit_logger.write(
            {
                "event": "hard_filter_reject",
                "sentence_index": reject.sentence_index,
                "token_index": reject.token_index,
                "target_text": reject.target_text,
                "reason": reject.reason,
            }
        )


def write_question_filter_rejects(audit_logger: AuditLogger, rejects: list[CandidateReject]) -> None:
    for reject in rejects:
        audit_logger.write(
            {
                "event": "question_filter_reject",
                "candidate_id": reject.candidate_id,
                "sentence_index": reject.sentence_index,
                "token_index": reject.token_index,
                "target_text": reject.target_text,
                "reason": reject.reason,
            }
        )


def reject_reason_to_record(candidate: QuestionCandidate, reason: str) -> CandidateReject:
    return CandidateReject(
        candidate_id=candidate.candidate_id,
        sentence_index=candidate.sentence_index,
        token_index=candidate.token_index,
        target_text=candidate.target_text,
        reason=reason,
    )


def write_selection_events(
    audit_logger: AuditLogger,
    selection_output: AIContextSelectionBatchOutput,
    auto_selected_group_ids: set[str],
) -> None:
    for selection in selection_output.selections:
        audit_logger.write(
            {
                "event": "context_selection",
                "group_id": selection.group_id,
                "sentence_candidate_id": selection.sentence_candidate_id,
                "auto_selected": selection.group_id in auto_selected_group_ids,
                "reason": selection.reason,
            }
        )


def select_question_candidates(
    mapped_payload: dict[str, Any],
    allowed_question_types: list[str],
    selection_top_k: int,
    selection_batch_size: int,
    selection_max_workers: int,
    selection_llm: Any,
    full_transcript_text: str,
    audit_logger: AuditLogger,
    cached_content_name: str | None = None,
) -> tuple[list[QuestionCandidate], list[CandidateReject]]:
    occurrences, hard_rejects = extract_question_occurrences(
        mapped_payload,
        allowed_question_types=allowed_question_types,
    )
    write_hard_filter_rejects(audit_logger, hard_rejects)

    groups = build_context_selection_groups(
        occurrences,
        top_k=selection_top_k,
    )
    auto_selected_group_ids = {
        group.group_id for group in groups if len(group.sentence_candidates) == 1
    }
    selection_output = select_context_groups_with_llm(
        groups=groups,
        selection_llm=selection_llm,
        full_transcript_text=full_transcript_text,
        selection_batch_size=selection_batch_size,
        selection_max_workers=selection_max_workers,
        cached_content_name=cached_content_name,
    )
    candidates = apply_context_selections(groups, selection_output)
    write_selection_events(audit_logger, selection_output, auto_selected_group_ids)
    audit_logger.write(
        {
            "event": "candidate_checkpoint_created",
            "occurrence_count": len(occurrences),
            "group_count": len(groups),
            "selected_candidate_count": len(candidates),
        }
    )
    return candidates, hard_rejects


def filter_question_generation_candidates(
    candidates: list[QuestionCandidate],
    candidate_score_threshold: float,
) -> tuple[list[QuestionCandidate], dict[str, str]]:
    eligible: list[QuestionCandidate] = []
    reject_reasons: dict[str, str] = {}

    for candidate in candidates:
        if looks_like_proper_name(candidate.raw_target_text, candidate.base_form):
            reject_reasons[candidate.candidate_id] = "专有名词"
            continue
        if candidate.candidate_score < candidate_score_threshold:
            reject_reasons[candidate.candidate_id] = "candidate_score 低于阈值"
            continue

        eligible.append(candidate)

    return eligible, reject_reasons


def validate_batch_candidate_ids(
    batch_output: AIQuestionBatchOutput,
    batch_candidates_by_id: dict[str, QuestionCandidate],
) -> None:
    seen_result_ids: set[str] = set()
    for result in batch_output.results:
        if result.candidate_id not in batch_candidates_by_id:
            raise ValueError(f"AI returned unknown candidate_id: {result.candidate_id}")
        if result.candidate_id in seen_result_ids:
            raise ValueError(f"AI returned duplicate result candidate_id: {result.candidate_id}")
        seen_result_ids.add(result.candidate_id)
    for rejection in batch_output.rejections:
        if rejection.candidate_id not in batch_candidates_by_id:
            raise ValueError(f"AI returned unknown rejection candidate_id: {rejection.candidate_id}")


def run_generation(
    mapped_json: Path,
    output_json: Path,
    allowed_question_types: list[str],
    batch_size: int,
    llm: Any,
    model_name: str,
    gemini_client: Any | None = None,
    selection_llm: Any | None = None,
    selection_model_name: str = DEFAULT_SELECTION_MODEL,
    selection_top_k: int = DEFAULT_SELECTION_TOP_K,
    selection_batch_size: int = DEFAULT_SELECTION_BATCH_SIZE,
    selection_max_workers: int = DEFAULT_SELECTION_MAX_WORKERS,
    candidate_score_threshold: float = DEFAULT_CANDIDATE_SCORE_THRESHOLD,
    video_gcs_uri: str | None = None,
    video_mime_type: str = DEFAULT_VIDEO_MIME_TYPE,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> FinalOutput:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if selection_top_k < 1:
        raise ValueError("selection_top_k must be at least 1")
    if selection_batch_size < 1:
        raise ValueError("selection_batch_size must be at least 1")
    if selection_max_workers < 1:
        raise ValueError("selection_max_workers must be at least 1")
    if candidate_score_threshold < 0:
        raise ValueError("candidate_score_threshold must be non-negative")
    validate_video_gcs_uri(video_gcs_uri or "")
    if gemini_client is None:
        raise ValueError("gemini_client is required when video_gcs_uri is provided")

    mapped_payload = load_json(mapped_json)
    full_transcript_text = build_full_transcript_text(mapped_payload)
    temp_dir, log_dir = ensure_output_dirs(output_json)
    audit_logger = AuditLogger(log_dir / f"{mapped_json.name}.question_audit.jsonl")
    intermediate_output_path = get_intermediate_output_path(output_json)

    (
        existing_questions,
        processed_candidate_ids,
        existing_rejected_count,
        existing_source,
        candidate_checkpoint,
        existing_ref_reject_reasons,
    ) = (
        load_existing_question_output(
            output_json,
            mapped_json=mapped_json,
            model_name=model_name,
            allowed_question_types=allowed_question_types,
            selection_model_name=selection_model_name,
            selection_top_k=selection_top_k,
            candidate_score_threshold=candidate_score_threshold,
        )
    )
    hard_rejects: list[CandidateReject] = []
    cached_content_name: str | None = None
    try:
        if candidate_checkpoint is None:
            if video_gcs_uri:
                cached_content_name = create_video_context_cache(
                    client=gemini_client,
                    model_name=selection_model_name,
                    video_gcs_uri=video_gcs_uri,
                    full_transcript_text=full_transcript_text,
                    video_mime_type=video_mime_type,
                    cache_ttl_seconds=cache_ttl_seconds,
                )
            candidates, hard_rejects = select_question_candidates(
                mapped_payload=mapped_payload,
                allowed_question_types=allowed_question_types,
                selection_top_k=selection_top_k,
                selection_batch_size=selection_batch_size,
                selection_max_workers=selection_max_workers,
                selection_llm=selection_llm,
                full_transcript_text=full_transcript_text,
                audit_logger=audit_logger,
                cached_content_name=cached_content_name,
            )
            candidate_checkpoint = build_candidate_checkpoint(
                candidates=candidates,
                selection_model_name=selection_model_name,
                selection_top_k=selection_top_k,
                allowed_question_types=allowed_question_types,
                candidate_score_threshold=candidate_score_threshold,
            )
    finally:
        if cached_content_name and gemini_client is not None:
            try:
                delete_context_cache(gemini_client, cached_content_name)
            except Exception as exc:
                print(
                    f"⚠️  explicit context cache 删除失败: {cached_content_name} | {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if candidate_checkpoint is None:
        raise ValueError("candidate checkpoint was not created")

    validate_candidate_checkpoint(
        candidate_checkpoint,
        selection_model_name=selection_model_name,
        selection_top_k=selection_top_k,
        allowed_question_types=allowed_question_types,
        candidate_score_threshold=candidate_score_threshold,
    )
    candidates = [candidate_from_payload(candidate) for candidate in candidate_checkpoint.candidates]
    if existing_source != "none":
        audit_logger.write(
            {
                "event": "candidate_checkpoint_loaded",
                "source": existing_source,
                "candidate_count": len(candidates),
            }
        )

    question_reject_reasons: dict[str, str] = {}
    for candidate in candidates:
        key = (candidate.coarse_unit_id, normalize_candidate_key(candidate.target_text))
        existing_reason = existing_ref_reject_reasons.get(key)
        if existing_reason:
            question_reject_reasons[candidate.candidate_id] = existing_reason

    eligible_candidates, question_filter_reject_reasons = filter_question_generation_candidates(
        candidates,
        candidate_score_threshold=candidate_score_threshold,
    )
    new_question_filter_rejects = []
    for candidate_id, reason in question_filter_reject_reasons.items():
        question_reject_reasons[candidate_id] = reason
        if candidate_id not in processed_candidate_ids:
            candidate = next(candidate for candidate in candidates if candidate.candidate_id == candidate_id)
            new_question_filter_rejects.append(reject_reason_to_record(candidate, reason))
            processed_candidate_ids.add(candidate_id)
    if new_question_filter_rejects:
        write_question_filter_rejects(audit_logger, new_question_filter_rejects)
    question_filter_rejection_count = len(new_question_filter_rejects)
    candidate_filtered_count = len(question_filter_reject_reasons)

    atomic_write_json(
        target_path=intermediate_output_path,
        temp_dir=temp_dir,
        input_path=mapped_json,
        payload=build_intermediate_output_payload(
            mapped_json=mapped_json,
            model_name=model_name,
            questions=existing_questions,
            ref_count=len(candidates),
            candidate_count=len(eligible_candidates),
            candidate_filtered_count=candidate_filtered_count,
            rejected_count=existing_rejected_count + question_filter_rejection_count,
            processed_candidate_ids=processed_candidate_ids,
            candidate_checkpoint=candidate_checkpoint,
            question_reject_reasons=question_reject_reasons,
        ),
    )

    log_step(f"ref_count: {len(candidates)}")
    log_step(f"candidate_count: {len(eligible_candidates)}")
    log_step(f"hard_filter_reject_count: {len(hard_rejects)}")
    log_step(f"candidate_filtered_count: {candidate_filtered_count}")

    existing_question_keys = {question_resume_key(question) for question in existing_questions}
    remaining_candidates = [
        candidate
        for candidate in eligible_candidates
        if candidate.candidate_id not in processed_candidate_ids
        and candidate_resume_key(candidate) not in existing_question_keys
    ]
    if existing_source != "none":
        audit_logger.write(
            {
                "event": "resume_loaded",
                "source": existing_source,
                "existing_question_count": len(existing_questions),
                "processed_candidate_count": len(processed_candidate_ids),
                "remaining_candidate_count": len(remaining_candidates),
            }
        )
    total_batches = count_candidate_batches(len(remaining_candidates), batch_size)
    log_step(f"remaining_candidate_count: {len(remaining_candidates)}")
    log_step(f"ai_batch_count: {total_batches}")

    final_questions: list[FinalQuestion] = list(existing_questions)
    ai_rejection_count = 0
    validation_rejection_count = 0

    for batch_no, batch in enumerate(chunk_candidates(remaining_candidates, batch_size), start=1):
        log_step(f"[AI batch {batch_no}/{total_batches}] candidates={len(batch)}", indent=1)
        messages = build_ai_messages(batch, allowed_question_types, full_transcript_text)
        batch_output = llm.invoke_question_batch(messages)
        batch_candidates_by_id = {candidate.candidate_id: candidate for candidate in batch}
        validate_batch_candidate_ids(batch_output, batch_candidates_by_id)

        for rejection in batch_output.rejections:
            ai_rejection_count += 1
            processed_candidate_ids.add(rejection.candidate_id)
            question_reject_reasons[rejection.candidate_id] = (
                f"question generation 阶段主动拒绝：{rejection.reason}"
            )
            audit_logger.write(
                {
                    "event": "ai_rejection",
                    "candidate_id": rejection.candidate_id,
                    "reason": rejection.reason,
                }
            )

        for result in batch_output.results:
            candidate = batch_candidates_by_id[result.candidate_id]
            if result.question_type not in allowed_question_types:
                validation_rejection_count += 1
                processed_candidate_ids.add(result.candidate_id)
                question_reject_reasons[result.candidate_id] = (
                    f"question validation 失败：unsupported question_type for this run: {result.question_type}"
                )
                audit_logger.write(
                    {
                        "event": "validation_reject",
                        "candidate_id": result.candidate_id,
                        "reason": f"unsupported question_type for this run: {result.question_type}",
                    }
                )
                continue
            try:
                final_question = merge_ai_result(candidate, result)
            except ValueError as exc:
                validation_rejection_count += 1
                processed_candidate_ids.add(result.candidate_id)
                question_reject_reasons[result.candidate_id] = f"question validation 失败：{exc}"
                audit_logger.write(
                    {
                        "event": "validation_reject",
                        "candidate_id": result.candidate_id,
                        "reason": str(exc),
                    }
                )
                continue
            final_questions.append(final_question)
            existing_question_keys.add(question_resume_key(final_question))
            processed_candidate_ids.add(result.candidate_id)
            question_reject_reasons.pop(result.candidate_id, None)
            audit_logger.write(
                {
                    "event": "accepted_question",
                    "candidate_id": result.candidate_id,
                    "question_type": result.question_type,
                    "coarse_unit_id": candidate.coarse_unit_id,
                    "target_text": candidate.target_text,
                }
            )

        intermediate_payload_dict = build_intermediate_output_payload(
            mapped_json=mapped_json,
            model_name=model_name,
            questions=final_questions,
            ref_count=len(candidates),
            candidate_count=len(eligible_candidates),
            candidate_filtered_count=candidate_filtered_count,
            rejected_count=(
                existing_rejected_count
                + question_filter_rejection_count
                + ai_rejection_count
                + validation_rejection_count
            ),
            processed_candidate_ids=processed_candidate_ids,
            candidate_checkpoint=candidate_checkpoint,
            question_reject_reasons=question_reject_reasons,
        )
        atomic_write_json(
            target_path=intermediate_output_path,
            temp_dir=temp_dir,
            input_path=mapped_json,
            payload=intermediate_payload_dict,
        )

    final_output = build_final_output(
        mapped_json=mapped_json,
        model_name=model_name,
        questions=final_questions,
        ref_count=len(candidates),
        candidate_count=len(eligible_candidates),
        candidate_filtered_count=candidate_filtered_count,
        rejected_count=(
            existing_rejected_count
            + question_filter_rejection_count
            + ai_rejection_count
            + validation_rejection_count
        ),
        candidate_checkpoint=candidate_checkpoint,
        question_reject_reasons=question_reject_reasons,
    )

    atomic_write_json(
        target_path=output_json,
        temp_dir=output_json.parent,
        input_path=mapped_json,
        payload=final_output.model_dump(),
    )
    if intermediate_output_path.exists():
        intermediate_output_path.unlink()
    return final_output


def build_final_output(
    mapped_json: Path,
    model_name: str,
    questions: list[FinalQuestion],
    ref_count: int,
    candidate_count: int,
    candidate_filtered_count: int,
    rejected_count: int,
    candidate_checkpoint: CandidateCheckpoint,
    question_reject_reasons: dict[str, str],
) -> FinalOutput:
    return FinalOutput(
        source=FinalSource(
            mapped_json=str(mapped_json),
            model=model_name,
        ),
        questions=questions,
        audit=FinalAudit(
            ref_count=ref_count,
            candidate_count=candidate_count,
            candidate_filtered_count=candidate_filtered_count,
            generated_count=len(questions),
            rejected_count=rejected_count,
        ),
        selected_coarse_unit_refs=build_selected_coarse_unit_refs(
            candidate_checkpoint,
            question_reject_reasons=question_reject_reasons,
        ),
    )


def build_intermediate_output_payload(
    mapped_json: Path,
    model_name: str,
    questions: list[FinalQuestion],
    ref_count: int,
    candidate_count: int,
    candidate_filtered_count: int,
    rejected_count: int,
    processed_candidate_ids: set[str],
    candidate_checkpoint: CandidateCheckpoint,
    question_reject_reasons: dict[str, str],
) -> dict[str, Any]:
    payload = build_final_output(
        mapped_json=mapped_json,
        model_name=model_name,
        questions=questions,
        ref_count=ref_count,
        candidate_count=candidate_count,
        candidate_filtered_count=candidate_filtered_count,
        rejected_count=rejected_count,
        candidate_checkpoint=candidate_checkpoint,
        question_reject_reasons=question_reject_reasons,
    ).model_dump()
    payload["audit"]["processed_candidate_ids"] = sorted(processed_candidate_ids)
    payload["candidate_checkpoint"] = candidate_checkpoint.model_dump()
    return payload


def get_gcloud_project() -> str | None:
    """读取当前 gcloud project，供未设置 GOOGLE_CLOUD_PROJECT 时使用。"""

    try:
        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    project = completed.stdout.strip()
    if completed.returncode != 0 or not project or project == "(unset)":
        return None
    return project


def create_gemini_client(env_path: Path) -> Any:
    """创建 Vertex AI Gemini client。认证走 Application Default Credentials。"""

    load_dotenv(env_path)
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or get_gcloud_project()
    )
    if not project:
        raise ValueError(
            "Google Cloud project is required. Set GOOGLE_CLOUD_PROJECT, "
            "or run: gcloud config set project <PROJECT_ID>"
        )
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_VERTEX_LOCATION")
        or DEFAULT_VERTEX_LOCATION
    )
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )


def create_gemini_question_llm(
    client: Any,
    model_name: str,
    thinking_level: str,
) -> GeminiQuestionLLM:
    return GeminiQuestionLLM(client, model_name, thinking_level.lower())


def create_gemini_context_selection_llm(
    client: Any,
    model_name: str,
    thinking_level: str,
) -> GeminiContextSelectionLLM:
    return GeminiContextSelectionLLM(client, model_name, thinking_level.lower())


def parse_question_types(value: str) -> list[str]:
    question_types = [item.strip() for item in value.split(",") if item.strip()]
    if not question_types:
        raise argparse.ArgumentTypeError("At least one question type is required")
    unsupported = sorted(set(question_types) - SUPPORTED_QUESTION_TYPES)
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported question types: {unsupported}")
    return question_types


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video-context quiz question JSON.")
    parser.add_argument("mapped_json", type=Path, help="Path to mapped transcript JSON.")
    parser.add_argument("output_questions_json", type=Path, help="Path to output question JSON.")
    parser.add_argument(
        "--question-types",
        type=parse_question_types,
        default=parse_question_types("context_meaning_choice,context_cloze_choice"),
        help="Comma-separated question types. Default: context_meaning_choice,context_cloze_choice.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of selected candidates per AI call. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=ROOT_DIR / ".env",
        help="Path to .env containing optional GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_QUESTION_MODEL,
        help=f"Gemini model for question generation. Default: {DEFAULT_QUESTION_MODEL}.",
    )
    parser.add_argument(
        "--question-thinking-level",
        default=DEFAULT_QUESTION_THINKING_LEVEL,
        choices=sorted(ALLOWED_GEMINI_THINKING_LEVELS),
        help=f"Gemini thinking level for question generation. Default: {DEFAULT_QUESTION_THINKING_LEVEL}.",
    )
    parser.add_argument(
        "--selection-model",
        default=DEFAULT_SELECTION_MODEL,
        help=f"Gemini model for context selection. Default: {DEFAULT_SELECTION_MODEL}.",
    )
    parser.add_argument(
        "--selection-thinking-level",
        default=DEFAULT_SELECTION_THINKING_LEVEL,
        choices=sorted(ALLOWED_GEMINI_THINKING_LEVELS),
        help=f"Gemini thinking level for context selection. Default: {DEFAULT_SELECTION_THINKING_LEVEL}.",
    )
    parser.add_argument(
        "--selection-top-k",
        type=int,
        default=DEFAULT_SELECTION_TOP_K,
        help=f"Top sentence candidates per coarse unit for context selection. Default: {DEFAULT_SELECTION_TOP_K}.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=DEFAULT_SELECTION_BATCH_SIZE,
        help=f"Context selection groups per AI call. Default: {DEFAULT_SELECTION_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--selection-max-workers",
        type=int,
        default=DEFAULT_SELECTION_MAX_WORKERS,
        help=f"Maximum concurrent context-selection AI calls. Default: {DEFAULT_SELECTION_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--candidate-score-threshold",
        type=float,
        default=DEFAULT_CANDIDATE_SCORE_THRESHOLD,
        help=f"Minimum weighted ref score to send to question generation. Default: {DEFAULT_CANDIDATE_SCORE_THRESHOLD}.",
    )
    parser.add_argument(
        "--video-gcs-uri",
        required=True,
        help="Required gs:// video URI used during context selection.",
    )
    parser.add_argument(
        "--video-mime-type",
        default=DEFAULT_VIDEO_MIME_TYPE,
        help=f"Video MIME type for --video-gcs-uri. Default: {DEFAULT_VIDEO_MIME_TYPE}.",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=DEFAULT_CACHE_TTL_SECONDS,
        help=f"Explicit context cache TTL in seconds. Default: {DEFAULT_CACHE_TTL_SECONDS}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gemini_client = create_gemini_client(args.env_path)
    llm = create_gemini_question_llm(
        client=gemini_client,
        model_name=args.model,
        thinking_level=args.question_thinking_level,
    )
    selection_llm = create_gemini_context_selection_llm(
        client=gemini_client,
        model_name=args.selection_model,
        thinking_level=args.selection_thinking_level,
    )

    log_header("启动题目生成")
    log_step(f"mapped_json: {args.mapped_json}")
    log_step(f"output: {args.output_questions_json}")
    log_step(f"model: {args.model}")
    log_step(f"question_thinking_level: {args.question_thinking_level}")
    log_step(f"selection_model: {args.selection_model}")
    log_step(f"selection_thinking_level: {args.selection_thinking_level}")
    log_step(f"selection_top_k: {args.selection_top_k}")
    log_step(f"selection_batch_size: {args.selection_batch_size}")
    log_step(f"selection_max_workers: {args.selection_max_workers}")
    log_step(f"candidate_score_threshold: {args.candidate_score_threshold}")
    log_step(f"video_gcs_uri: {args.video_gcs_uri}")
    log_step(f"question_types: {args.question_types}")
    log_step(f"batch_size: {args.batch_size}")

    final_output = run_generation(
        mapped_json=args.mapped_json,
        output_json=args.output_questions_json,
        allowed_question_types=args.question_types,
        batch_size=args.batch_size,
        llm=llm,
        model_name=args.model,
        gemini_client=gemini_client,
        selection_llm=selection_llm,
        selection_model_name=args.selection_model,
        selection_top_k=args.selection_top_k,
        selection_batch_size=args.selection_batch_size,
        selection_max_workers=args.selection_max_workers,
        candidate_score_threshold=args.candidate_score_threshold,
        video_gcs_uri=args.video_gcs_uri,
        video_mime_type=args.video_mime_type,
        cache_ttl_seconds=args.cache_ttl_seconds,
    )

    log_header("执行完成")
    log_step(f"ref_count: {final_output.audit.ref_count}")
    log_step(f"candidate_count: {final_output.audit.candidate_count}")
    log_step(f"candidate_filtered_count: {final_output.audit.candidate_filtered_count}")
    log_step(f"generated_count: {final_output.audit.generated_count}")
    log_step(f"rejected_count: {final_output.audit.rejected_count}")
    log_step(f"output: {args.output_questions_json}")


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print("\n=== 执行失败 ===", file=sys.stderr, flush=True)
        print(f"错误信息：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
