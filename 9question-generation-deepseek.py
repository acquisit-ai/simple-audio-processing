#!/usr/bin/env python3
"""
Generate video-context quiz questions from mapped transcript JSON.

This script reads the output produced by `5agent-mapping-deepseek.py`, filters
mapped semantic tokens globally, asks DeepSeek to submit question content through
a tool call, then fills stable pre-ingest question metadata in code.

Usage:
  python 9question-generation-deepseek.py <mapped_json> <output_questions_json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_QUESTION_MODEL = "deepseek-v4-pro"
DEFAULT_QUESTION_REASONING_EFFORT = "high"
DEFAULT_QUESTION_THINKING = {"type": "enabled"}
DEFAULT_SELECTION_MODEL = "deepseek-v4-flash"
DEFAULT_SELECTION_REASONING_EFFORT = "high"
DEFAULT_SELECTION_THINKING = {"type": "enabled"}
DEFAULT_SELECTION_TOP_K = 5
DEFAULT_SELECTION_BATCH_SIZE = 10
DEFAULT_SELECTION_MAX_WORKERS = 4
CHECKPOINT_VERSION = 1
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_QUESTIONS = 20
DEFAULT_TOOL_CALL_PARSE_MAX_ATTEMPTS = 3
SUPPORTED_QUESTION_TYPES = {"context_meaning_choice", "context_cloze_choice"}
EXPECTED_OPTION_IDS = ["correct", "wrong_1", "wrong_2", "wrong_3"]
SELECTION_TOOL_NAME = "submit_context_selection_batch"
QUESTION_TOOL_NAME = "submit_question_batch"

SELECTION_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": SELECTION_TOOL_NAME,
        "strict": True,
        "description": "提交每个 coarse group 的最佳上下文句子选择结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "description": "每个输入 group 必须且只能有一个选择结果。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_id": {
                                "type": "string",
                                "description": "当前输入 batch 中的 group_id。",
                            },
                            "sentence_candidate_id": {
                                "type": "string",
                                "description": "当前 group 中被选中的 sentence_candidate_id。",
                            },
                            "reason": {
                                "type": "string",
                                "description": "简短说明为什么这个句子最适合出题，只用于审计。",
                            },
                        },
                        "required": ["group_id", "sentence_candidate_id", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["selections"],
            "additionalProperties": False,
        },
    },
}

QUESTION_OUTPUT_TOOL = {
    "type": "function",
    "function": {
        "name": QUESTION_TOOL_NAME,
        "strict": True,
        "description": (
            "提交视频上下文题目内容和候选拒绝原因。"
            "只包含 AI 负责生成的题目内容，不包含数据库元数据。"
            "不要包含 coarse_id、sentence_index、token_index、start/end、status。"
            "不要为同一个 candidate 同时提交 result 和 rejection。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "适合出题的 candidate 对应的题目内容；candidate_id 必须来自当前输入 batch。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "description": "当前输入 batch 中的 candidate_id。",
                            },
                            "question_type": {
                                "type": "string",
                                "enum": [
                                    "context_meaning_choice",
                                    "context_cloze_choice",
                                ],
                                "description": (
                                    "从 allowed_question_types 中选择的题型。"
                                    "context_meaning_choice：给出上下文，询问目标词在当前语境中的意思。"
                                    "context_cloze_choice：把上下文里的目标词隐藏为 ____，不能泄露答案。"
                                ),
                            },
                            "content_payload": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "展示给学习者的题目问题文本，必须使用中文。",
                                    },
                                    "context_text": {
                                        "type": "string",
                                        "description": "展示给学习者的上下文文本；cloze 题的上下文不能泄露答案。",
                                    },
                                    "options": {
                                        "type": "array",
                                        "description": (
                                            "必须正好四个选项，顺序固定为 correct、wrong_1、wrong_2、wrong_3。正确选项必须是第一个。"
                                            "错误选项必须有迷惑性，但不能在当前语境和语法上也成立；如果选项放回原句后语义和语法都自然，它就不是错误选项。"
                                            "context_meaning_choice 的 options.text 必须使用中文释义。"
                                            "context_cloze_choice 的 options.text 应使用英文单词或短语。"
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {
                                                    "type": "string",
                                                    "enum": [
                                                        "correct",
                                                        "wrong_1",
                                                        "wrong_2",
                                                        "wrong_3",
                                                    ],
                                                },
                                                "text": {"type": "string"},
                                            },
                                            "required": ["id", "text"],
                                            "additionalProperties": False,
                                        },
                                    },
                                    "explanation": {
                                        "type": "string",
                                        "description": "必须使用中文，只说明正确选项为什么正确，不用说明错误选项为什么错。",
                                    },
                                },
                                "required": [
                                    "question",
                                    "context_text",
                                    "options",
                                    "explanation",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "candidate_id",
                            "question_type",
                            "content_payload",
                        ],
                        "additionalProperties": False,
                    },
                },
                "rejections": {
                    "type": "array",
                    "description": "不适合生成高质量题目的 candidate；candidate_id 必须来自当前输入 batch。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "description": "当前输入 batch 中的 candidate_id。",
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "简短说明为什么该 candidate 不适合出题。"
                                    "常见原因包括上下文不足、答案太显然、干扰项难以构造。"
                                ),
                            },
                        },
                        "required": ["candidate_id", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results", "rejections"],
            "additionalProperties": False,
        },
    },
}

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
    "this", "these", "those",
    "there", "here",
    "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "up", "down", "out", "off",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "please",
    "sorry",
    "thanks", "thank", "thank you",
    "bye", "goodbye",
    "hello", "hi",
    "nah", "nope",
    "ha", "haha", "hahaha",
    "yeah", "yep", "yup",
    "ok", "okay", "alright", "all right",
    "oh", "ooh", "ah", "uh", "um", "uhh", "umm", "hm", "hmm", "huh", "uh-huh",
    "wow", "whoa", "oops",
    "mm", "mmm", "mhm", "mm-hmm",
    "er", "erm",
    "hey",
}
LOW_VALUE_BASE_FORMS = DIRECT_NO_MATCH_BASEFORMS


QUESTION_SYSTEM_PROMPT = """
你是英语学习题目生成器。你只根据输入 candidates 生成视频上下文选择题内容。

ROLE AND BOUNDARY:
- 你只能生成题目内容，必须调用 `submit_question_batch` 工具提交结果，不要额外解释。
- 只根据当前输入 batch 的 candidates 判断是否生成题目。
- 题目面向中文读者：question 和 explanation 必须使用中文。
- context_meaning_choice 的 options.text 必须使用中文释义。
- context_cloze_choice 的 options.text 应使用英文单词或短语。
- context_text 保持英文视频原句或英文 cloze 上下文。

CONTENT EXAMPLES:
- 正面例子 context_meaning_choice：如果目标词是 sacred，当前句子是 "The most sacred thing I do is care and provide for my workers."，题目可以问“这里的 sacred 最接近什么意思？”。正确选项可表达“神圣、非常重要”。错误项可以是“普通、随便”“昂贵、奢侈”“快速、临时”，因为它们和当前语境下 sacred 的“重要、不可轻视”不一致。解释只说明 sacred 在这里是比喻用法，表示说话人认为这件事非常重要、不可轻视。
- 反面例子 context_meaning_choice：不要把“重要、有意义”作为 sacred 的错误项，因为它和正确含义过近，放在当前语境里也能成立；这类错误项为什么不好：它不是合格错误项，会让题目出现多个可接受答案。
- 正面例子 context_cloze_choice：如果目标短语是 provide for，当前句子是 "The most sacred thing I do is care and provide for my workers."，上下文应改写为 "The most sacred thing I do is care and ____ my workers."。正确选项应是 "provide for"。错误项可以是 "take off"、"work out"、"look up"，因为它们放回原句后语义或语法不自然。
- 反面例子 context_cloze_choice：不要把 "look after" 作为 provide for 的错误项；在 "care and ____ my workers" 里，它在语境和语法上也可能成立。它不是合格错误项，因为学习者选择它并不一定错。
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


class AIContextSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    sentence_candidate_id: str
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
    candidate_count: int
    generated_count: int
    rejected_count: int


class QuestionCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    target_text: str
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


class CandidateCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    selection_model: str
    selection_top_k: int
    allowed_question_types: list[str]
    candidates: list[QuestionCandidatePayload]


class SelectedCoarseUnitRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coarse_unit_id: int
    sentence_index: int
    token_index: int


class SelectedCoarseUnitRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    selection_model: str
    selection_top_k: int
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


class DeepSeekToolCallLLM:
    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def invoke_question_batch(self, messages: list[dict[str, str]]) -> AIQuestionBatchOutput:
        last_error: Exception | None = None
        for _ in range(DEFAULT_TOOL_CALL_PARSE_MAX_ATTEMPTS):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[QUESTION_OUTPUT_TOOL],
                reasoning_effort=DEFAULT_QUESTION_REASONING_EFFORT,
                extra_body={"thinking": DEFAULT_QUESTION_THINKING},
            )
            try:
                return parse_llm_tool_call_response(response.choices[0].message)
            except (TypeError, ValueError, ValidationError) as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("DeepSeek question batch returned no parse attempts")
        raise last_error


class DeepSeekContextSelectionLLM:
    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def invoke_context_selection_batch(self, messages: list[dict[str, str]]) -> AIContextSelectionBatchOutput:
        last_error: Exception | None = None
        for _ in range(DEFAULT_TOOL_CALL_PARSE_MAX_ATTEMPTS):
            request_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "tools": [SELECTION_OUTPUT_TOOL],
                "extra_body": {"thinking": DEFAULT_SELECTION_THINKING},
            }
            if DEFAULT_SELECTION_REASONING_EFFORT is not None:
                request_kwargs["reasoning_effort"] = DEFAULT_SELECTION_REASONING_EFFORT
            response = self.client.chat.completions.create(**request_kwargs)
            try:
                return parse_llm_context_selection_response(response.choices[0].message)
            except (TypeError, ValueError, ValidationError) as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("DeepSeek context selection returned no parse attempts")
        raise last_error


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


def clean_target_text(value: str) -> str:
    normalized = normalize_whitespace(value)
    return re.sub(r"^[^\w]+|[^\w]+$", "", normalized)


def is_numeric_text(value: str) -> bool:
    normalized = clean_target_text(value)
    return bool(normalized) and normalized.replace(".", "", 1).isdigit()


def is_punctuation_only(value: str) -> bool:
    return not bool(re.search(r"[A-Za-z0-9]", value))


def looks_like_name(token_text: str, base_form: str, pos: str) -> bool:
    cleaned = clean_target_text(token_text)
    if not cleaned or " " in cleaned or "-" in cleaned:
        return False
    if pos:
        return False
    return cleaned[:1].isupper() and base_form[:1].isupper()


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

            target_text = clean_target_text(str(token.get("text", "")))
            semantic = token.get("semantic_element")
            if not isinstance(semantic, dict):
                rejects.append(reject_record("semantic_element is missing", sentence, token, target_text))
                continue

            coarse_id = semantic.get("coarse_id")
            if coarse_id is None:
                rejects.append(reject_record("coarse_id is null", sentence, token, target_text))
                continue
            if not isinstance(coarse_id, int):
                rejects.append(reject_record("coarse_id is not an integer", sentence, token, target_text))
                continue

            if not {"index", "text", "start", "end"}.issubset(token):
                rejects.append(reject_record("token timing is missing", sentence, token, target_text))
                continue

            if not target_text:
                rejects.append(reject_record("target text is empty", sentence, token, target_text))
                continue
            if is_punctuation_only(target_text):
                rejects.append(reject_record("target text is punctuation only", sentence, token, target_text))
                continue
            if is_numeric_text(target_text):
                rejects.append(reject_record("target text is numeric", sentence, token, target_text))
                continue

            base_form = normalize_whitespace(str(semantic.get("base_form", "")).strip())
            translation = normalize_whitespace(str(semantic.get("translation", "")).strip())
            dictionary = normalize_whitespace(str(semantic.get("dictionary", "")).strip())
            kind = normalize_lookup_text(str(semantic.get("kind", "")))
            pos = normalize_lookup_text(str(semantic.get("pos", "")))
            if not translation or not dictionary:
                rejects.append(reject_record("semantic translation or dictionary is missing", sentence, token, target_text))
                continue

            lookup_values = {
                normalize_lookup_text(target_text),
                normalize_lookup_text(base_form),
            }
            lookup_values.discard("")
            if lookup_values & LOW_VALUE_BASE_FORMS:
                rejects.append(reject_record("target is a low-value function word", sentence, token, target_text))
                continue
            if looks_like_name(str(token.get("text", "")), base_form, pos):
                rejects.append(reject_record("target looks like a proper name", sentence, token, target_text))
                continue

            occurrences.append(
                QuestionOccurrence(
                    target_text=target_text,
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
    max_groups: int,
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
    for group_index, (best_occurrence, group_occurrences) in enumerate(group_items[:max_groups], start=1):
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
        selected.append(
            QuestionCandidate(
                candidate_id=f"c_{len(selected) + 1:06d}",
                target_text=sentence_candidate.target_text,
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
            )
        )

    missing_group_ids = sorted(set(groups_by_id) - seen_group_ids)
    if missing_group_ids:
        raise ValueError(f"missing selections for groups: {missing_group_ids}")
    return selected


def extract_question_candidates(
    mapped_payload: dict[str, Any],
    allowed_question_types: list[str],
    max_candidates: int,
) -> tuple[list[QuestionCandidate], list[CandidateReject]]:
    occurrences, rejects = extract_question_occurrences(
        mapped_payload,
        allowed_question_types=allowed_question_types,
    )
    groups = build_context_selection_groups(
        occurrences,
        top_k=1,
        max_groups=max_candidates,
    )
    selection_output = AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": group.group_id,
                    "sentence_candidate_id": group.sentence_candidates[0].sentence_candidate_id,
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
) -> AIContextSelectionBatchOutput:
    selections_by_group_id: dict[str, AIContextSelection] = {}
    llm_groups: list[ContextSelectionGroup] = []
    for group in groups:
        if len(group.sentence_candidates) == 1:
            selections_by_group_id[group.group_id] = AIContextSelection(
                group_id=group.group_id,
                sentence_candidate_id=group.sentence_candidates[0].sentence_candidate_id,
                reason="only one sentence candidate",
            )
        else:
            llm_groups.append(group)

    if llm_groups and selection_llm is None:
        raise ValueError("selection_llm is required when a coarse group has multiple sentence candidates")

    def invoke_batch(batch: list[ContextSelectionGroup]) -> AIContextSelectionBatchOutput:
        messages = build_context_selection_messages(batch, full_transcript_text)
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


def read_message_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def parse_first_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if not stripped:
        raise ValueError("DeepSeek tool call arguments are empty")
    try:
        payload, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek tool call arguments are invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek tool call arguments must be a JSON object")
    return payload


def parse_tool_call_arguments(message: Any, expected_tool_name: str) -> dict[str, Any]:
    tool_calls = read_message_field(message, "tool_calls")
    if not tool_calls:
        raise ValueError("DeepSeek did not return tool_calls")

    tool_call = tool_calls[0]
    function = read_message_field(tool_call, "function")
    if function is None:
        raise ValueError("DeepSeek tool call is missing function")

    function_name = read_message_field(function, "name")
    if function_name != expected_tool_name:
        raise ValueError(f"DeepSeek called unsupported tool: {function_name}")

    arguments = read_message_field(function, "arguments")
    if not isinstance(arguments, str):
        raise TypeError("DeepSeek tool call arguments must be a JSON string")

    payload = parse_first_json_object(arguments)
    return payload


def parse_llm_tool_call_response(message: Any) -> AIQuestionBatchOutput:
    payload = parse_tool_call_arguments(message, QUESTION_TOOL_NAME)
    return AIQuestionBatchOutput.model_validate(payload)


def parse_llm_context_selection_response(message: Any) -> AIContextSelectionBatchOutput:
    payload = parse_tool_call_arguments(message, SELECTION_TOOL_NAME)
    return AIContextSelectionBatchOutput.model_validate(payload)


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
) -> list[dict[str, str]]:
    batch_payload = {
        "groups": [group.to_selection_payload() for group in groups],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是英语学习题目上下文选择器。"
                "你必须调用 submit_context_selection_batch 工具。"
                "每个 group 必须选择且只能选择一个最适合生成视频上下文题的句子。"
                "优先选择目标词或短语在句子里的含义清楚、不是只靠前后文才能猜到的句子。"
                "优先选择能自然生成中文释义选择题或英文 cloze 选择题的句子："
                "句子应有足够上下文，目标词和周围搭配关系明确，隐藏目标词后仍能构造唯一合理答案。"
                "优先选择真实表达完整、语气和场景信息足够的句子。"
                "不要优先选择纯寒暄、残句、引用不完整、代词指代过重、过短、过长、"
                "只有语法功能但学习价值低，或目标词在该句里只是人名/数字/噪声的句子。"
                "如果多个句子都可用，选择最适合让中文读者理解该 coarse unit 核心含义的句子。"
            ),
        },
        {
            "role": "user",
            "content": (
                "FULL_VIDEO_TRANSCRIPT:\n"
                f"{full_transcript_text}\n\n"
                "CURRENT_CONTEXT_SELECTION_GROUPS:\n"
                + json.dumps(batch_payload, ensure_ascii=False, indent=2)
            ),
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
    return asdict(candidate)


def candidate_from_payload(payload: QuestionCandidatePayload) -> QuestionCandidate:
    return QuestionCandidate(**payload.model_dump())


def build_candidate_checkpoint(
    candidates: list[QuestionCandidate],
    selection_model_name: str,
    selection_top_k: int,
    allowed_question_types: list[str],
) -> CandidateCheckpoint:
    return CandidateCheckpoint(
        version=CHECKPOINT_VERSION,
        selection_model=selection_model_name,
        selection_top_k=selection_top_k,
        allowed_question_types=list(allowed_question_types),
        candidates=[
            QuestionCandidatePayload.model_validate(candidate_to_payload(candidate))
            for candidate in candidates
        ],
    )


def validate_candidate_checkpoint(
    checkpoint: CandidateCheckpoint,
    selection_model_name: str,
    selection_top_k: int,
    allowed_question_types: list[str],
) -> None:
    if checkpoint.version != CHECKPOINT_VERSION:
        raise ValueError(f"candidate checkpoint version mismatch: {checkpoint.version}")
    if checkpoint.selection_model != selection_model_name:
        raise ValueError(f"candidate checkpoint selection model mismatch: {checkpoint.selection_model}")
    if checkpoint.selection_top_k != selection_top_k:
        raise ValueError(f"candidate checkpoint selection_top_k mismatch: {checkpoint.selection_top_k}")
    if checkpoint.allowed_question_types != list(allowed_question_types):
        raise ValueError("candidate checkpoint allowed_question_types mismatch")

    seen_candidate_ids: set[str] = set()
    for candidate in checkpoint.candidates:
        if candidate.candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate candidate_id in checkpoint: {candidate.candidate_id}")
        seen_candidate_ids.add(candidate.candidate_id)


def build_selected_coarse_unit_refs(
    candidate_checkpoint: CandidateCheckpoint,
) -> SelectedCoarseUnitRefs:
    return SelectedCoarseUnitRefs(
        version=candidate_checkpoint.version,
        selection_model=candidate_checkpoint.selection_model,
        selection_top_k=candidate_checkpoint.selection_top_k,
        allowed_question_types=list(candidate_checkpoint.allowed_question_types),
        refs=[
            SelectedCoarseUnitRef(
                coarse_unit_id=candidate.coarse_unit_id,
                sentence_index=candidate.sentence_index,
                token_index=candidate.token_index,
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
) -> tuple[list[FinalQuestion], set[str], int, str, CandidateCheckpoint | None]:
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
        return [], set(), 0, source_label, None

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
        )

    return questions, processed_candidate_ids, rejected_count, source_label, checkpoint


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
    max_candidates: int,
    selection_top_k: int,
    selection_batch_size: int,
    selection_max_workers: int,
    selection_llm: Any,
    full_transcript_text: str,
    audit_logger: AuditLogger,
) -> tuple[list[QuestionCandidate], list[CandidateReject]]:
    occurrences, hard_rejects = extract_question_occurrences(
        mapped_payload,
        allowed_question_types=allowed_question_types,
    )
    write_hard_filter_rejects(audit_logger, hard_rejects)

    groups = build_context_selection_groups(
        occurrences,
        top_k=selection_top_k,
        max_groups=max_candidates,
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
    max_questions: int,
    batch_size: int,
    llm: Any,
    model_name: str,
    selection_llm: Any | None = None,
    selection_model_name: str = DEFAULT_SELECTION_MODEL,
    selection_top_k: int = DEFAULT_SELECTION_TOP_K,
    selection_batch_size: int = DEFAULT_SELECTION_BATCH_SIZE,
    selection_max_workers: int = DEFAULT_SELECTION_MAX_WORKERS,
) -> FinalOutput:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_questions < 1:
        raise ValueError("max_questions must be at least 1")
    if selection_top_k < 1:
        raise ValueError("selection_top_k must be at least 1")
    if selection_batch_size < 1:
        raise ValueError("selection_batch_size must be at least 1")
    if selection_max_workers < 1:
        raise ValueError("selection_max_workers must be at least 1")

    mapped_payload = load_json(mapped_json)
    full_transcript_text = build_full_transcript_text(mapped_payload)
    temp_dir, log_dir = ensure_output_dirs(output_json)
    audit_logger = AuditLogger(log_dir / f"{mapped_json.name}.question_audit.jsonl")
    intermediate_output_path = get_intermediate_output_path(output_json)

    existing_questions, processed_candidate_ids, existing_rejected_count, existing_source, candidate_checkpoint = (
        load_existing_question_output(
            output_json,
            mapped_json=mapped_json,
            model_name=model_name,
            allowed_question_types=allowed_question_types,
            selection_model_name=selection_model_name,
            selection_top_k=selection_top_k,
        )
    )
    if len(existing_questions) > max_questions:
        raise ValueError("existing question count exceeds max_questions")

    candidate_pool_limit = max(max_questions * 4, batch_size)
    hard_rejects: list[CandidateReject] = []
    if candidate_checkpoint is None:
        candidates, hard_rejects = select_question_candidates(
            mapped_payload=mapped_payload,
            allowed_question_types=allowed_question_types,
            max_candidates=candidate_pool_limit,
            selection_top_k=selection_top_k,
            selection_batch_size=selection_batch_size,
            selection_max_workers=selection_max_workers,
            selection_llm=selection_llm,
            full_transcript_text=full_transcript_text,
            audit_logger=audit_logger,
        )
        candidate_checkpoint = build_candidate_checkpoint(
            candidates=candidates,
            selection_model_name=selection_model_name,
            selection_top_k=selection_top_k,
            allowed_question_types=allowed_question_types,
        )
        atomic_write_json(
            target_path=intermediate_output_path,
            temp_dir=temp_dir,
            input_path=mapped_json,
            payload=build_intermediate_output_payload(
                mapped_json=mapped_json,
                model_name=model_name,
                questions=existing_questions,
                candidate_count=len(candidates),
                rejected_count=existing_rejected_count,
                processed_candidate_ids=processed_candidate_ids,
                candidate_checkpoint=candidate_checkpoint,
            ),
        )
    else:
        validate_candidate_checkpoint(
            candidate_checkpoint,
            selection_model_name=selection_model_name,
            selection_top_k=selection_top_k,
            allowed_question_types=allowed_question_types,
        )
        candidates = [candidate_from_payload(candidate) for candidate in candidate_checkpoint.candidates]
        audit_logger.write(
            {
                "event": "candidate_checkpoint_loaded",
                "source": existing_source,
                "candidate_count": len(candidates),
            }
        )

    log_step(f"candidate_count: {len(candidates)}")
    log_step(f"hard_filter_reject_count: {len(hard_rejects)}")

    existing_question_keys = {question_resume_key(question) for question in existing_questions}
    remaining_candidates = [
        candidate
        for candidate in candidates
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
        if len(final_questions) >= max_questions:
            break
        log_step(f"[AI batch {batch_no}/{total_batches}] candidates={len(batch)}", indent=1)
        messages = build_ai_messages(batch, allowed_question_types, full_transcript_text)
        batch_output = llm.invoke_question_batch(messages)
        batch_candidates_by_id = {candidate.candidate_id: candidate for candidate in batch}
        validate_batch_candidate_ids(batch_output, batch_candidates_by_id)

        for rejection in batch_output.rejections:
            ai_rejection_count += 1
            processed_candidate_ids.add(rejection.candidate_id)
            audit_logger.write(
                {
                    "event": "ai_rejection",
                    "candidate_id": rejection.candidate_id,
                    "reason": rejection.reason,
                }
            )

        for result in batch_output.results:
            if len(final_questions) >= max_questions:
                break
            candidate = batch_candidates_by_id[result.candidate_id]
            if result.question_type not in allowed_question_types:
                validation_rejection_count += 1
                processed_candidate_ids.add(result.candidate_id)
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
            candidate_count=len(candidates),
            rejected_count=existing_rejected_count + ai_rejection_count + validation_rejection_count,
            processed_candidate_ids=processed_candidate_ids,
            candidate_checkpoint=candidate_checkpoint,
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
        candidate_count=len(candidates),
        rejected_count=existing_rejected_count + ai_rejection_count + validation_rejection_count,
        candidate_checkpoint=candidate_checkpoint,
    )
    if len(final_output.questions) > max_questions:
        raise ValueError("final question count exceeds max_questions")

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
    candidate_count: int,
    rejected_count: int,
    candidate_checkpoint: CandidateCheckpoint,
) -> FinalOutput:
    return FinalOutput(
        source=FinalSource(
            mapped_json=str(mapped_json),
            model=model_name,
        ),
        questions=questions,
        audit=FinalAudit(
            candidate_count=candidate_count,
            generated_count=len(questions),
            rejected_count=rejected_count,
        ),
        selected_coarse_unit_refs=build_selected_coarse_unit_refs(candidate_checkpoint),
    )


def build_intermediate_output_payload(
    mapped_json: Path,
    model_name: str,
    questions: list[FinalQuestion],
    candidate_count: int,
    rejected_count: int,
    processed_candidate_ids: set[str],
    candidate_checkpoint: CandidateCheckpoint,
) -> dict[str, Any]:
    payload = build_final_output(
        mapped_json=mapped_json,
        model_name=model_name,
        questions=questions,
        candidate_count=candidate_count,
        rejected_count=rejected_count,
        candidate_checkpoint=candidate_checkpoint,
    ).model_dump()
    payload["audit"]["processed_candidate_ids"] = sorted(processed_candidate_ids)
    payload["candidate_checkpoint"] = candidate_checkpoint.model_dump()
    return payload


def load_deepseek_config(env_path: Path) -> tuple[str, str]:
    env_values = dotenv_values(env_path) if env_path.exists() else {}
    api_key = env_values.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(f"DEEPSEEK_API_KEY not found in {env_path}")
    base_url = (
        env_values.get("DEEPSEEK_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or DEFAULT_DEEPSEEK_BASE_URL
    )
    return api_key, base_url


def create_deepseek_tool_call_llm(env_path: Path, model_name: str) -> DeepSeekToolCallLLM:
    api_key, base_url = load_deepseek_config(env_path)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return DeepSeekToolCallLLM(client, model_name)


def create_deepseek_context_selection_llm(env_path: Path, model_name: str) -> DeepSeekContextSelectionLLM:
    api_key, base_url = load_deepseek_config(env_path)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return DeepSeekContextSelectionLLM(client, model_name)


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
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help=f"Maximum final questions to generate. Default: {DEFAULT_MAX_QUESTIONS}.",
    )
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
        help="Path to .env containing DEEPSEEK_API_KEY and optional DEEPSEEK_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_QUESTION_MODEL,
        help=f"DeepSeek model for question generation. Default: {DEFAULT_QUESTION_MODEL}.",
    )
    parser.add_argument(
        "--selection-model",
        default=DEFAULT_SELECTION_MODEL,
        help=f"DeepSeek model for context selection. Default: {DEFAULT_SELECTION_MODEL}.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    llm = create_deepseek_tool_call_llm(args.env_path, args.model)
    selection_llm = create_deepseek_context_selection_llm(args.env_path, args.selection_model)

    log_header("启动题目生成")
    log_step(f"mapped_json: {args.mapped_json}")
    log_step(f"output: {args.output_questions_json}")
    log_step(f"model: {args.model}")
    log_step(f"selection_model: {args.selection_model}")
    log_step(f"selection_top_k: {args.selection_top_k}")
    log_step(f"selection_batch_size: {args.selection_batch_size}")
    log_step(f"selection_max_workers: {args.selection_max_workers}")
    log_step(f"question_types: {args.question_types}")
    log_step(f"max_questions: {args.max_questions}")
    log_step(f"batch_size: {args.batch_size}")

    final_output = run_generation(
        mapped_json=args.mapped_json,
        output_json=args.output_questions_json,
        allowed_question_types=args.question_types,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
        llm=llm,
        model_name=args.model,
        selection_llm=selection_llm,
        selection_model_name=args.selection_model,
        selection_top_k=args.selection_top_k,
        selection_batch_size=args.selection_batch_size,
        selection_max_workers=args.selection_max_workers,
    )

    log_header("执行完成")
    log_step(f"candidate_count: {final_output.audit.candidate_count}")
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
