#!/usr/bin/env python3
"""
Generate video-context quiz questions from mapped transcript JSON.

This script reads the output produced by `5agent-mapping-deepseek.py`, filters
mapped semantic tokens globally, asks DeepSeek for question content only, then
fills stable `catalog.questions` metadata in code.

Usage:
  python 9question-generation-deepseek.py <mapped_json> <output_questions_json> \
    --video-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_QUESTION_MODEL = "deepseek-v4-pro"
DEFAULT_QUESTION_REASONING_EFFORT = "max"
DEFAULT_QUESTION_THINKING = {"type": "enabled"}
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_QUESTIONS = 20
SUPPORTED_QUESTION_TYPES = {"context_meaning_choice", "context_cloze_choice"}
EXPECTED_OPTION_IDS = ["correct", "wrong_1", "wrong_2", "wrong_3"]

LOW_VALUE_BASE_FORMS = {
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
}


QUESTION_SYSTEM_PROMPT = """
你是英语学习题目生成器。你只根据输入 candidates 生成视频上下文选择题内容。

ROLE AND BOUNDARY:
- 你只能生成题目内容，只输出合法 JSON object，不要额外解释。
- 你可以为适合出题的 candidate 写入 results，也可以为不适合出题的 candidate 写入 rejections。
- 不要为同一个 candidate 同时写 result 和 rejection。

OUTPUT JSON SCHEMA:
{
  "results": [
    {
      "candidate_id": "string, 必须来自当前输入 batch",
      "question_type": "context_meaning_choice | context_cloze_choice",
      "content_payload": {
        "question": "string, 题目问题文本",
        "context_text": "string, 前端展示的上下文文本",
        "options": [
          { "id": "correct", "text": "string, 正确选项" },
          { "id": "wrong_1", "text": "string, 错误选项" },
          { "id": "wrong_2", "text": "string, 错误选项" },
          { "id": "wrong_3", "text": "string, 错误选项" }
        ],
        "explanation": "string, 只说明正确选项为什么对"
      }
    }
  ],
  "rejections": [
    {
      "candidate_id": "string, 必须来自当前输入 batch",
      "reason": "string, 简短说明为什么不适合生成题目"
    }
  ]
}

OUTPUT RULES:
- 顶层必须只有 results 和 rejections 两个字段。
- result 必须只有 candidate_id、question_type、content_payload 三个字段。
- content_payload 必须只有 question、context_text、options、explanation 四个字段。
- options 必须正好 4 个，顺序固定为 correct、wrong_1、wrong_2、wrong_3。
- 正确选项必须是 options[0]，id 必须是 "correct"。
- 错误选项必须有迷惑性，但不能是正确答案的同义或近义表达。
- explanation 只说明正确选项为什么对，不用说明错误选项为什么错。

QUESTION TYPE RULES:
- context_meaning_choice：给出上下文，询问目标词在当前语境中的意思。
- context_cloze_choice：把上下文里的目标词隐藏为 ____，不能泄露答案。
- 如果某个 candidate 的上下文不足、答案太显然、干扰项难以构造，放入 rejections。

EXAMPLE context_meaning_choice:
{
  "candidate_id": "c_000001",
  "question_type": "context_meaning_choice",
  "content_payload": {
    "question": "这里的 “sacred” 最接近什么意思？",
    "context_text": "The most sacred thing I do is care and provide for my workers.",
    "options": [
      { "id": "correct", "text": "神圣、非常重要" },
      { "id": "wrong_1", "text": "普通、随便" },
      { "id": "wrong_2", "text": "昂贵、奢侈" },
      { "id": "wrong_3", "text": "快速、临时" }
    ],
    "explanation": "sacred 在这里是比喻用法，表示说话人认为这件事非常重要、不可轻视。"
  }
}

EXAMPLE context_cloze_choice:
{
  "candidate_id": "c_000002",
  "question_type": "context_cloze_choice",
  "content_payload": {
    "question": "根据上下文，空格处最合适的是哪一个？",
    "context_text": "The most sacred thing I do is care and ____ my workers.",
    "options": [
      { "id": "correct", "text": "provide for" },
      { "id": "wrong_1", "text": "take off" },
      { "id": "wrong_2", "text": "work out" },
      { "id": "wrong_3", "text": "look up" }
    ],
    "explanation": "provide for someone 表示供养、养活某人，符合 workers 这个宾语。"
  }
}
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


class FinalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_type: Literal["video_unit"]
    question_type: Literal["context_meaning_choice", "context_cloze_choice"]
    coarse_unit_id: int
    target_text: str
    video_id: str
    context_sentence_index: int
    context_span_index: int
    context_start_ms: int
    context_end_ms: int
    content_payload: dict[str, Any]
    status: Literal["draft"]


class FinalSource(BaseModel):
    mapped_json: str
    video_id: str
    model: str


class FinalAudit(BaseModel):
    candidate_count: int
    generated_count: int
    rejected_count: int


class FinalOutput(BaseModel):
    source: FinalSource
    questions: list[FinalQuestion]
    audit: FinalAudit


@dataclass(frozen=True)
class QuestionCandidate:
    candidate_id: str
    target_text: str
    base_form: str
    coarse_unit_id: int
    coarse_label: str
    coarse_definition: str
    kind: str
    pos: str
    sentence_index: int
    sentence_text: str
    sentence_translation: str
    sentence_start_ms: int
    sentence_end_ms: int
    token_index: int
    token_explanation: str
    score: int

    def to_ai_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_text": self.target_text,
            "base_form": self.base_form,
            "coarse_label": self.coarse_label,
            "coarse_definition": self.coarse_definition,
            "kind": self.kind,
            "pos": self.pos,
            "sentence_text": self.sentence_text,
            "sentence_translation": self.sentence_translation,
            "token_explanation": self.token_explanation,
        }


@dataclass(frozen=True)
class CandidateReject:
    candidate_id: str | None
    sentence_index: int | None
    token_index: int | None
    target_text: str | None
    reason: str


class DeepSeekJsonLLM:
    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def invoke_json(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort=DEFAULT_QUESTION_REASONING_EFFORT,
            extra_body={"thinking": DEFAULT_QUESTION_THINKING},
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        if not isinstance(content, str):
            raise TypeError("DeepSeek returned a non-string content payload")
        return content


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


def extract_question_candidates(
    mapped_payload: dict[str, Any],
    allowed_question_types: list[str],
    max_candidates: int,
    max_per_sentence: int = 2,
    max_per_coarse_unit: int = 1,
) -> tuple[list[QuestionCandidate], list[CandidateReject]]:
    if not set(allowed_question_types).issubset(SUPPORTED_QUESTION_TYPES):
        unsupported = sorted(set(allowed_question_types) - SUPPORTED_QUESTION_TYPES)
        raise ValueError(f"Unsupported question types: {unsupported}")

    sentences = mapped_payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError("Mapped JSON must contain a sentences array")

    rejects: list[CandidateReject] = []
    rough_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()

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

            dedupe_key = (coarse_id, normalize_candidate_key(target_text))
            if dedupe_key in seen_keys:
                rejects.append(reject_record("duplicate candidate key", sentence, token, target_text))
                continue
            seen_keys.add(dedupe_key)

            rough_candidates.append(
                {
                    "target_text": target_text,
                    "base_form": base_form,
                    "coarse_unit_id": coarse_id,
                    "coarse_label": translation,
                    "coarse_definition": dictionary,
                    "kind": kind,
                    "pos": pos,
                    "sentence_index": int(sentence["index"]),
                    "sentence_text": normalize_whitespace(str(sentence["text"])),
                    "sentence_translation": normalize_whitespace(str(sentence["translation"])),
                    "sentence_start_ms": int(sentence["start"]),
                    "sentence_end_ms": int(sentence["end"]),
                    "token_index": int(token["index"]),
                    "token_explanation": normalize_whitespace(str(token.get("explanation", ""))),
                    "score": score_candidate(kind, pos, target_text, str(sentence["text"])),
                }
            )

    rough_candidates.sort(
        key=lambda item: (
            -item["score"],
            item["sentence_index"],
            item["token_index"],
            item["target_text"].lower(),
        )
    )

    selected: list[QuestionCandidate] = []
    sentence_counts: dict[int, int] = {}
    coarse_counts: dict[int, int] = {}
    for item in rough_candidates:
        sentence_index = item["sentence_index"]
        coarse_id = item["coarse_unit_id"]
        if sentence_counts.get(sentence_index, 0) >= max_per_sentence:
            rejects.append(
                CandidateReject(
                    candidate_id=None,
                    sentence_index=sentence_index,
                    token_index=item["token_index"],
                    target_text=item["target_text"],
                    reason="sentence candidate limit reached",
                )
            )
            continue
        if coarse_counts.get(coarse_id, 0) >= max_per_coarse_unit:
            rejects.append(
                CandidateReject(
                    candidate_id=None,
                    sentence_index=sentence_index,
                    token_index=item["token_index"],
                    target_text=item["target_text"],
                    reason="coarse unit candidate limit reached",
                )
            )
            continue
        if len(selected) >= max_candidates:
            rejects.append(
                CandidateReject(
                    candidate_id=None,
                    sentence_index=sentence_index,
                    token_index=item["token_index"],
                    target_text=item["target_text"],
                    reason="max candidate limit reached",
                )
            )
            continue

        candidate_id = f"c_{len(selected) + 1:06d}"
        selected.append(QuestionCandidate(candidate_id=candidate_id, **item))
        sentence_counts[sentence_index] = sentence_counts.get(sentence_index, 0) + 1
        coarse_counts[coarse_id] = coarse_counts.get(coarse_id, 0) + 1

    return selected, rejects


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
    video_id: str,
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
        video_id=video_id,
        context_sentence_index=candidate.sentence_index,
        context_span_index=candidate.token_index,
        context_start_ms=candidate.sentence_start_ms,
        context_end_ms=candidate.sentence_end_ms,
        content_payload=ai_result.content_payload.model_dump(),
        status="draft",
    )


def parse_llm_json_response(content: str) -> AIQuestionBatchOutput:
    stripped = content.strip()
    if not stripped:
        raise ValueError("LLM returned empty content")
    if stripped.startswith("```"):
        raise ValueError("LLM returned Markdown fenced content instead of raw JSON")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object")
    return AIQuestionBatchOutput.model_validate(payload)


def build_ai_messages(
    candidates: list[QuestionCandidate],
    allowed_question_types: list[str],
) -> list[dict[str, str]]:
    batch_payload = {
        "candidates": [candidate.to_ai_payload() for candidate in candidates],
        "allowed_question_types": allowed_question_types,
    }
    return [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "请只基于下面 JSON 生成题目内容：\n"
            + json.dumps(batch_payload, ensure_ascii=False, indent=2),
        },
    ]


def chunk_candidates(candidates: list[QuestionCandidate], batch_size: int) -> list[list[QuestionCandidate]]:
    return [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON file must contain an object")
    return payload


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


def ensure_output_dirs(output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "temp"
    log_dir = output_path.parent / "log"
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, log_dir


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
    video_id: str,
    allowed_question_types: list[str],
    max_questions: int,
    batch_size: int,
    llm: Any,
    model_name: str,
) -> FinalOutput:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_questions < 1:
        raise ValueError("max_questions must be at least 1")

    mapped_payload = load_json(mapped_json)
    temp_dir, log_dir = ensure_output_dirs(output_json)
    audit_logger = AuditLogger(log_dir / f"{mapped_json.name}.question_audit.jsonl")

    candidate_pool_limit = max(max_questions * 4, batch_size)
    candidates, hard_rejects = extract_question_candidates(
        mapped_payload,
        allowed_question_types=allowed_question_types,
        max_candidates=candidate_pool_limit,
    )
    write_hard_filter_rejects(audit_logger, hard_rejects)

    final_questions: list[FinalQuestion] = []
    ai_rejection_count = 0
    validation_rejection_count = 0

    for batch_no, batch in enumerate(chunk_candidates(candidates, batch_size), start=1):
        if len(final_questions) >= max_questions:
            break
        log_step(f"[AI batch {batch_no}] candidates={len(batch)}", indent=1)
        messages = build_ai_messages(batch, allowed_question_types)
        batch_output = parse_llm_json_response(llm.invoke_json(messages))
        batch_candidates_by_id = {candidate.candidate_id: candidate for candidate in batch}
        validate_batch_candidate_ids(batch_output, batch_candidates_by_id)

        for rejection in batch_output.rejections:
            ai_rejection_count += 1
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
                audit_logger.write(
                    {
                        "event": "validation_reject",
                        "candidate_id": result.candidate_id,
                        "reason": f"unsupported question_type for this run: {result.question_type}",
                    }
                )
                continue
            try:
                final_question = merge_ai_result(candidate, result, video_id=video_id)
            except ValueError as exc:
                validation_rejection_count += 1
                audit_logger.write(
                    {
                        "event": "validation_reject",
                        "candidate_id": result.candidate_id,
                        "reason": str(exc),
                    }
                )
                continue
            final_questions.append(final_question)
            audit_logger.write(
                {
                    "event": "accepted_question",
                    "candidate_id": result.candidate_id,
                    "question_type": result.question_type,
                    "coarse_unit_id": candidate.coarse_unit_id,
                    "target_text": candidate.target_text,
                }
            )

        intermediate_payload = build_final_output(
            mapped_json=mapped_json,
            video_id=video_id,
            model_name=model_name,
            questions=final_questions,
            candidate_count=len(candidates),
            rejected_count=ai_rejection_count + validation_rejection_count,
        )
        atomic_write_json(
            target_path=output_json.parent / "temp" / output_json.name,
            temp_dir=temp_dir,
            input_path=mapped_json,
            payload=intermediate_payload.model_dump(),
        )

    final_output = build_final_output(
        mapped_json=mapped_json,
        video_id=video_id,
        model_name=model_name,
        questions=final_questions,
        candidate_count=len(candidates),
        rejected_count=ai_rejection_count + validation_rejection_count,
    )
    if len(final_output.questions) > max_questions:
        raise ValueError("final question count exceeds max_questions")

    atomic_write_json(
        target_path=output_json,
        temp_dir=output_json.parent,
        input_path=mapped_json,
        payload=final_output.model_dump(),
    )
    intermediate_path = output_json.parent / "temp" / output_json.name
    if intermediate_path.exists():
        intermediate_path.unlink()
    return final_output


def build_final_output(
    mapped_json: Path,
    video_id: str,
    model_name: str,
    questions: list[FinalQuestion],
    candidate_count: int,
    rejected_count: int,
) -> FinalOutput:
    return FinalOutput(
        source=FinalSource(
            mapped_json=str(mapped_json),
            video_id=video_id,
            model=model_name,
        ),
        questions=questions,
        audit=FinalAudit(
            candidate_count=candidate_count,
            generated_count=len(questions),
            rejected_count=rejected_count,
        ),
    )


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


def create_deepseek_json_llm(env_path: Path, model_name: str) -> DeepSeekJsonLLM:
    api_key, base_url = load_deepseek_config(env_path)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return DeepSeekJsonLLM(client, model_name)


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
    parser.add_argument("--video-id", required=True, help="catalog.videos.video_id for video_unit questions.")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    llm = create_deepseek_json_llm(args.env_path, args.model)

    log_header("启动题目生成")
    log_step(f"mapped_json: {args.mapped_json}")
    log_step(f"output: {args.output_questions_json}")
    log_step(f"video_id: {args.video_id}")
    log_step(f"model: {args.model}")
    log_step(f"question_types: {args.question_types}")
    log_step(f"max_questions: {args.max_questions}")
    log_step(f"batch_size: {args.batch_size}")

    final_output = run_generation(
        mapped_json=args.mapped_json,
        output_json=args.output_questions_json,
        video_id=args.video_id,
        allowed_question_types=args.question_types,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
        llm=llm,
        model_name=args.model,
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
