from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module():
    module_name = "question_generation_deepseek"
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / "9question-generation-deepseek.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mapped_payload():
    return {
        "sentences": [
            {
                "index": 0,
                "text": "The most sacred thing I do is care and provide for my workers.",
                "translation": "我做的最神圣的事情，就是关心并供养我的员工。",
                "start": 1000,
                "end": 5000,
                "tokens": [
                    {
                        "index": 0,
                        "text": "The",
                        "explanation": "冠词。",
                        "start": 1000,
                        "end": 1100,
                        "semantic_element": {
                            "coarse_id": None,
                            "base_form": "the",
                            "translation": "这个",
                            "dictionary": "定冠词。",
                            "reason": "基础功能词。",
                        },
                    },
                    {
                        "index": 1,
                        "text": "sacred",
                        "explanation": "在这里是比喻用法，表示非常重要、不可轻视。",
                        "start": 1200,
                        "end": 1500,
                        "semantic_element": {
                            "coarse_id": 138446,
                            "base_form": "sacred",
                            "translation": "神圣（宗教或比喻）",
                            "dictionary": "表示被视为神圣或应受特殊、不可侵犯的尊重。",
                            "kind": "word",
                            "pos": "adjective",
                            "reason": "语义可靠匹配。",
                        },
                    },
                    {
                        "index": 2,
                        "text": "provide for",
                        "explanation": "表示供养、养活某人。",
                        "start": 3000,
                        "end": 3800,
                        "semantic_element": {
                            "coarse_id": 130328,
                            "base_form": "provide",
                            "translation": "供养/养活",
                            "dictionary": "为某人提供生活所需或经济上的供养。",
                            "kind": "phrase",
                            "pos": "verb",
                            "reason": "语义可靠匹配。",
                        },
                    },
                ],
            },
            {
                "index": 1,
                "text": "Numbers 123.",
                "translation": "数字 123。",
                "start": 6000,
                "end": 7000,
                "tokens": [
                    {
                        "index": 0,
                        "text": "123",
                        "explanation": "数字。",
                        "start": 6100,
                        "end": 6200,
                        "semantic_element": {
                            "coarse_id": 999,
                            "base_form": "123",
                            "translation": "一二三",
                            "dictionary": "数字。",
                            "kind": "word",
                            "pos": "noun",
                            "reason": "测试用。",
                        },
                    }
                ],
            },
        ]
    }


def test_hard_filter_keeps_valid_mapped_candidates_and_rejects_invalid_tokens():
    question_gen = load_module()

    candidates, rejects = question_gen.extract_question_candidates(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        max_candidates=10,
    )

    assert [candidate.target_text for candidate in candidates] == ["provide for", "sacred"]
    assert {reject.reason for reject in rejects} >= {
        "coarse_id is null",
        "target text is numeric",
    }


def test_hard_filter_rejects_missing_timing_and_duplicate_coarse_units():
    question_gen = load_module()
    payload = mapped_payload()
    duplicate = json.loads(json.dumps(payload["sentences"][0]["tokens"][1], ensure_ascii=False))
    duplicate["index"] = 3
    duplicate["text"] = "sacred."
    payload["sentences"][0]["tokens"].append(duplicate)
    del payload["sentences"][0]["tokens"][2]["start"]

    candidates, rejects = question_gen.extract_question_candidates(
        payload,
        allowed_question_types=["context_meaning_choice"],
        max_candidates=10,
    )

    assert [candidate.target_text for candidate in candidates] == ["sacred"]
    assert {reject.reason for reject in rejects} >= {
        "token timing is missing",
        "duplicate candidate key",
    }


def test_ai_output_schema_rejects_metadata_fields():
    question_gen = load_module()

    with pytest.raises(ValidationError):
        question_gen.AIQuestionBatchOutput.model_validate(
            {
                "results": [
                    {
                        "candidate_id": "c_000001",
                        "question_type": "context_meaning_choice",
                        "coarse_unit_id": 138446,
                        "video_id": "00000000-0000-0000-0000-000000000001",
                        "context_start_ms": 1000,
                        "content_payload": {
                            "question": "这里的 sacred 是什么意思？",
                            "context_text": "The most sacred thing I do is care.",
                            "options": [
                                {"id": "correct", "text": "神圣、非常重要"},
                                {"id": "wrong_1", "text": "普通、随便"},
                                {"id": "wrong_2", "text": "昂贵、奢侈"},
                                {"id": "wrong_3", "text": "快速、临时"},
                            ],
                            "explanation": "sacred 在这里是比喻用法。",
                        },
                    }
                ],
                "rejections": [],
            }
        )


def test_system_prompt_limits_explanation_to_correct_option():
    question_gen = load_module()

    assert "explanation 只说明正确选项为什么对" in question_gen.QUESTION_SYSTEM_PROMPT
    assert "不用说明错误选项为什么错" in question_gen.QUESTION_SYSTEM_PROMPT


def test_system_prompt_describes_output_json_schema_without_input_schema():
    question_gen = load_module()

    assert "OUTPUT JSON SCHEMA" in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"results"' in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"rejections"' in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"content_payload"' in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"options"' in question_gen.QUESTION_SYSTEM_PROMPT
    assert "不要输出 coarse_id、video_id、sentence_index、token_index、start/end、status" in (
        question_gen.QUESTION_SYSTEM_PROMPT
    )
    assert "INPUT JSON SCHEMA" not in question_gen.QUESTION_SYSTEM_PROMPT


def test_merge_fills_script_owned_catalog_question_fields():
    question_gen = load_module()
    candidates, _ = question_gen.extract_question_candidates(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice"],
        max_candidates=10,
    )
    candidate = next(candidate for candidate in candidates if candidate.target_text == "sacred")
    ai_result = question_gen.AIQuestionResult.model_validate(
        {
            "candidate_id": candidate.candidate_id,
            "question_type": "context_meaning_choice",
            "content_payload": {
                "question": "这里的 “sacred” 最接近什么意思？",
                "context_text": candidate.sentence_text,
                "options": [
                    {"id": "correct", "text": "神圣、非常重要"},
                    {"id": "wrong_1", "text": "普通、随便"},
                    {"id": "wrong_2", "text": "昂贵、奢侈"},
                    {"id": "wrong_3", "text": "快速、临时"},
                ],
                "explanation": "sacred 在这里是比喻用法。",
            },
        }
    )

    question = question_gen.merge_ai_result(
        candidate,
        ai_result,
        video_id="00000000-0000-0000-0000-000000000001",
    )

    assert question.model_dump() == {
        "scope_type": "video_unit",
        "question_type": "context_meaning_choice",
        "coarse_unit_id": 138446,
        "target_text": "sacred",
        "video_id": "00000000-0000-0000-0000-000000000001",
        "context_sentence_index": 0,
        "context_span_index": 1,
        "context_start_ms": 1000,
        "context_end_ms": 5000,
        "content_payload": ai_result.content_payload.model_dump(),
        "status": "draft",
    }


def test_cloze_validation_rejects_context_that_leaks_target_text():
    question_gen = load_module()
    payload = question_gen.AIContentPayload.model_validate(
        {
            "question": "根据上下文，空格处最合适的是哪一个？",
            "context_text": "The most sacred thing I do is care.",
            "options": [
                {"id": "correct", "text": "sacred"},
                {"id": "wrong_1", "text": "ordinary"},
                {"id": "wrong_2", "text": "expensive"},
                {"id": "wrong_3", "text": "temporary"},
            ],
            "explanation": "sacred 表示神圣、非常重要。",
        }
    )

    with pytest.raises(ValueError, match="cloze context_text leaks target text"):
        question_gen.validate_content_payload(
            payload,
            question_type="context_cloze_choice",
            target_text="sacred",
        )


def test_option_validation_rejects_duplicate_text_and_missing_correct():
    question_gen = load_module()
    duplicate_payload = question_gen.AIContentPayload.model_validate(
        {
            "question": "这里的 sacred 是什么意思？",
            "context_text": "The most sacred thing I do is care.",
            "options": [
                {"id": "correct", "text": "神圣"},
                {"id": "wrong_1", "text": "神圣"},
                {"id": "wrong_2", "text": "昂贵"},
                {"id": "wrong_3", "text": "快速"},
            ],
            "explanation": "sacred 表示神圣。",
        }
    )
    missing_correct_payload = question_gen.AIContentPayload.model_validate(
        {
            "question": "这里的 sacred 是什么意思？",
            "context_text": "The most sacred thing I do is care.",
            "options": [
                {"id": "wrong_0", "text": "神圣"},
                {"id": "wrong_1", "text": "普通"},
                {"id": "wrong_2", "text": "昂贵"},
                {"id": "wrong_3", "text": "快速"},
            ],
            "explanation": "sacred 表示神圣。",
        }
    )

    with pytest.raises(ValueError, match="option texts must be unique"):
        question_gen.validate_content_payload(
            duplicate_payload,
            question_type="context_meaning_choice",
            target_text="sacred",
        )
    with pytest.raises(ValueError, match="options must use ids"):
        question_gen.validate_content_payload(
            missing_correct_payload,
            question_type="context_meaning_choice",
            target_text="sacred",
        )


def test_run_generation_with_fake_llm_writes_wrapper_and_audit(tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    class FakeLLM:
        model_name = "fake-deepseek"

        def invoke_json(self, messages):
            assert "context_start_ms" not in messages[-1]["content"]
            return json.dumps(
                {
                    "results": [
                        {
                            "candidate_id": "c_000001",
                            "question_type": "context_cloze_choice",
                            "content_payload": {
                                "question": "根据上下文，空格处最合适的是哪一个？",
                                "context_text": "The most sacred thing I do is care and ____ my workers.",
                                "options": [
                                    {"id": "correct", "text": "provide for"},
                                    {"id": "wrong_1", "text": "take off"},
                                    {"id": "wrong_2", "text": "work out"},
                                    {"id": "wrong_3", "text": "look up"},
                                ],
                                "explanation": "provide for someone 表示供养、养活某人。",
                            },
                        }
                    ],
                    "rejections": [
                        {
                            "candidate_id": "c_000002",
                            "reason": "示例中只生成一道题。",
                        }
                    ],
                },
                ensure_ascii=False,
            )

    question_gen.run_generation(
        mapped_json=mapped_path,
        output_json=output_path,
        video_id="00000000-0000-0000-0000-000000000001",
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        max_questions=1,
        batch_size=10,
        llm=FakeLLM(),
        model_name="fake-deepseek",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"]["mapped_json"] == str(mapped_path)
    assert payload["source"]["video_id"] == "00000000-0000-0000-0000-000000000001"
    assert payload["source"]["model"] == "fake-deepseek"
    assert payload["audit"] == {
        "candidate_count": 2,
        "generated_count": 1,
        "rejected_count": 1,
    }
    assert payload["questions"][0]["scope_type"] == "video_unit"
    assert payload["questions"][0]["coarse_unit_id"] == 130328
    assert payload["questions"][0]["context_start_ms"] == 1000
    assert payload["questions"][0]["context_end_ms"] == 5000
    assert (tmp_path / "log" / "mapped.json.question_audit.jsonl").exists()
