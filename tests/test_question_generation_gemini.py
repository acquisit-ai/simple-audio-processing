from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module():
    module_name = "question_generation_gemini"
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / "6question-generation-gemini.py")
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
                "text": "Michael helps workers and sacred work.",
                "translation": "迈克尔帮助员工，也做重要的工作。",
                "start": 1000,
                "end": 5000,
                "tokens": [
                    {
                        "index": 0,
                        "text": "Michael",
                        "explanation": "人名。",
                        "start": 1000,
                        "end": 1200,
                        "semantic_element": {
                            "coarse_id": 1,
                            "base_form": "Michael",
                            "translation": "迈克尔",
                            "dictionary": "人名。",
                            "kind": "word",
                            "pos": "noun",
                        },
                    },
                    {
                        "index": 1,
                        "text": "Workers!",
                        "explanation": "指员工。",
                        "start": 1300,
                        "end": 1500,
                        "semantic_element": {
                            "coarse_id": 2,
                            "base_form": "worker",
                            "translation": "员工",
                            "dictionary": "员工。",
                            "kind": "word",
                            "pos": "noun",
                        },
                    },
                    {
                        "index": 2,
                        "text": "sacred",
                        "explanation": "表示重要。",
                        "start": 1600,
                        "end": 1800,
                        "semantic_element": {
                            "coarse_id": 3,
                            "base_form": "sacred",
                            "translation": "神圣/重要",
                            "dictionary": "神圣或非常重要。",
                            "kind": "word",
                            "pos": "adjective",
                        },
                    },
                ],
            }
        ]
    }


def select_refs(question_gen):
    occurrences, _ = question_gen.extract_question_occurrences(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
    )
    groups = question_gen.build_context_selection_groups(occurrences, top_k=1)
    selection_output = question_gen.AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": group.group_id,
                    "sentence_candidate_id": group.sentence_candidates[0].sentence_candidate_id,
                    "scores": {
                        "visual_context": 8 if group.coarse_unit_id == 2 else 1,
                        "context_clarity": 8 if group.coarse_unit_id == 2 else 2,
                        "learning_value": 8 if group.coarse_unit_id == 2 else 2,
                        "question_fit": 8 if group.coarse_unit_id == 2 else 2,
                    },
                    "reason": "测试选择。",
                }
                for group in groups
            ]
        }
    )
    return question_gen.apply_context_selections(groups, selection_output)


def test_selected_refs_keep_all_coarse_units_with_scores_and_target_text():
    question_gen = load_module()

    refs = select_refs(question_gen)
    checkpoint = question_gen.build_selected_ref_checkpoint(
        refs=refs,
        selection_model_name="fake-selection",
        selection_top_k=1,
        allowed_question_types=["context_meaning_choice"],
        candidate_score_threshold=6.5,
    )
    selected_refs = question_gen.build_selected_coarse_unit_refs(
        checkpoint,
        question_reject_reasons={},
    )

    assert {ref.coarse_unit_id for ref in selected_refs.refs} == {1, 2, 3}
    workers = next(ref for ref in selected_refs.refs if ref.coarse_unit_id == 2)
    assert workers.target_text == "workers"
    assert workers.scores.visual_context == 8
    assert workers.candidate_score == 8.0
    assert workers.question_reject_reason is None


def test_candidate_filter_only_sets_reject_reason_without_removing_refs():
    question_gen = load_module()
    refs = select_refs(question_gen)

    candidates, reject_reasons = question_gen.filter_question_generation_candidates(
        refs,
        candidate_score_threshold=6.5,
    )

    assert [candidate.target_text for candidate in candidates] == ["workers"]
    refs_by_id = {ref.candidate_id: ref for ref in refs}
    assert {
        refs_by_id[candidate_id].target_text: reason
        for candidate_id, reason in reject_reasons.items()
    } == {
        "sacred": "candidate_score 低于阈值",
        "Michael": "专有名词",
    }


def test_question_generation_rejection_is_written_to_ref_reason(tmp_path: Path):
    question_gen = load_module()

    question_gen.validate_video_gcs_uri = lambda video_gcs_uri: None

    class FakeCaches:
        def create(self, model, config):
            class Cache:
                name = "projects/test/locations/global/cachedContents/test-cache"
                usage_metadata = None

            return Cache()

        def delete(self, name):
            return None

    class FakeGeminiClient:
        caches = FakeCaches()

    class FakeSelectionLLM:
        def invoke_context_selection_batch(self, messages, cached_content_name=None):
            payload = json.loads(messages[-1]["content"].split("CURRENT_CONTEXT_SELECTION_GROUPS:\n", 1)[1])
            return question_gen.AIContextSelectionBatchOutput.model_validate(
                {
                    "selections": [
                        {
                            "group_id": group["group_id"],
                            "sentence_candidate_id": group["sentence_candidates"][0]["sentence_candidate_id"],
                            "scores": {
                                "visual_context": 8 if group["coarse_unit_id"] == 2 else 1,
                                "context_clarity": 8 if group["coarse_unit_id"] == 2 else 2,
                                "learning_value": 8 if group["coarse_unit_id"] == 2 else 2,
                                "question_fit": 8 if group["coarse_unit_id"] == 2 else 2,
                            },
                            "reason": "测试选择。",
                        }
                        for group in payload["groups"]
                    ]
                }
            )

    class FakeQuestionLLM:
        def invoke_question_batch(self, messages):
            payload = json.loads(messages[-1]["content"].split("CURRENT_CANDIDATE_BATCH:\n", 1)[1])
            candidate_id = payload["candidates"][0]["candidate_id"]
            return question_gen.AIQuestionBatchOutput.model_validate(
                {
                    "results": [],
                    "rejections": [
                        {
                            "candidate_id": candidate_id,
                            "reason": "上下文不足。",
                        }
                    ],
                }
            )

    input_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    input_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    final_output = question_gen.run_generation(
        mapped_json=input_path,
        output_json=output_path,
        allowed_question_types=["context_meaning_choice"],
        batch_size=10,
        llm=FakeQuestionLLM(),
        model_name="fake-question",
        gemini_client=FakeGeminiClient(),
        selection_llm=FakeSelectionLLM(),
        selection_model_name="fake-selection",
        selection_top_k=1,
        selection_batch_size=10,
        selection_max_workers=1,
        candidate_score_threshold=6.5,
        video_gcs_uri="gs://test-bucket/test-video.mp4",
    )

    reason_by_target = {
        ref.target_text: ref.question_reject_reason
        for ref in final_output.selected_coarse_unit_refs.refs
    }
    assert reason_by_target["workers"] == "question generation 阶段主动拒绝：上下文不足。"
    assert reason_by_target["sacred"] == "candidate_score 低于阈值"
    assert reason_by_target["Michael"] == "专有名词"
