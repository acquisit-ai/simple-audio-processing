from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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


def select_first_sentence_candidates(question_gen, payload=None):
    occurrences, _ = question_gen.extract_question_occurrences(
        payload or mapped_payload(),
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
    )
    groups = question_gen.build_context_selection_groups(
        occurrences,
        top_k=question_gen.DEFAULT_SELECTION_TOP_K,
    )
    selection_output = question_gen.AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": group.group_id,
                    "sentence_candidate_id": group.sentence_candidates[0].sentence_candidate_id,
                    "reason": "测试中选择每组第一句。",
                }
                for group in groups
            ]
        }
    )
    return question_gen.apply_context_selections(groups, selection_output)


def test_hard_filter_keeps_valid_mapped_candidates_and_rejects_invalid_tokens():
    question_gen = load_module()

    occurrences, rejects = question_gen.extract_question_occurrences(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
    )

    assert [occurrence.target_text for occurrence in occurrences] == ["provide for", "sacred"]
    assert {reject.reason for reject in rejects} >= {
        "coarse_id is null",
        "target text is numeric",
    }


def test_candidate_ai_payload_excludes_translation_and_coarse_definition():
    question_gen = load_module()
    candidates = select_first_sentence_candidates(question_gen)
    sacred = next(candidate for candidate in candidates if candidate.target_text == "sacred")
    ai_payload = sacred.to_ai_payload()

    assert "sentence_text" in ai_payload
    assert "sentence_translation" not in ai_payload
    assert "coarse_definition" not in ai_payload


def test_build_ai_messages_uses_full_transcript_prefix_without_extra_candidate_fields():
    question_gen = load_module()
    candidates = select_first_sentence_candidates(question_gen)

    messages = question_gen.build_ai_messages(
        candidates[:1],
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        full_transcript_text="The most sacred thing I do is care and provide for my workers.\nNumbers 123.",
    )

    user_content = messages[-1]["content"]
    assert user_content.startswith("FULL_VIDEO_TRANSCRIPT:")
    assert "The most sacred thing I do is care and provide for my workers.\nNumbers 123." in user_content
    assert "CURRENT_CANDIDATE_BATCH:" in user_content
    assert '"sentence_translation"' not in user_content
    assert '"coarse_definition"' not in user_content


def test_count_candidate_batches_rounds_up_and_handles_empty():
    question_gen = load_module()

    assert question_gen.count_candidate_batches(0, 10) == 0
    assert question_gen.count_candidate_batches(1, 10) == 1
    assert question_gen.count_candidate_batches(10, 10) == 1
    assert question_gen.count_candidate_batches(11, 10) == 2


def test_hard_filter_rejects_missing_timing_and_duplicate_coarse_units():
    question_gen = load_module()
    payload = mapped_payload()
    duplicate = json.loads(json.dumps(payload["sentences"][0]["tokens"][1], ensure_ascii=False))
    duplicate["index"] = 3
    duplicate["text"] = "holy"
    duplicate["semantic_element"]["base_form"] = "holy"
    payload["sentences"][0]["tokens"].append(duplicate)
    del payload["sentences"][0]["tokens"][2]["start"]

    occurrences, rejects = question_gen.extract_question_occurrences(
        payload,
        allowed_question_types=["context_meaning_choice"],
    )

    assert [occurrence.target_text for occurrence in occurrences] == ["sacred", "holy"]
    reject_reasons = {reject.reason for reject in rejects}
    assert reject_reasons >= {"token timing is missing"}
    assert "duplicate coarse unit" not in reject_reasons
    assert "duplicate candidate key" not in reject_reasons
    assert "coarse unit candidate limit reached" not in reject_reasons
    assert "sentence candidate limit reached" not in reject_reasons
    assert "max candidate limit reached" not in reject_reasons


def test_hard_filter_allows_multiple_distinct_coarse_units_in_one_sentence():
    question_gen = load_module()
    payload = mapped_payload()
    extra = json.loads(json.dumps(payload["sentences"][0]["tokens"][1], ensure_ascii=False))
    extra["index"] = 3
    extra["text"] = "restore"
    extra["semantic_element"]["coarse_id"] = 222222
    extra["semantic_element"]["base_form"] = "restore"
    extra["semantic_element"]["translation"] = "恢复"
    extra["semantic_element"]["dictionary"] = "使某物恢复到原来的状态。"
    extra["semantic_element"]["pos"] = "verb"
    payload["sentences"][0]["tokens"].append(extra)

    occurrences, rejects = question_gen.extract_question_occurrences(
        payload,
        allowed_question_types=["context_meaning_choice"],
    )

    assert {occurrence.coarse_unit_id for occurrence in occurrences} == {130328, 138446, 222222}
    assert "sentence candidate limit reached" not in {reject.reason for reject in rejects}


def test_context_selection_groups_include_all_unique_coarse_units():
    question_gen = load_module()

    occurrences, rejects = question_gen.extract_question_occurrences(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice"],
    )
    groups = question_gen.build_context_selection_groups(occurrences, top_k=5)

    assert {group.coarse_unit_id for group in groups} == {130328, 138446}
    assert "max candidate limit reached" not in {reject.reason for reject in rejects}


def test_context_selection_groups_merge_same_coarse_sentence_and_keep_top_k():
    question_gen = load_module()
    payload = mapped_payload()
    same_sentence_late = json.loads(json.dumps(payload["sentences"][0]["tokens"][1], ensure_ascii=False))
    same_sentence_late["index"] = 4
    same_sentence_late["text"] = "holy"
    same_sentence_late["semantic_element"]["base_form"] = "holy"
    other_sentence = json.loads(json.dumps(payload["sentences"][0], ensure_ascii=False))
    other_sentence["index"] = 2
    other_sentence["text"] = "This work is sacred."
    other_sentence["start"] = 8000
    other_sentence["end"] = 9000
    other_sentence["tokens"] = [json.loads(json.dumps(payload["sentences"][0]["tokens"][1], ensure_ascii=False))]
    other_sentence["tokens"][0]["index"] = 0
    payload["sentences"][0]["tokens"].append(same_sentence_late)
    payload["sentences"].append(other_sentence)

    occurrences, _ = question_gen.extract_question_occurrences(
        payload,
        allowed_question_types=["context_meaning_choice"],
    )
    groups = question_gen.build_context_selection_groups(occurrences, top_k=1)
    sacred_group = next(group for group in groups if group.coarse_unit_id == 138446)

    assert len(sacred_group.sentence_candidates) == 1
    assert sacred_group.sentence_candidates[0].token_index == 1


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


def test_tool_schema_limits_explanation_to_correct_option():
    question_gen = load_module()
    explanation_description = (
        question_gen.QUESTION_OUTPUT_TOOL["function"]["parameters"]["properties"]["results"]["items"]["properties"][
            "content_payload"
        ]["properties"]["explanation"]["description"]
    )

    assert "必须使用中文" in explanation_description
    assert "只说明正确选项为什么正确，不用说明错误选项为什么错" in explanation_description
    assert "explanation 只说明正确选项为什么对" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "不用说明错误选项为什么错" not in question_gen.QUESTION_SYSTEM_PROMPT


def test_system_prompt_sets_chinese_reader_language_rules():
    question_gen = load_module()

    assert "面向中文读者" in question_gen.QUESTION_SYSTEM_PROMPT
    assert "context_meaning_choice 的 options.text 必须使用中文释义" in question_gen.QUESTION_SYSTEM_PROMPT
    assert "context_cloze_choice 的 options.text 应使用英文单词或短语" in question_gen.QUESTION_SYSTEM_PROMPT


def test_system_prompt_uses_tool_call_without_embedded_json_schema():
    question_gen = load_module()

    assert "必须调用 `submit_question_batch` 工具提交结果" in question_gen.QUESTION_SYSTEM_PROMPT
    assert "不要输出 coarse_id、sentence_index、token_index、start/end、status" not in (
        question_gen.QUESTION_SYSTEM_PROMPT
    )
    assert "工具参数结构由 submit_question_batch 的 function schema 约束" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "顶层必须只有 results 和 rejections 两个字段" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "result 必须只有 candidate_id、question_type、content_payload 三个字段" not in (
        question_gen.QUESTION_SYSTEM_PROMPT
    )
    assert "content_payload 必须只有 question、context_text、options、explanation 四个字段" not in (
        question_gen.QUESTION_SYSTEM_PROMPT
    )
    assert "options 必须正好 4 个" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert 'id 必须是 "correct"' not in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"candidate_id"' not in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"content_payload"' not in question_gen.QUESTION_SYSTEM_PROMPT
    assert '"options"' not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "OUTPUT RULES" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "QUESTION TYPE RULES" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "错误选项必须有迷惑性" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "context_meaning_choice：给出上下文" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "context_cloze_choice：把上下文里的目标词隐藏为 ____" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "TOOL ARGUMENTS JSON SCHEMA" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "OUTPUT JSON SCHEMA" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "只输出合法 JSON object" not in question_gen.QUESTION_SYSTEM_PROMPT
    assert "INPUT JSON SCHEMA" not in question_gen.QUESTION_SYSTEM_PROMPT


def test_system_prompt_content_examples_include_positive_and_negative_examples():
    question_gen = load_module()
    prompt = question_gen.QUESTION_SYSTEM_PROMPT

    assert "正面例子" in prompt
    assert "反面例子" in prompt
    assert "错误项为什么不好" in prompt
    assert "普通、随便" in prompt
    assert "重要" in prompt
    assert "look after" in prompt
    assert "语境和语法上也可能成立" in prompt
    assert "不是合格错误项" in prompt


def test_tool_schema_describes_question_batch_arguments():
    question_gen = load_module()

    tool = question_gen.QUESTION_OUTPUT_TOOL
    function = tool["function"]
    parameters = function["parameters"]
    result_item = parameters["properties"]["results"]["items"]
    content_payload = result_item["properties"]["content_payload"]

    assert tool["type"] == "function"
    assert function["name"] == "submit_question_batch"
    assert function["strict"] is True
    assert function["description"].startswith(
        "提交视频上下文题目内容和候选拒绝原因。只包含 AI 负责生成的题目内容，不包含数据库元数据。"
    )
    assert parameters["properties"]["results"]["description"].startswith("适合出题的 candidate 对应的题目内容")
    assert parameters["properties"]["rejections"]["description"].startswith("不适合生成高质量题目的 candidate")
    assert result_item["properties"]["candidate_id"]["description"] == "当前输入 batch 中的 candidate_id。"
    assert "不要为同一个 candidate 同时提交 result 和 rejection" in function["description"]
    assert "不要包含 coarse_id、sentence_index、token_index、start/end、status" in (
        function["description"]
    )
    assert "题目面向中文读者" not in function["description"]
    assert "context_meaning_choice 的 options.text" not in function["description"]
    assert "context_cloze_choice 的 options.text" not in function["description"]
    assert "给出上下文，询问目标词在当前语境中的意思" in (
        result_item["properties"]["question_type"]["description"]
    )
    assert "把上下文里的目标词隐藏为 ____" in result_item["properties"]["question_type"]["description"]
    assert "错误选项必须有迷惑性" in content_payload["properties"]["options"]["description"]
    assert "不能在当前语境和语法上也成立" in content_payload["properties"]["options"]["description"]
    assert "cloze 题的上下文不能泄露答案" in content_payload["properties"]["context_text"]["description"]
    assert "必须使用中文" in content_payload["properties"]["question"]["description"]
    assert "context_meaning_choice 的 options.text 必须使用中文释义" in (
        content_payload["properties"]["options"]["description"]
    )
    assert "context_cloze_choice 的 options.text 应使用英文单词或短语" in (
        content_payload["properties"]["options"]["description"]
    )
    assert "必须使用中文" in content_payload["properties"]["explanation"]["description"]
    assert "只说明正确选项为什么正确，不用说明错误选项为什么错" in (
        result_item["properties"]["content_payload"]["properties"]["explanation"]["description"]
    )
    assert "上下文不足、答案太显然、干扰项难以构造" in (
        parameters["properties"]["rejections"]["items"]["properties"]["reason"]["description"]
    )
    assert parameters["required"] == ["results", "rejections"]
    assert parameters["additionalProperties"] is False
    assert result_item["additionalProperties"] is False
    assert "content_payload" in result_item["properties"]
    assert "options" in content_payload["properties"]
    assert content_payload["additionalProperties"] is False


def test_selection_tool_schema_is_strict_and_has_no_rejections():
    question_gen = load_module()

    tool = question_gen.SELECTION_OUTPUT_TOOL
    function = tool["function"]
    parameters = function["parameters"]
    selection_item = parameters["properties"]["selections"]["items"]

    assert tool["type"] == "function"
    assert function["name"] == "submit_context_selection_batch"
    assert function["strict"] is True
    assert "rejections" not in parameters["properties"]
    assert parameters["required"] == ["selections"]
    assert parameters["additionalProperties"] is False
    assert selection_item["required"] == ["group_id", "sentence_candidate_id", "reason"]
    assert selection_item["additionalProperties"] is False


def test_context_selection_prompt_has_concrete_selection_criteria():
    question_gen = load_module()
    occurrences, _ = question_gen.extract_question_occurrences(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
    )
    groups = question_gen.build_context_selection_groups(occurrences, top_k=5)

    messages = question_gen.build_context_selection_messages(groups, "full transcript")
    prompt = messages[0]["content"]

    assert "目标词或短语在句子里的含义清楚" in prompt
    assert "能自然生成中文释义选择题或英文 cloze 选择题" in prompt
    assert "不要优先选择纯寒暄、残句、引用不完整" in prompt
    assert "不要拒绝任何 group" not in prompt


def test_selection_defaults_use_small_batches_and_four_workers():
    question_gen = load_module()

    assert question_gen.DEFAULT_SELECTION_REASONING_EFFORT == "high"
    assert question_gen.DEFAULT_SELECTION_THINKING == {"type": "enabled"}
    assert question_gen.DEFAULT_SELECTION_BATCH_SIZE == 10
    assert question_gen.DEFAULT_SELECTION_MAX_WORKERS == 4


def test_deepseek_context_selection_llm_sends_high_reasoning_effort():
    question_gen = load_module()
    payload = {
        "selections": [
            {
                "group_id": "g_000001",
                "sentence_candidate_id": "s_000001",
                "reason": "上下文清楚。",
            }
        ]
    }

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name=question_gen.SELECTION_TOOL_NAME,
                                        arguments=json.dumps(payload, ensure_ascii=False),
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = question_gen.DeepSeekContextSelectionLLM(client, "fake-selection-model")

    result = llm.invoke_context_selection_batch([{"role": "user", "content": "hi"}])

    assert result.selections[0].group_id == "g_000001"
    assert completions.kwargs["tools"] == [question_gen.SELECTION_OUTPUT_TOOL]
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["extra_body"] == {
        "thinking": question_gen.DEFAULT_SELECTION_THINKING,
    }


def test_deepseek_context_selection_llm_retries_malformed_tool_arguments():
    question_gen = load_module()
    payload = {
        "selections": [
            {
                "group_id": "g_000001",
                "sentence_candidate_id": "s_000001",
                "reason": "上下文清楚。",
            }
        ]
    }

    class FakeCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            arguments = '{"selections": [{"group_id": "g_000001" "sentence_candidate_id": "s_000001"}]}'
            if self.call_count == 2:
                arguments = json.dumps(payload, ensure_ascii=False)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name=question_gen.SELECTION_TOOL_NAME,
                                        arguments=arguments,
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = question_gen.DeepSeekContextSelectionLLM(client, "fake-selection-model")

    result = llm.invoke_context_selection_batch([{"role": "user", "content": "hi"}])

    assert completions.call_count == 2
    assert result.selections[0].group_id == "g_000001"


def test_context_selection_llm_batches_run_concurrently():
    question_gen = load_module()

    def sentence_candidate(group_no: int, candidate_no: int):
        return question_gen.SentenceCandidate(
            sentence_candidate_id=f"s_{group_no:06d}_{candidate_no}",
            target_text=f"target {group_no}",
            base_form=f"target {group_no}",
            coarse_unit_id=group_no,
            coarse_label=f"释义 {group_no}",
            kind="word",
            pos="noun",
            sentence_index=group_no * 10 + candidate_no,
            sentence_text=f"Sentence {group_no} candidate {candidate_no}.",
            sentence_start_ms=group_no * 1000,
            sentence_end_ms=group_no * 1000 + 500,
            token_index=candidate_no,
            token_explanation="测试解释。",
            score=100 - candidate_no,
        )

    groups = [
        question_gen.ContextSelectionGroup(
            group_id=f"g_{group_no:06d}",
            coarse_unit_id=group_no,
            coarse_label=f"释义 {group_no}",
            kind="word",
            pos="noun",
            sentence_candidates=[sentence_candidate(group_no, 1), sentence_candidate(group_no, 2)],
        )
        for group_no in range(1, 5)
    ]

    class FakeSelectionLLM:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.call_count = 0
            self.lock = threading.Lock()

        def invoke_context_selection_batch(self, messages):
            payload = json.loads(messages[-1]["content"].split("CURRENT_CONTEXT_SELECTION_GROUPS:\n", 1)[1])
            with self.lock:
                self.active += 1
                self.call_count += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1
            return question_gen.AIContextSelectionBatchOutput.model_validate(
                {
                    "selections": [
                        {
                            "group_id": group["group_id"],
                            "sentence_candidate_id": group["sentence_candidates"][0]["sentence_candidate_id"],
                            "reason": "测试选择。",
                        }
                        for group in payload["groups"]
                    ]
                }
            )

    selection_llm = FakeSelectionLLM()
    output = question_gen.select_context_groups_with_llm(
        groups,
        selection_llm=selection_llm,
        full_transcript_text="full transcript",
        selection_batch_size=1,
        selection_max_workers=4,
    )

    assert [selection.group_id for selection in output.selections] == [group.group_id for group in groups]
    assert selection_llm.call_count == 4
    assert selection_llm.max_active > 1


def test_deepseek_tool_call_llm_sends_tools_without_tool_choice():
    question_gen = load_module()
    payload = {
        "results": [
            {
                "candidate_id": "c_000001",
                "question_type": "context_meaning_choice",
                "content_payload": {
                    "question": "这里的 sacred 是什么意思？",
                    "context_text": "The most sacred thing I do is care.",
                    "options": [
                        {"id": "correct", "text": "神圣、非常重要"},
                        {"id": "wrong_1", "text": "普通、随便"},
                        {"id": "wrong_2", "text": "昂贵、奢侈"},
                        {"id": "wrong_3", "text": "快速、临时"},
                    ],
                    "explanation": "sacred 在这里表示非常重要。",
                },
            }
        ],
        "rejections": [],
    }

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name=question_gen.QUESTION_TOOL_NAME,
                                        arguments=json.dumps(payload, ensure_ascii=False),
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = question_gen.DeepSeekToolCallLLM(client, "fake-model")

    result = llm.invoke_question_batch([{"role": "user", "content": "hi"}])

    assert result.results[0].candidate_id == "c_000001"
    assert "response_format" not in completions.kwargs
    assert completions.kwargs["tools"] == [question_gen.QUESTION_OUTPUT_TOOL]
    assert "tool_choice" not in completions.kwargs
    assert completions.kwargs["reasoning_effort"] == question_gen.DEFAULT_QUESTION_REASONING_EFFORT
    assert completions.kwargs["extra_body"] == {
        "thinking": question_gen.DEFAULT_QUESTION_THINKING,
    }


def test_deepseek_tool_call_llm_retries_malformed_tool_arguments():
    question_gen = load_module()
    payload = {
        "results": [],
        "rejections": [
            {
                "candidate_id": "c_000001",
                "reason": "上下文不足。",
            }
        ],
    }

    class FakeCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            arguments = '{"results": [], "rejections": [{"candidate_id": "c_000001" "reason": "bad"}]}'
            if self.call_count == 2:
                arguments = json.dumps(payload, ensure_ascii=False)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name=question_gen.QUESTION_TOOL_NAME,
                                        arguments=arguments,
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = question_gen.DeepSeekToolCallLLM(client, "fake-model")

    result = llm.invoke_question_batch([{"role": "user", "content": "hi"}])

    assert completions.call_count == 2
    assert result.rejections[0].candidate_id == "c_000001"


def test_validate_context_selection_requires_one_selection_per_group():
    question_gen = load_module()
    candidates = select_first_sentence_candidates(question_gen)
    groups = question_gen.build_context_selection_groups(
        question_gen.extract_question_occurrences(
            mapped_payload(),
            allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        )[0],
        top_k=5,
    )
    valid_output = question_gen.AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": group.group_id,
                    "sentence_candidate_id": group.sentence_candidates[0].sentence_candidate_id,
                    "reason": "测试选择。",
                }
                for group in groups
            ]
        }
    )

    assert [candidate.coarse_unit_id for candidate in candidates] == [
        candidate.coarse_unit_id for candidate in question_gen.apply_context_selections(groups, valid_output)
    ]

    missing_output = question_gen.AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": groups[0].group_id,
                    "sentence_candidate_id": groups[0].sentence_candidates[0].sentence_candidate_id,
                    "reason": "只选一个 group。",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="missing selections"):
        question_gen.apply_context_selections(groups, missing_output)

    duplicate_output = question_gen.AIContextSelectionBatchOutput.model_validate(
        {
            "selections": [
                {
                    "group_id": groups[0].group_id,
                    "sentence_candidate_id": groups[0].sentence_candidates[0].sentence_candidate_id,
                    "reason": "第一次。",
                },
                {
                    "group_id": groups[0].group_id,
                    "sentence_candidate_id": groups[0].sentence_candidates[0].sentence_candidate_id,
                    "reason": "重复。",
                },
            ]
        }
    )
    with pytest.raises(ValueError, match="duplicate selection"):
        question_gen.apply_context_selections(groups, duplicate_output)


def test_parse_llm_context_selection_response_rejects_wrong_ids():
    question_gen = load_module()
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=question_gen.SELECTION_TOOL_NAME,
                    arguments='{"selections": [{"group_id": "unknown", "sentence_candidate_id": "s_000001", "reason": "x"}]}',
                )
            )
        ]
    )

    output = question_gen.parse_llm_context_selection_response(message)
    occurrences, _ = question_gen.extract_question_occurrences(
        mapped_payload(),
        allowed_question_types=["context_meaning_choice"],
    )
    groups = question_gen.build_context_selection_groups(occurrences, top_k=5)
    with pytest.raises(ValueError, match="unknown group_id"):
        question_gen.apply_context_selections(groups, output)


def test_load_deepseek_config_defaults_to_beta_base_url_for_strict_tools(tmp_path: Path, monkeypatch):
    question_gen = load_module()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    _, base_url = question_gen.load_deepseek_config(tmp_path / ".env")

    assert base_url == "https://api.deepseek.com/beta"


def test_parse_llm_tool_call_response_accepts_valid_arguments():
    question_gen = load_module()
    payload = {
        "results": [],
        "rejections": [
            {
                "candidate_id": "c_000001",
                "reason": "上下文不足。",
            }
        ],
    }
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=question_gen.QUESTION_TOOL_NAME,
                    arguments=json.dumps(payload, ensure_ascii=False),
                )
            )
        ]
    )

    result = question_gen.parse_llm_tool_call_response(message)

    assert result.rejections[0].candidate_id == "c_000001"


def test_parse_llm_tool_call_response_accepts_first_json_object_with_extra_data():
    question_gen = load_module()
    payload = {
        "results": [],
        "rejections": [
            {
                "candidate_id": "c_000001",
                "reason": "上下文不足。",
            }
        ],
    }
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=question_gen.QUESTION_TOOL_NAME,
                    arguments=json.dumps(payload, ensure_ascii=False) + "\n{\"results\": [], \"rejections\": []}",
                )
            )
        ]
    )

    result = question_gen.parse_llm_tool_call_response(message)

    assert result.rejections[0].candidate_id == "c_000001"


def test_parse_llm_tool_call_response_rejects_missing_rejections():
    question_gen = load_module()
    payload = {
        "results": [
            {
                "candidate_id": "c_000001",
                "question_type": "context_meaning_choice",
                "content_payload": {
                    "question": "这里的 sacred 是什么意思？",
                    "context_text": "The most sacred thing I do is care.",
                    "options": [
                        {"id": "correct", "text": "神圣、非常重要"},
                        {"id": "wrong_1", "text": "普通、随便"},
                        {"id": "wrong_2", "text": "昂贵、奢侈"},
                        {"id": "wrong_3", "text": "快速、临时"},
                    ],
                    "explanation": "sacred 在这里表示非常重要。",
                },
            }
        ]
    }
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=question_gen.QUESTION_TOOL_NAME,
                    arguments=json.dumps(payload, ensure_ascii=False),
                )
            )
        ]
    )

    with pytest.raises(ValidationError, match="rejections"):
        question_gen.parse_llm_tool_call_response(message)


def test_parse_llm_tool_call_response_rejects_missing_tool_calls():
    question_gen = load_module()

    with pytest.raises(ValueError, match="did not return tool_calls"):
        question_gen.parse_llm_tool_call_response(SimpleNamespace(content='{"results": []}'))


def test_parse_llm_tool_call_response_rejects_wrong_tool_name():
    question_gen = load_module()
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name="other_tool",
                    arguments='{"results": [], "rejections": []}',
                )
            )
        ]
    )

    with pytest.raises(ValueError, match="unsupported tool"):
        question_gen.parse_llm_tool_call_response(message)


def test_parse_llm_tool_call_response_rejects_invalid_json_arguments():
    question_gen = load_module()
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name=question_gen.QUESTION_TOOL_NAME,
                    arguments="{bad json",
                )
            )
        ]
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        question_gen.parse_llm_tool_call_response(message)


def test_merge_fills_script_owned_catalog_question_fields():
    question_gen = load_module()
    candidates = select_first_sentence_candidates(question_gen)
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
    )

    assert question.model_dump() == {
        "scope_type": "video_unit",
        "question_type": "context_meaning_choice",
        "coarse_unit_id": 138446,
        "target_text": "sacred",
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


def test_run_generation_with_fake_llm_writes_wrapper_audit_and_progress(capsys, tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    class FakeLLM:
        model_name = "fake-deepseek"

        def invoke_question_batch(self, messages):
            assert "context_start_ms" not in messages[-1]["content"]
            return question_gen.AIQuestionBatchOutput.model_validate(
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
                }
            )

    question_gen.run_generation(
        mapped_json=mapped_path,
        output_json=output_path,
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        batch_size=10,
        llm=FakeLLM(),
        model_name="fake-deepseek",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"]["mapped_json"] == str(mapped_path)
    assert payload["source"]["model"] == "fake-deepseek"
    assert payload["audit"] == {
        "candidate_count": 2,
        "generated_count": 1,
        "rejected_count": 1,
    }
    assert payload["questions"][0]["scope_type"] == "video_unit"
    assert "video_id" not in payload["source"]
    assert "video_id" not in payload["questions"][0]
    assert payload["questions"][0]["coarse_unit_id"] == 130328
    assert payload["questions"][0]["context_start_ms"] == 1000
    assert payload["questions"][0]["context_end_ms"] == 5000
    assert list(payload.keys())[-1] == "selected_coarse_unit_refs"
    assert payload["selected_coarse_unit_refs"]["refs"] == [
        {"coarse_unit_id": 130328, "sentence_index": 0, "token_index": 2},
        {"coarse_unit_id": 138446, "sentence_index": 0, "token_index": 1},
    ]
    assert "candidate_checkpoint" not in payload
    assert (tmp_path / "log" / "mapped.json.question_audit.jsonl").exists()
    captured = capsys.readouterr().out
    assert "candidate_count: 2" in captured
    assert "hard_filter_reject_count:" in captured
    assert "remaining_candidate_count: 2" in captured
    assert "ai_batch_count: 1" in captured
    assert "[AI batch 1/1] candidates=2" in captured


def test_run_generation_writes_candidate_checkpoint_before_question_batches(tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    class FailingQuestionLLM:
        def invoke_question_batch(self, messages):
            raise RuntimeError("question generation stopped")

    with pytest.raises(RuntimeError, match="question generation stopped"):
        question_gen.run_generation(
            mapped_json=mapped_path,
            output_json=output_path,
            allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
            batch_size=10,
            llm=FailingQuestionLLM(),
            model_name="fake-deepseek",
        )

    temp_payload = json.loads((tmp_path / "temp" / output_path.name).read_text(encoding="utf-8"))
    checkpoint = temp_payload["candidate_checkpoint"]
    assert list(temp_payload.keys())[-1] == "candidate_checkpoint"
    assert checkpoint["version"] == question_gen.CHECKPOINT_VERSION
    assert checkpoint["selection_model"] == question_gen.DEFAULT_SELECTION_MODEL
    assert checkpoint["selection_top_k"] == question_gen.DEFAULT_SELECTION_TOP_K
    assert [candidate["candidate_id"] for candidate in checkpoint["candidates"]] == ["c_000001", "c_000002"]


def test_run_generation_resumes_candidate_checkpoint_without_selection_llm(tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    class FailingQuestionLLM:
        def invoke_question_batch(self, messages):
            raise RuntimeError("question generation stopped")

    with pytest.raises(RuntimeError):
        question_gen.run_generation(
            mapped_json=mapped_path,
            output_json=output_path,
            allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
            batch_size=10,
            llm=FailingQuestionLLM(),
            model_name="fake-deepseek",
        )

    class FailingSelectionLLM:
        def invoke_context_selection_batch(self, messages):
            raise AssertionError("selection checkpoint should be reused")

    class FakeQuestionLLM:
        def invoke_question_batch(self, messages):
            assert "c_000001" in messages[-1]["content"]
            return question_gen.AIQuestionBatchOutput.model_validate(
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
                        {"candidate_id": "c_000002", "reason": "测试只生成一道题。"}
                    ],
                }
            )

    final_output = question_gen.run_generation(
        mapped_json=mapped_path,
        output_json=output_path,
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        batch_size=10,
        llm=FakeQuestionLLM(),
        model_name="fake-deepseek",
        selection_llm=FailingSelectionLLM(),
    )

    assert final_output.audit.candidate_count == 2
    final_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(final_payload.keys())[-1] == "selected_coarse_unit_refs"
    assert final_payload["selected_coarse_unit_refs"]["refs"] == [
        {"coarse_unit_id": 130328, "sentence_index": 0, "token_index": 2},
        {"coarse_unit_id": 138446, "sentence_index": 0, "token_index": 1},
    ]
    assert "candidate_checkpoint" not in final_payload
    assert "processed_candidate_ids" not in final_payload["audit"]


def test_run_generation_resumes_from_temp_and_skips_processed_candidates(tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    candidates = select_first_sentence_candidates(question_gen)
    existing_question = question_gen.merge_ai_result(
        candidates[0],
        question_gen.AIQuestionResult.model_validate(
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
        ),
    )
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    temp_payload = question_gen.build_final_output(
        mapped_json=mapped_path,
        model_name="fake-deepseek",
        questions=[existing_question],
        candidate_count=2,
        rejected_count=1,
        candidate_checkpoint=question_gen.build_candidate_checkpoint(
            candidates=candidates,
            selection_model_name=question_gen.DEFAULT_SELECTION_MODEL,
            selection_top_k=question_gen.DEFAULT_SELECTION_TOP_K,
            allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        ),
    ).model_dump()
    temp_payload["audit"]["processed_candidate_ids"] = ["c_000001", "c_000002"]
    (temp_dir / output_path.name).write_text(json.dumps(temp_payload, ensure_ascii=False), encoding="utf-8")

    class FailingLLM:
        def invoke_question_batch(self, messages):
            raise AssertionError("resume should skip already processed candidates")

    final_output = question_gen.run_generation(
        mapped_json=mapped_path,
        output_json=output_path,
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        batch_size=10,
        llm=FailingLLM(),
        model_name="fake-deepseek",
    )

    assert len(final_output.questions) == 1
    assert final_output.audit.rejected_count == 1
    assert output_path.exists()
    assert not (temp_dir / output_path.name).exists()
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "processed_candidate_ids" not in output_payload["audit"]


def test_run_generation_resumes_from_temp_and_continues_unprocessed_candidates(tmp_path: Path):
    question_gen = load_module()
    mapped_path = tmp_path / "mapped.json"
    output_path = tmp_path / "questions.json"
    mapped_path.write_text(json.dumps(mapped_payload(), ensure_ascii=False), encoding="utf-8")

    candidates = select_first_sentence_candidates(question_gen)
    existing_question = question_gen.merge_ai_result(
        candidates[0],
        question_gen.AIQuestionResult.model_validate(
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
        ),
    )
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    temp_payload = question_gen.build_final_output(
        mapped_json=mapped_path,
        model_name="fake-deepseek",
        questions=[existing_question],
        candidate_count=2,
        rejected_count=0,
        candidate_checkpoint=question_gen.build_candidate_checkpoint(
            candidates=candidates,
            selection_model_name=question_gen.DEFAULT_SELECTION_MODEL,
            selection_top_k=question_gen.DEFAULT_SELECTION_TOP_K,
            allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        ),
    ).model_dump()
    temp_payload["audit"]["processed_candidate_ids"] = ["c_000001"]
    (temp_dir / output_path.name).write_text(json.dumps(temp_payload, ensure_ascii=False), encoding="utf-8")

    class FakeLLM:
        def invoke_question_batch(self, messages):
            assert "c_000001" not in messages[-1]["content"]
            assert "c_000002" in messages[-1]["content"]
            return question_gen.AIQuestionBatchOutput.model_validate(
                {
                    "results": [
                        {
                            "candidate_id": "c_000002",
                            "question_type": "context_meaning_choice",
                            "content_payload": {
                                "question": "这里的 sacred 最接近什么意思？",
                                "context_text": "The most sacred thing I do is care and provide for my workers.",
                                "options": [
                                    {"id": "correct", "text": "神圣、非常重要"},
                                    {"id": "wrong_1", "text": "普通、随便"},
                                    {"id": "wrong_2", "text": "昂贵、奢侈"},
                                    {"id": "wrong_3", "text": "快速、临时"},
                                ],
                                "explanation": "sacred 在这里表示非常重要、不可轻视。",
                            },
                        }
                    ],
                    "rejections": [],
                }
            )

    final_output = question_gen.run_generation(
        mapped_json=mapped_path,
        output_json=output_path,
        allowed_question_types=["context_meaning_choice", "context_cloze_choice"],
        batch_size=10,
        llm=FakeLLM(),
        model_name="fake-deepseek",
    )

    assert [question.target_text for question in final_output.questions] == ["provide for", "sacred"]
    assert not (temp_dir / output_path.name).exists()
