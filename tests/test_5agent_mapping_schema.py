from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *_args, **_kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

langchain_core_module = types.ModuleType("langchain_core")
langchain_messages_module = types.ModuleType("langchain_core.messages")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


langchain_messages_module.HumanMessage = _Message
langchain_messages_module.SystemMessage = _Message
langchain_core_module.messages = langchain_messages_module
sys.modules.setdefault("langchain_core", langchain_core_module)
sys.modules.setdefault("langchain_core.messages", langchain_messages_module)

langchain_openai_module = types.ModuleType("langchain_openai")


class _ChatOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def with_structured_output(self, schema):
        return self


langchain_openai_module.ChatOpenAI = _ChatOpenAI
sys.modules.setdefault("langchain_openai", langchain_openai_module)

pydantic_module = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs) -> None:
        annotations = {}
        for cls in reversed(type(self).__mro__):
            annotations.update(getattr(cls, "__annotations__", {}))
        for field_name in annotations:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
                continue
            if hasattr(type(self), field_name):
                setattr(self, field_name, getattr(type(self), field_name))
                continue
            setattr(self, field_name, None)

    def model_dump(self) -> dict[str, object]:
        annotations = {}
        for cls in reversed(type(self).__mro__):
            annotations.update(getattr(cls, "__annotations__", {}))
        return {field_name: getattr(self, field_name) for field_name in annotations}

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Cannot validate value for {cls.__name__}: {value!r}")


def _field(*, default=None, description=None):
    return default


class _ValidationError(Exception):
    pass


pydantic_module.BaseModel = _BaseModel
pydantic_module.Field = _field
pydantic_module.ValidationError = _ValidationError
sys.modules.setdefault("pydantic", pydantic_module)


MODULE_PATH = Path(__file__).resolve().parents[1] / "5agent-mapping.py"
SPEC = importlib.util.spec_from_file_location("agent_mapping", MODULE_PATH)
assert SPEC and SPEC.loader
agent_mapping = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("agent_mapping", agent_mapping)
SPEC.loader.exec_module(agent_mapping)


class AgentMappingSchemaTests(unittest.TestCase):
    def test_validate_and_index_batch_preserves_snake_case_semantic_fields(self) -> None:
        original_batch = [
            {
                "index": 0,
                "text": "wanted to",
                "tokens": [
                    {"text": "wanted"},
                    {"text": "to"},
                ],
            }
        ]
        stage_one_output = {
            "sentences": [
                {
                    "index": 0,
                    "text": "wanted to",
                    "explanation": "想要做某事",
                    "tokens": [
                        {
                            "text": "wanted to",
                            "explanation": "想要做某事",
                            "semantic_element": {
                                "base_form": "want",
                                "dictionary": "想要；希望",
                            },
                        }
                    ],
                }
            ]
        }

        validated = agent_mapping.validate_and_index_batch(original_batch, stage_one_output)

        token = validated["sentences"][0]["tokens"][0]
        self.assertIn("semantic_element", token)
        self.assertNotIn("semanticElement", token)
        self.assertEqual(token["semantic_element"]["base_form"], "want")

    def test_finalize_match_writes_database_fields_into_snake_case_semantic_element(self) -> None:
        token_runtime = agent_mapping.TokenRuntime(
            sentence_index=0,
            token_index=0,
            sentence_text="wanted to",
            token={
                "text": "wanted to",
                "explanation": "想要做某事",
                "semantic_element": {
                    "base_form": "want",
                    "dictionary": "想要；希望",
                },
            },
        )
        decision = agent_mapping.StageThreeDecision(
            action="match",
            coarse_id=166670,
            reason="匹配到 want 的动词义。",
        )
        rounds = [
            agent_mapping.SearchRoundRecord(
                round_no=1,
                mode="exact",
                queries=["wanted to", "want"],
                results={
                    "results": [
                        {
                            "query": "want",
                            "rows": [
                                {
                                    "id": 166670,
                                    "label": "want",
                                    "chinese_def": "表达主观的想要、请求或寻求某人/某物。",
                                }
                            ],
                        }
                    ]
                },
            )
        ]

        final_action, final_coarse_id, final_reason = agent_mapping.finalize_match(
            token_runtime,
            decision,
            rounds,
        )

        semantic_element = token_runtime.token["semantic_element"]
        self.assertEqual(final_action, "match")
        self.assertEqual(final_coarse_id, 166670)
        self.assertEqual(final_reason, "匹配到 want 的动词义。")
        self.assertEqual(semantic_element["coarse_id"], 166670)
        self.assertEqual(semantic_element["base_form"], "want")
        self.assertEqual(semantic_element["dictionary"], "表达主观的想要、请求或寻求某人/某物。")
        self.assertEqual(semantic_element["reason"], "匹配到 want 的动词义。")

    def test_process_single_token_reads_base_form_from_snake_case_for_direct_no_match(self) -> None:
        records: list[dict[str, object]] = []

        class AuditLoggerStub:
            def write(self, record: dict[str, object]) -> None:
                records.append(record)

        class QueryRunnerStub:
            def run(self, mode: str, queries: list[str]) -> dict[str, object]:
                raise AssertionError("direct no_match path should not hit the query runner")

        token_runtime = agent_mapping.TokenRuntime(
            sentence_index=0,
            token_index=0,
            sentence_text="It's fine.",
            token={
                "text": "It's",
                "explanation": "It is 的缩写。",
                "semantic_element": {
                    "base_form": "it",
                    "dictionary": "它；这；那",
                },
            },
        )

        agent_mapping.process_single_token(
            stage_three_llm=None,
            query_runner=QueryRunnerStub(),
            audit_logger=AuditLoggerStub(),
            shared_context={"sentences": []},
            token_runtime=token_runtime,
            token_position=1,
            total_tokens=1,
        )

        semantic_element = token_runtime.token["semantic_element"]
        self.assertIsNone(semantic_element["coarse_id"])
        self.assertEqual(semantic_element["reason"], agent_mapping.DIRECT_NO_MATCH_REASON)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["final_action"], "no_match")


if __name__ == "__main__":
    unittest.main()
