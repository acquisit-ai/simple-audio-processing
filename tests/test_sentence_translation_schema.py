from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pydantic import ValidationError
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("filename", "module_name"),
    [
        ("5agent-mapping.py", "agent_mapping_translation_schema"),
        ("5agent-mapping-deepseek.py", "agent_mapping_deepseek_translation_schema"),
    ],
)
def test_stage_one_sentence_uses_translation_not_explanation(filename: str, module_name: str):
    mapping = load_module(filename, module_name)

    sentence = mapping.StageOneSentence.model_validate(
        {
            "index": 0,
            "text": "I wanted to help.",
            "translation": "我当时想帮忙。",
            "tokens": [
                {
                    "text": "I",
                    "explanation": "第一人称单数主语。",
                    "semantic_element": {
                        "base_form": "I",
                        "translation": "我",
                        "dictionary": "第一人称单数代词。",
                    },
                }
            ],
        }
    )

    assert sentence.translation == "我当时想帮忙。"


@pytest.mark.parametrize(
    ("filename", "module_name"),
    [
        ("5agent-mapping.py", "agent_mapping_explanation_rejection"),
        ("5agent-mapping-deepseek.py", "agent_mapping_deepseek_explanation_rejection"),
    ],
)
def test_stage_one_sentence_rejects_sentence_level_explanation(filename: str, module_name: str):
    mapping = load_module(filename, module_name)

    with pytest.raises(ValidationError):
        mapping.StageOneSentence.model_validate(
            {
                "index": 0,
                "text": "I wanted to help.",
                "translation": "我当时想帮忙。",
                "explanation": "我当时想帮忙。",
                "tokens": [],
            }
        )
