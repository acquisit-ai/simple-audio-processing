from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_ffmpeg_clipping_module():
    spec = importlib.util.spec_from_file_location(
        "ffmpeg_clipping_under_test",
        ROOT_DIR / "4ffmpeg-clipping.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_slice_transcript_includes_complete_clip_metadata():
    ffmpeg_clipping = load_ffmpeg_clipping_module()
    transcript_data = {
        "total_sentences": 2,
        "total_tokens": 2,
        "sentences": [
            {
                "index": 10,
                "text": "First.",
                "start": 1_000,
                "end": 2_000,
                "tokens": [
                    {"index": 4, "text": "First.", "start": 1_000, "end": 2_000}
                ],
            },
            {
                "index": 11,
                "text": "Second.",
                "start": 2_500,
                "end": 3_000,
                "tokens": [
                    {"index": 7, "text": "Second.", "start": 2_500, "end": 3_000}
                ],
            },
        ],
    }
    clip_plan = {
        "clip_id": 1,
        "title": "塞伯商店试营业",
        "description": "德怀特负责塞伯商店的试营业。",
        "engagement": {
            "drama": 4,
            "humor": 7,
            "payoff": 6,
            "standalone": 7,
            "reasoning": "适合作为开篇。",
        },
        "start_index": 10,
        "end_index": 11,
        "start_time": 1_000,
        "end_time": 3_000,
        "buffered_start_time": 900,
        "buffered_end_time": 3_050,
        "duration_time": 2_150,
        "reasoning": "自然结束。",
    }

    clip_transcript = ffmpeg_clipping.slice_transcript_for_clip(
        transcript_data=transcript_data,
        start_index=clip_plan["start_index"],
        end_index=clip_plan["end_index"],
        time_offset_ms=clip_plan["buffered_start_time"],
        clip_metadata=clip_plan,
    )

    for key, value in clip_plan.items():
        assert clip_transcript[key] == value

    assert clip_transcript["sentences"][0]["index"] == 0
    assert clip_transcript["sentences"][0]["start"] == 100
    assert clip_transcript["sentences"][0]["tokens"][0]["index"] == 0
    assert clip_transcript["sentences"][0]["tokens"][0]["start"] == 100
    assert clip_transcript["total_sentences"] == 2
    assert clip_transcript["total_tokens"] == 2
