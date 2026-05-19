from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module():
    module_name = "question_generation_gemini_batch"
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / "6question-generation-batch.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_video_gcs_uri_uses_input_stem():
    batch = load_module()

    uri = batch.build_video_gcs_uri(
        "gs://videos2077/test-video/original",
        Path("resource/The Office BD/mapped/The Office (US) - clip1.json"),
    )

    assert uri == "gs://videos2077/test-video/original/The Office (US) - clip1.mp4"


def test_build_pending_jobs_skips_existing_targets(tmp_path: Path):
    batch = load_module()
    source_dir = tmp_path / "mapped"
    target_dir = tmp_path / "questions"
    source_dir.mkdir()
    target_dir.mkdir()
    source_a = source_dir / "clip1.json"
    source_b = source_dir / "clip2.json"
    source_a.write_text("{}", encoding="utf-8")
    source_b.write_text("{}", encoding="utf-8")
    (target_dir / "clip1.json").write_text("{}", encoding="utf-8")

    pending, skipped = batch.build_pending_jobs([source_a, source_b], target_dir)

    assert skipped == 1
    assert pending == [(source_b, target_dir / "clip2.json")]


def test_run_single_file_invokes_gemini_script_with_video_uri(monkeypatch, tmp_path: Path):
    batch = load_module()
    source = tmp_path / "mapped.json"
    target = tmp_path / "questions.json"
    calls = []

    def fake_run(cmd, cwd, capture_output, text, check):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "check": check,
            }
        )

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    success, detail, video_gcs_uri = batch.run_single_file(
        source_path=source,
        target_path=target,
        video_gcs_dir="gs://videos2077/test-video/original/",
        question_types="context_meaning_choice,context_cloze_choice",
        batch_size=4,
        env_path=tmp_path / ".env",
        model="gemini-3.1-pro-preview",
        question_thinking_level="high",
        selection_model="gemini-3.1-flash-lite-preview",
        selection_thinking_level="high",
        selection_top_k=6,
        selection_batch_size=12,
        selection_max_workers=4,
        candidate_score_threshold=6.0,
        video_mime_type="video/mp4",
        cache_ttl_seconds=1800,
    )

    assert success is True
    assert detail == "ok"
    assert video_gcs_uri == "gs://videos2077/test-video/original/mapped.mp4"
    assert calls == [
        {
            "cmd": [
                sys.executable,
                str(batch.QUESTION_SCRIPT),
                str(source),
                str(target),
                "--question-types",
                "context_meaning_choice,context_cloze_choice",
                "--batch-size",
                "4",
                "--env-path",
                str(tmp_path / ".env"),
                "--model",
                "gemini-3.1-pro-preview",
                "--question-thinking-level",
                "high",
                "--selection-model",
                "gemini-3.1-flash-lite-preview",
                "--selection-thinking-level",
                "high",
                "--selection-top-k",
                "6",
                "--selection-batch-size",
                "12",
                "--selection-max-workers",
                "4",
                "--candidate-score-threshold",
                "6.0",
                "--video-gcs-uri",
                "gs://videos2077/test-video/original/mapped.mp4",
                "--video-mime-type",
                "video/mp4",
                "--cache-ttl-seconds",
                "1800",
            ],
            "cwd": batch.ROOT_DIR,
            "capture_output": True,
            "text": True,
            "check": False,
        }
    ]
