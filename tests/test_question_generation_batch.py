from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module():
    module_name = "question_generation_batch"
    spec = importlib.util.spec_from_file_location(module_name, ROOT_DIR / "9question-generation-batch.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_source_files_uses_natural_json_order(tmp_path: Path):
    batch = load_module()
    for name in ["clip10.json", "clip2.json", "clip1.json", "notes.txt"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    files = batch.collect_source_files(tmp_path)

    assert [path.name for path in files] == ["clip1.json", "clip2.json", "clip10.json"]


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


def test_run_single_file_invokes_question_generation_script(monkeypatch, tmp_path: Path):
    batch = load_module()
    source = tmp_path / "mapped.json"
    target = tmp_path / "questions.json"
    calls = []

    def fake_run(cmd, cwd, stdout, stderr, check):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "stdout": stdout,
                "stderr": stderr,
                "check": check,
            }
        )

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    success, detail = batch.run_single_file(
        source_path=source,
        target_path=target,
        video_id="00000000-0000-0000-0000-000000000001",
        max_questions=12,
        question_types="context_meaning_choice,context_cloze_choice",
        batch_size=4,
        env_path=tmp_path / ".env",
        model="deepseek-v4-pro",
    )

    assert success is True
    assert detail == "ok"
    assert calls == [
        {
            "cmd": [
                sys.executable,
                str(batch.QUESTION_SCRIPT),
                str(source),
                str(target),
                "--video-id",
                "00000000-0000-0000-0000-000000000001",
                "--max-questions",
                "12",
                "--question-types",
                "context_meaning_choice,context_cloze_choice",
                "--batch-size",
                "4",
                "--env-path",
                str(tmp_path / ".env"),
                "--model",
                "deepseek-v4-pro",
            ],
            "cwd": batch.ROOT_DIR,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "check": False,
        }
    ]


def test_run_single_file_reports_nonzero_exit(monkeypatch, tmp_path: Path):
    batch = load_module()

    def fake_run(cmd, cwd, stdout, stderr, check):
        class Completed:
            returncode = 2

        return Completed()

    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    success, detail = batch.run_single_file(
        source_path=tmp_path / "mapped.json",
        target_path=tmp_path / "questions.json",
        video_id="00000000-0000-0000-0000-000000000001",
        max_questions=20,
        question_types="context_meaning_choice",
        batch_size=10,
        env_path=tmp_path / ".env",
        model="deepseek-v4-pro",
    )

    assert success is False
    assert detail == "exit code 2"
