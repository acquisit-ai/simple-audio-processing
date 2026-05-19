from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_batch_module():
    spec = importlib.util.spec_from_file_location(
        "ffmpeg_clipping_batch_under_test",
        ROOT_DIR / "4ffmpeg-clipping-batch.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubFfmpegClippingModule:
    def __init__(self):
        self.calls = []

    def is_valid_existing_clip(self, video_path: Path, transcript_path: Path) -> bool:
        if not video_path.exists() or video_path.stat().st_size == 0:
            return False
        if not transcript_path.exists() or transcript_path.stat().st_size == 0:
            return False
        try:
            json.loads(transcript_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return True

    def clip_video_and_transcript(self, **kwargs):
        self.calls.append(kwargs)


def write_clip_plan(path: Path, clip_count: int) -> None:
    path.write_text(
        json.dumps(
            {
                "clips": [
                    {"start_index": index, "end_index": index}
                    for index in range(clip_count)
                ]
            }
        ),
        encoding="utf-8",
    )


def write_existing_clip_pair(output_dir: Path, stem: str, clip_number: int) -> None:
    (output_dir / f"{stem}-clip{clip_number}.mp4").write_bytes(b"video")
    (output_dir / f"{stem}-clip{clip_number}.json").write_text(
        json.dumps({"sentences": []}),
        encoding="utf-8",
    )


def test_batch_skip_existing_defaults_to_true():
    batch = load_batch_module()

    skip_existing_default = inspect.signature(batch.run_batch_clipping).parameters[
        "skip_existing"
    ].default

    assert skip_existing_default is True


def test_run_single_video_job_skips_when_all_expected_outputs_exist(tmp_path: Path):
    batch = load_batch_module()
    stub = StubFfmpegClippingModule()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"source video")
    transcript_path = tmp_path / "episode.json"
    transcript_path.write_text(json.dumps({"sentences": []}), encoding="utf-8")
    clipping_plan_path = tmp_path / "plan.json"
    write_clip_plan(clipping_plan_path, clip_count=2)
    write_existing_clip_pair(output_dir, "episode", 1)
    write_existing_clip_pair(output_dir, "episode", 2)

    result = batch.run_single_video_job(
        ffmpeg_clipping_module=stub,
        task_number=1,
        total_jobs=1,
        job={
            "video_path": video_path,
            "transcript_path": transcript_path,
            "clipping_plan_path": clipping_plan_path,
        },
        output_dir=output_dir,
        skip_existing=True,
    )

    assert result["success"] is True
    assert result["skipped_existing"] is True
    assert stub.calls == []


def test_run_single_video_job_overwrites_all_when_any_expected_output_is_missing(
    tmp_path: Path,
):
    batch = load_batch_module()
    stub = StubFfmpegClippingModule()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"source video")
    transcript_path = tmp_path / "episode.json"
    transcript_path.write_text(json.dumps({"sentences": []}), encoding="utf-8")
    clipping_plan_path = tmp_path / "plan.json"
    write_clip_plan(clipping_plan_path, clip_count=2)
    write_existing_clip_pair(output_dir, "episode", 1)

    result = batch.run_single_video_job(
        ffmpeg_clipping_module=stub,
        task_number=1,
        total_jobs=1,
        job={
            "video_path": video_path,
            "transcript_path": transcript_path,
            "clipping_plan_path": clipping_plan_path,
        },
        output_dir=output_dir,
        skip_existing=True,
    )

    assert result["success"] is True
    assert result["skipped_existing"] is False
    assert len(stub.calls) == 1
    assert stub.calls[0]["skip_existing"] is False
