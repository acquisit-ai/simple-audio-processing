#!/usr/bin/env python3
"""
批量调用 6question-generation-gemini.py 处理 mapped JSON。

默认行为：
- 从 `resource/The Office BD/mapped` 读取 mapped JSON
- 输出到 `resource/The Office BD/questions`
- 默认从 GCS 视频目录拼接同名 .mp4 作为多模态输入
- 最多开启 4 个并发任务
- 如果目标目录已存在同名文件，则直接跳过
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("resource/The Office BD/mapped")
DEFAULT_TARGET_DIR = Path("resource/The Office BD/questions")
DEFAULT_VIDEO_GCS_DIR = "gs://videos2077/test-video/clips/"
DEFAULT_MAX_WORKERS = 5
DEFAULT_BATCH_SIZE = 12
DEFAULT_QUESTION_TYPES = "context_meaning_choice,context_cloze_choice"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_QUESTION_THINKING_LEVEL = "high"
DEFAULT_SELECTION_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_SELECTION_THINKING_LEVEL = "high"
DEFAULT_SELECTION_TOP_K = 6
DEFAULT_SELECTION_BATCH_SIZE = 12
DEFAULT_SELECTION_MAX_WORKERS = 4
DEFAULT_CANDIDATE_SCORE_THRESHOLD = 6.0
DEFAULT_VIDEO_MIME_TYPE = "video/mp4"
DEFAULT_CACHE_TTL_SECONDS = 30 * 60
ROOT_DIR = Path(__file__).resolve().parent
QUESTION_SCRIPT = ROOT_DIR / "6question-generation-gemini.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run 6question-generation-gemini.py over mapped JSON files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing mapped JSON files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory to store generated question JSON files.",
    )
    parser.add_argument(
        "--video-gcs-dir",
        default=DEFAULT_VIDEO_GCS_DIR,
        help=f"GCS video directory. The batch runner appends each input JSON stem plus .mp4. Default: {DEFAULT_VIDEO_GCS_DIR}.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent question generation jobs.",
    )
    parser.add_argument(
        "--question-types",
        default=DEFAULT_QUESTION_TYPES,
        help="Comma-separated question types passed to the single-file generator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Candidate batch size passed to the single-file generator.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=ROOT_DIR / ".env",
        help="Path to .env containing optional GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model passed to the single-file generator.",
    )
    parser.add_argument(
        "--question-thinking-level",
        default=DEFAULT_QUESTION_THINKING_LEVEL,
        choices=["low", "medium", "high"],
        help="Gemini thinking level for question generation.",
    )
    parser.add_argument(
        "--selection-model",
        default=DEFAULT_SELECTION_MODEL,
        help="Gemini context-selection model passed to the single-file generator.",
    )
    parser.add_argument(
        "--selection-thinking-level",
        default=DEFAULT_SELECTION_THINKING_LEVEL,
        choices=["low", "medium", "high"],
        help="Gemini thinking level for context selection.",
    )
    parser.add_argument(
        "--selection-top-k",
        type=int,
        default=DEFAULT_SELECTION_TOP_K,
        help="Top sentence candidates per coarse unit passed to the single-file generator.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=DEFAULT_SELECTION_BATCH_SIZE,
        help="Context selection groups per AI call passed to the single-file generator.",
    )
    parser.add_argument(
        "--selection-max-workers",
        type=int,
        default=DEFAULT_SELECTION_MAX_WORKERS,
        help="Maximum concurrent context-selection AI calls passed to the single-file generator.",
    )
    parser.add_argument(
        "--candidate-score-threshold",
        type=float,
        default=DEFAULT_CANDIDATE_SCORE_THRESHOLD,
        help="Minimum weighted ref score to send to question generation.",
    )
    parser.add_argument(
        "--video-mime-type",
        default=DEFAULT_VIDEO_MIME_TYPE,
        help="Video MIME type passed to the single-file generator.",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=DEFAULT_CACHE_TTL_SECONDS,
        help="Explicit context cache TTL seconds passed to the single-file generator.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[object]:
    """按数字自然顺序排序，避免 clip10 排在 clip2 前面。"""

    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def collect_source_files(source_dir: Path) -> list[Path]:
    """读取源目录下所有 JSON 文件，并按自然顺序排序。"""

    return sorted(source_dir.glob("*.json"), key=natural_sort_key)


def build_video_gcs_uri(video_gcs_dir: str, source_path: Path) -> str:
    """根据输入 JSON 文件名拼出同名 GCS mp4 地址。"""

    normalized_dir = video_gcs_dir.rstrip("/") + "/"
    return f"{normalized_dir}{source_path.stem}.mp4"


def build_pending_jobs(source_files: list[Path], target_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
    """构建待处理任务，并统计已存在目标文件的跳过数量。"""

    skipped = 0
    pending_jobs: list[tuple[Path, Path]] = []
    for source_path in source_files:
        target_path = target_dir / source_path.name
        if target_path.exists():
            skipped += 1
            continue
        pending_jobs.append((source_path, target_path))
    return pending_jobs, skipped


def summarize_subprocess_output(stdout: str, stderr: str) -> str:
    detail = stderr.strip() or stdout.strip()
    if not detail:
        return ""
    lines = detail.splitlines()
    return "\n".join(lines[-20:])


def run_single_file(
    source_path: Path,
    target_path: Path,
    video_gcs_dir: str,
    question_types: str,
    batch_size: int,
    env_path: Path,
    model: str,
    question_thinking_level: str,
    selection_model: str,
    selection_thinking_level: str,
    selection_top_k: int,
    selection_batch_size: int,
    selection_max_workers: int,
    candidate_score_threshold: float,
    video_mime_type: str,
    cache_ttl_seconds: int,
) -> tuple[bool, str, str]:
    """
    调用 6question-generation-gemini.py 处理单个 mapped JSON。

    返回：
    - success: 是否成功
    - detail: 成功或失败的简短说明
    - video_gcs_uri: 本次传给单文件脚本的视频地址
    """

    video_gcs_uri = build_video_gcs_uri(video_gcs_dir, source_path)
    command = [
        sys.executable,
        str(QUESTION_SCRIPT),
        str(source_path),
        str(target_path),
        "--question-types",
        question_types,
        "--batch-size",
        str(batch_size),
        "--env-path",
        str(env_path),
        "--model",
        model,
        "--question-thinking-level",
        question_thinking_level,
        "--selection-model",
        selection_model,
        "--selection-thinking-level",
        selection_thinking_level,
        "--selection-top-k",
        str(selection_top_k),
        "--selection-batch-size",
        str(selection_batch_size),
        "--selection-max-workers",
        str(selection_max_workers),
        "--candidate-score-threshold",
        str(candidate_score_threshold),
        "--video-gcs-uri",
        video_gcs_uri,
        "--video-mime-type",
        video_mime_type,
        "--cache-ttl-seconds",
        str(cache_ttl_seconds),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"启动失败: {exc}", video_gcs_uri

    if completed.returncode == 0:
        return True, "ok", video_gcs_uri
    detail = summarize_subprocess_output(completed.stdout, completed.stderr)
    return False, f"exit code {completed.returncode}" + (f"\n{detail}" if detail else ""), video_gcs_uri


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.selection_top_k < 1:
        raise SystemExit("--selection-top-k must be at least 1")
    if args.selection_batch_size < 1:
        raise SystemExit("--selection-batch-size must be at least 1")
    if args.selection_max_workers < 1:
        raise SystemExit("--selection-max-workers must be at least 1")
    if args.candidate_score_threshold < 0:
        raise SystemExit("--candidate-score-threshold must be non-negative")
    if args.cache_ttl_seconds < 1:
        raise SystemExit("--cache-ttl-seconds must be at least 1")

    if not QUESTION_SCRIPT.exists():
        raise FileNotFoundError(f"Question generation script not found: {QUESTION_SCRIPT}")

    source_dir = args.source_dir
    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_source_files(source_dir)
    if not source_files:
        print("没有找到可处理的 JSON 文件。", flush=True)
        return

    normalized_video_gcs_dir = args.video_gcs_dir.rstrip("/") + "/"

    print("=== Gemini 题目生成批处理启动 ===", flush=True)
    print(f"源目录：{source_dir}", flush=True)
    print(f"目标目录：{target_dir}", flush=True)
    print(f"GCS 视频目录：{normalized_video_gcs_dir}", flush=True)
    print(f"并发数：{args.max_workers}", flush=True)
    print(f"题型：{args.question_types}", flush=True)
    print(f"候选 batch size：{args.batch_size}", flush=True)
    print(f"模型：{args.model}", flush=True)
    print(f"question thinking：{args.question_thinking_level}", flush=True)
    print(f"选句模型：{args.selection_model}", flush=True)
    print(f"selection thinking：{args.selection_thinking_level}", flush=True)
    print(f"选句 top K：{args.selection_top_k}", flush=True)
    print(f"选句 batch size：{args.selection_batch_size}", flush=True)
    print(f"选句并发数：{args.selection_max_workers}", flush=True)
    print(f"candidate score threshold：{args.candidate_score_threshold}", flush=True)
    print(f"cache TTL seconds：{args.cache_ttl_seconds}", flush=True)
    print(f"源文件总数：{len(source_files)}", flush=True)

    pending_jobs, skipped = build_pending_jobs(source_files, target_dir)
    total_to_run = len(pending_jobs)
    print(f"已跳过：{skipped}", flush=True)
    print(f"待处理：{total_to_run}", flush=True)

    if total_to_run == 0:
        print("没有需要新处理的文件。", flush=True)
        return

    completed_count = 0
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_single_file,
                source_path,
                target_path,
                normalized_video_gcs_dir,
                args.question_types,
                args.batch_size,
                args.env_path,
                args.model,
                args.question_thinking_level,
                args.selection_model,
                args.selection_thinking_level,
                args.selection_top_k,
                args.selection_batch_size,
                args.selection_max_workers,
                args.candidate_score_threshold,
                args.video_mime_type,
                args.cache_ttl_seconds,
            ): (source_path, target_path)
            for source_path, target_path in pending_jobs
        }

        print(f"已提交任务：{len(future_to_job)}", flush=True)

        for future in as_completed(future_to_job):
            source_path, target_path = future_to_job[future]
            completed_count += 1

            try:
                success, detail, video_gcs_uri = future.result()
            except Exception as exc:
                success = False
                detail = f"线程异常: {exc}"
                video_gcs_uri = build_video_gcs_uri(normalized_video_gcs_dir, source_path)

            if success:
                success_count += 1
                status = "成功"
            else:
                failed_count += 1
                status = "失败"

            print(
                f"[{completed_count}/{total_to_run}] {status} | "
                f"{source_path.name} -> {target_path.name} | video={video_gcs_uri} | {detail}",
                flush=True,
            )

    print("\n=== Gemini 题目生成批处理结束 ===", flush=True)
    print(f"总文件数：{len(source_files)}", flush=True)
    print(f"跳过：{skipped}", flush=True)
    print(f"执行：{total_to_run}", flush=True)
    print(f"成功：{success_count}", flush=True)
    print(f"失败：{failed_count}", flush=True)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
