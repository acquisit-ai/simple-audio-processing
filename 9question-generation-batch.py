#!/usr/bin/env python3
"""
批量调用 9question-generation-deepseek.py 处理 mapped JSON。

默认行为：
- 从 `resource/The Office BD/mapped` 读取 mapped JSON
- 输出到 `resource/The Office BD/questions`
- 最多开启 3 个并发任务
- 如果目标目录已存在同名文件，则直接跳过
- 屏蔽子进程 `9question-generation-deepseek.py` 的命令行输出
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
DEFAULT_MAX_WORKERS = 3
DEFAULT_MAX_QUESTIONS = 20
DEFAULT_BATCH_SIZE = 10
DEFAULT_QUESTION_TYPES = "context_meaning_choice,context_cloze_choice"
DEFAULT_MODEL = "deepseek-v4-pro"
ROOT_DIR = Path(__file__).resolve().parent
QUESTION_SCRIPT = ROOT_DIR / "9question-generation-deepseek.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run 9question-generation-deepseek.py over mapped JSON files.")
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
        "--video-id",
        required=True,
        help="catalog.videos.video_id passed to every generated video_unit question.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent question generation jobs.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Maximum final questions per input file.",
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
        help="Path to .env containing DEEPSEEK_API_KEY and optional DEEPSEEK_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DeepSeek model passed to the single-file generator.",
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


def run_single_file(
    source_path: Path,
    target_path: Path,
    video_id: str,
    max_questions: int,
    question_types: str,
    batch_size: int,
    env_path: Path,
    model: str,
) -> tuple[bool, str]:
    """
    调用 9question-generation-deepseek.py 处理单个 mapped JSON。

    返回：
    - success: 是否成功
    - detail: 成功或失败的简短说明
    """

    command = [
        sys.executable,
        str(QUESTION_SCRIPT),
        str(source_path),
        str(target_path),
        "--video-id",
        video_id,
        "--max-questions",
        str(max_questions),
        "--question-types",
        question_types,
        "--batch-size",
        str(batch_size),
        "--env-path",
        str(env_path),
        "--model",
        model,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception as exc:
        return False, f"启动失败: {exc}"

    if completed.returncode == 0:
        return True, "ok"
    return False, f"exit code {completed.returncode}"


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    if args.max_questions < 1:
        raise SystemExit("--max-questions must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    if not QUESTION_SCRIPT.exists():
        raise FileNotFoundError(f"Question generation script not found: {QUESTION_SCRIPT}")

    source_dir = args.source_dir
    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_source_files(source_dir)
    if not source_files:
        print("没有找到可处理的 JSON 文件。", flush=True)
        return

    print("=== 题目生成批处理启动 ===", flush=True)
    print(f"源目录：{source_dir}", flush=True)
    print(f"目标目录：{target_dir}", flush=True)
    print(f"video_id：{args.video_id}", flush=True)
    print(f"并发数：{args.max_workers}", flush=True)
    print(f"每文件题目上限：{args.max_questions}", flush=True)
    print(f"题型：{args.question_types}", flush=True)
    print(f"候选 batch size：{args.batch_size}", flush=True)
    print(f"模型：{args.model}", flush=True)
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
                args.video_id,
                args.max_questions,
                args.question_types,
                args.batch_size,
                args.env_path,
                args.model,
            ): (source_path, target_path)
            for source_path, target_path in pending_jobs
        }

        print(f"已提交任务：{len(future_to_job)}", flush=True)

        for future in as_completed(future_to_job):
            source_path, target_path = future_to_job[future]
            completed_count += 1

            try:
                success, detail = future.result()
            except Exception as exc:
                success = False
                detail = f"线程异常: {exc}"

            if success:
                success_count += 1
                status = "成功"
            else:
                failed_count += 1
                status = "失败"

            print(
                f"[{completed_count}/{total_to_run}] {status} | "
                f"{source_path.name} -> {target_path.name} | {detail}",
                flush=True,
            )

    print("\n=== 题目生成批处理结束 ===", flush=True)
    print(f"总文件数：{len(source_files)}", flush=True)
    print(f"跳过：{skipped}", flush=True)
    print(f"执行：{total_to_run}", flush=True)
    print(f"成功：{success_count}", flush=True)
    print(f"失败：{failed_count}", flush=True)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
