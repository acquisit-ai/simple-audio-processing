#!/usr/bin/env python3
"""
批量调用 5agent-mapping-deepseek.py 处理 transcript JSON。

默认行为：
- 从 `resource/The Office BD/clips-transcripts` 读取源 JSON
- 输出到 `resource/The Office BD/mapped`
- 最多开启 5 个并发任务
- 如果目标目录已存在同名文件，则直接跳过
- 屏蔽子进程 `5agent-mapping-deepseek.py` 的命令行输出
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("resource/The Office BD/clips-transcripts")
DEFAULT_TARGET_DIR = Path("resource/The Office BD/mapped")
DEFAULT_MAX_WORKERS = 5
ROOT_DIR = Path(__file__).resolve().parent
MAPPING_SCRIPT = ROOT_DIR / "5agent-mapping-deepseek.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run 5agent-mapping-deepseek.py over transcript JSON files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source transcript JSON files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory to store mapped JSON files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent mapping jobs.",
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


def run_single_file(source_path: Path, target_path: Path) -> tuple[bool, str]:
    """
    调用 5agent-mapping-deepseek.py 处理单个文件。

    返回：
    - success: 是否成功
    - detail: 成功或失败的简短说明
    """

    try:
        completed = subprocess.run(
            [sys.executable, str(MAPPING_SCRIPT), str(source_path), str(target_path)],
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

    if not MAPPING_SCRIPT.exists():
        raise FileNotFoundError(f"Mapping script not found: {MAPPING_SCRIPT}")

    source_dir = args.source_dir
    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_source_files(source_dir)
    if not source_files:
        print("没有找到可处理的 JSON 文件。", flush=True)
        return

    print("=== 批处理启动 ===", flush=True)
    print(f"源目录：{source_dir}", flush=True)
    print(f"目标目录：{target_dir}", flush=True)
    print(f"并发数：{args.max_workers}", flush=True)
    print(f"源文件总数：{len(source_files)}", flush=True)

    skipped = 0
    pending_jobs: list[tuple[Path, Path]] = []
    for source_path in source_files:
        target_path = target_dir / source_path.name
        if target_path.exists():
            skipped += 1
            continue
        pending_jobs.append((source_path, target_path))

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
            executor.submit(run_single_file, source_path, target_path): (source_path, target_path)
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
                f"[{completed_count}/{total_to_run}] {status} | {source_path.name} -> {target_path.name} | {detail}",
                flush=True,
            )

    print("\n=== 批处理结束 ===", flush=True)
    print(f"总文件数：{len(source_files)}", flush=True)
    print(f"跳过：{skipped}", flush=True)
    print(f"执行：{total_to_run}", flush=True)
    print(f"成功：{success_count}", flush=True)
    print(f"失败：{failed_count}", flush=True)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
