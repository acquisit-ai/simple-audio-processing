#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ==========================================
# 1. 默认路径配置
#    - 递归扫描视频目录中的所有常见视频文件
#    - 只有在 2cleaned-data 和 3clipped 中都找到同名 JSON 时才处理
#    - 所有切片结果统一输出到 The Office BD clips
# ==========================================
DEFAULT_VIDEO_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-original")
DEFAULT_TRANSCRIPT_DIR = Path("2cleaned-data")
DEFAULT_CLIPPING_PLAN_DIR = Path("3clipped")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-clips")
DEFAULT_MAX_WORKERS = 3
DEFAULT_SKIP_EXISTING = True
SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm",
    ".MP4", ".MKV", ".MOV", ".AVI", ".M4V", ".WEBM",
}


# ==========================================
# 2. 动态加载单文件切片脚本
#    - 由于脚本文件名以数字开头，不适合常规 import
#    - 这里用 importlib 动态加载 4ffmpeg-clipping.py
# ==========================================
def load_ffmpeg_clipping_module():
    module_path = Path(__file__).parent / "4ffmpeg-clipping.py"
    spec = importlib.util.spec_from_file_location("ffmpeg_clipping", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================
# 3. 收集视频文件
#    - 递归查找视频目录下的所有支持格式
#    - 按路径排序，保证执行顺序稳定
# ==========================================
def collect_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"未找到视频目录: {video_dir}")

    video_files = [
        path for path in video_dir.rglob("*")
        if path.is_file() and path.suffix in SUPPORTED_VIDEO_SUFFIXES
    ]
    return sorted(video_files)


# ==========================================
# 4. 匹配可处理任务
#    - 视频文件名 stem + .json 作为匹配键
#    - 只有 transcript 和 clipping plan 都存在时才纳入任务
# ==========================================
def collect_clip_jobs(
    video_dir: Path,
    transcript_dir: Path,
    clipping_plan_dir: Path,
) -> tuple[list[dict], list[dict]]:
    jobs = []
    skipped = []

    for video_path in collect_video_files(video_dir):
        json_name = f"{video_path.stem}.json"
        transcript_path = transcript_dir / json_name
        clipping_plan_path = clipping_plan_dir / json_name

        missing_parts = []
        if not transcript_path.exists():
            missing_parts.append("transcript")
        if not clipping_plan_path.exists():
            missing_parts.append("clipping_plan")

        if missing_parts:
            skipped.append(
                {
                    "video_path": video_path,
                    "json_name": json_name,
                    "missing_parts": missing_parts,
                }
            )
            continue

        jobs.append(
            {
                "video_path": video_path,
                "transcript_path": transcript_path,
                "clipping_plan_path": clipping_plan_path,
            }
        )

    return jobs, skipped


def load_clip_count(clipping_plan_path: Path) -> int:
    with open(clipping_plan_path, "r", encoding="utf-8") as f:
        clipping_plan = json.load(f)

    clips = clipping_plan.get("clips", [])
    if not clips:
        return 0
    return len(clips)


def has_complete_existing_outputs(
    *,
    ffmpeg_clipping_module,
    video_path: Path,
    clipping_plan_path: Path,
    output_dir: Path,
) -> bool:
    clip_count = load_clip_count(clipping_plan_path)
    if clip_count == 0:
        return False

    for clip_number in range(1, clip_count + 1):
        clip_name = f"{video_path.stem}-clip{clip_number}"
        output_video_path = output_dir / f"{clip_name}.mp4"
        output_transcript_path = output_dir / f"{clip_name}.json"
        if not ffmpeg_clipping_module.is_valid_existing_clip(
            output_video_path,
            output_transcript_path,
        ):
            return False

    return True


# ==========================================
# 5. 批量执行主流程
#    - 对每个已匹配任务直接调用 clip_video_and_transcript
#    - 输出目录统一为 The Office BD clips
#    - 打印已处理、跳过和失败信息
# ==========================================
def run_batch_clipping(
    video_dir: Path = DEFAULT_VIDEO_DIR,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
    clipping_plan_dir: Path = DEFAULT_CLIPPING_PLAN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
) -> int:
    if max_workers < 1:
        raise ValueError("max_workers 必须大于等于 1")

    ffmpeg_clipping_module = load_ffmpeg_clipping_module()
    jobs, skipped = collect_clip_jobs(video_dir, transcript_dir, clipping_plan_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"视频目录: {video_dir}")
    print(f"Transcript 目录: {transcript_dir}")
    print(f"切片方案目录: {clipping_plan_dir}")
    print(f"输出目录: {output_dir}")
    print(f"视频级并发数: {max_workers}")
    print(f"跳过已完整视频: {skip_existing}")
    print(f"可处理视频数: {len(jobs)}")
    print(f"跳过视频数: {len(skipped)}")

    if skipped:
        print("\n跳过列表:")
        for item in skipped:
            print(
                f"- {item['video_path'].name} | 缺少: {', '.join(item['missing_parts'])}"
            )

    if not jobs:
        print("\n没有可处理的视频任务。")
        return 0

    failed_jobs = []

    print("\n开始批量切片...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_single_video_job,
                ffmpeg_clipping_module=ffmpeg_clipping_module,
                task_number=task_number,
                total_jobs=len(jobs),
                job=job,
                output_dir=output_dir,
                skip_existing=skip_existing,
            ): job
            for task_number, job in enumerate(jobs, start=1)
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "success": False,
                    "video_path": job["video_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                }

            if result["success"] and result.get("skipped_existing"):
                print(
                    f"[{result['task_number']}/{len(jobs)}] 跳过已完整: "
                    f"{result['video_path'].name}"
                )
            elif result["success"]:
                print(
                    f"[{result['task_number']}/{len(jobs)}] 完成处理: "
                    f"{result['video_path'].name}"
                )
            else:
                failed_jobs.append(result)
                print(
                    f"[{result.get('task_number', '?')}/{len(jobs)}] 处理失败: "
                    f"{result['video_path'].name} | {result['error']}"
                )

    print("\n批处理完成")
    skipped_existing_count = sum(
        1
        for future in future_to_job
        if future.done()
        and not future.exception()
        and future.result().get("skipped_existing")
    )
    print(f"成功: {len(jobs) - len(failed_jobs) - skipped_existing_count}")
    print(f"失败: {len(failed_jobs)}")
    print(f"缺输入跳过: {len(skipped)}")
    print(f"已完整跳过: {skipped_existing_count}")

    if failed_jobs:
        print("\n失败列表:")
        for item in failed_jobs:
            print(f"- {item['video_path'].name} | {item['error']}")
        return 1

    return 0


def run_single_video_job(
    *,
    ffmpeg_clipping_module,
    task_number: int,
    total_jobs: int,
    job: dict,
    output_dir: Path,
    skip_existing: bool,
) -> dict:
    print(f"[{task_number}/{total_jobs}] 开始处理: {job['video_path'].name}", flush=True)
    try:
        if skip_existing and has_complete_existing_outputs(
            ffmpeg_clipping_module=ffmpeg_clipping_module,
            video_path=job["video_path"],
            clipping_plan_path=job["clipping_plan_path"],
            output_dir=output_dir,
        ):
            return {
                "success": True,
                "skipped_existing": True,
                "task_number": task_number,
                "video_path": job["video_path"],
            }

        ffmpeg_clipping_module.clip_video_and_transcript(
            clipping_plan_path=str(job["clipping_plan_path"]),
            video_path=str(job["video_path"]),
            transcript_path=str(job["transcript_path"]),
            output_dir=str(output_dir),
            skip_existing=False,
        )
    except Exception as exc:
        return {
            "success": False,
            "skipped_existing": False,
            "task_number": task_number,
            "video_path": job["video_path"],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "success": True,
        "skipped_existing": False,
        "task_number": task_number,
        "video_path": job["video_path"],
    }


# ==========================================
# 6. 命令行入口
#    - 默认就是用户当前描述的目录结构
#    - 也允许手动覆盖目录参数
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量执行视频与 transcript 切片")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="视频目录，默认 /Volumes/Dingzhen/STT/The Office BD-original")
    parser.add_argument("--transcript-dir", default=str(DEFAULT_TRANSCRIPT_DIR), help="原始 transcript 目录，默认 2cleaned-data")
    parser.add_argument("--clipping-plan-dir", default=str(DEFAULT_CLIPPING_PLAN_DIR), help="切片方案目录，默认 3clipped")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录，默认 /Volumes/Dingzhen/STT/The Office BD-clips")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="视频级并发数，默认 3")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_EXISTING,
        help="跳过已完整的视频任务；用 --no-skip-existing 强制全部重跑",
    )
    args = parser.parse_args()

    exit_code = run_batch_clipping(
        video_dir=Path(args.video_dir),
        transcript_dir=Path(args.transcript_dir),
        clipping_plan_dir=Path(args.clipping_plan_dir),
        output_dir=Path(args.output_dir),
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
    )
    raise SystemExit(exit_code)
