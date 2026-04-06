#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
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
) -> int:
    ffmpeg_clipping_module = load_ffmpeg_clipping_module()
    jobs, skipped = collect_clip_jobs(video_dir, transcript_dir, clipping_plan_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"视频目录: {video_dir}")
    print(f"Transcript 目录: {transcript_dir}")
    print(f"切片方案目录: {clipping_plan_dir}")
    print(f"输出目录: {output_dir}")
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
    for task_number, job in enumerate(jobs, start=1):
        print(f"[{task_number}/{len(jobs)}] 开始处理: {job['video_path'].name}")
        try:
            ffmpeg_clipping_module.clip_video_and_transcript(
                clipping_plan_path=str(job["clipping_plan_path"]),
                video_path=str(job["video_path"]),
                transcript_path=str(job["transcript_path"]),
                output_dir=str(output_dir),
            )
            print(f"[{task_number}/{len(jobs)}] 完成处理: {job['video_path'].name}")
        except Exception as exc:
            failed_jobs.append(
                {
                    "video_path": job["video_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[{task_number}/{len(jobs)}] 处理失败: {job['video_path'].name} | "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n批处理完成")
    print(f"成功: {len(jobs) - len(failed_jobs)}")
    print(f"失败: {len(failed_jobs)}")
    print(f"跳过: {len(skipped)}")

    if failed_jobs:
        print("\n失败列表:")
        for item in failed_jobs:
            print(f"- {item['video_path'].name} | {item['error']}")
        return 1

    return 0


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
    args = parser.parse_args()

    exit_code = run_batch_clipping(
        video_dir=Path(args.video_dir),
        transcript_dir=Path(args.transcript_dir),
        clipping_plan_dir=Path(args.clipping_plan_dir),
        output_dir=Path(args.output_dir),
    )
    raise SystemExit(exit_code)
