#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-original")
DEFAULT_TARGET_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-normalized")
DEFAULT_MAX_WORKERS = 3
SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm",
    ".MP4", ".MKV", ".MOV", ".AVI", ".M4V", ".WEBM",
}


def load_normalize_module():
    module_path = Path(__file__).parent / "0normalize-original-video.py"
    spec = importlib.util.spec_from_file_location("normalize_original_video", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix in SUPPORTED_VIDEO_SUFFIXES


def collect_video_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"未找到源视频目录: {source_dir}")

    video_files = [
        path for path in source_dir.rglob("*")
        if is_supported_video_file(path)
    ]
    return sorted(video_files, key=natural_sort_key)


def build_target_path(source_dir: Path, target_dir: Path, source_path: Path) -> Path:
    relative_path = source_path.relative_to(source_dir)
    return target_dir / relative_path.with_suffix(".mp4")


def is_valid_existing_video(video_path: Path) -> bool:
    return video_path.exists() and video_path.stat().st_size > 0


def run_single_normalize_job(
    *,
    normalize_module,
    task_number: int,
    total_jobs: int,
    source_path: Path,
    target_path: Path,
    overwrite: bool,
    output_height: int,
    video_bitrate: str,
    maxrate: str,
    bufsize: str,
    audio_bitrate: str,
    audio_sample_rate: int,
    audio_channels: int,
    gop_size: int,
) -> dict:
    if not overwrite and is_valid_existing_video(target_path):
        return {
            "status": "skipped",
            "task_number": task_number,
            "source_path": source_path,
            "target_path": target_path,
        }

    print(f"[{task_number}/{total_jobs}] 开始标准化: {source_path.name}", flush=True)

    try:
        normalize_module.normalize_original_video(
            source_video_path=str(source_path),
            output_video_path=str(target_path),
            output_height=output_height,
            video_bitrate=video_bitrate,
            maxrate=maxrate,
            bufsize=bufsize,
            audio_bitrate=audio_bitrate,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
            gop_size=gop_size,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "task_number": task_number,
            "source_path": source_path,
            "target_path": target_path,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "processed",
        "task_number": task_number,
        "source_path": source_path,
        "target_path": target_path,
    }


def run_batch_normalize(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    target_dir: Path = DEFAULT_TARGET_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    overwrite: bool = False,
    output_height: int = 720,
    video_bitrate: str = "1500k",
    maxrate: str = "2200k",
    bufsize: str = "4000k",
    audio_bitrate: str = "256k",
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
    gop_size: int = 48,
) -> int:
    if max_workers < 1:
        raise ValueError("max_workers 必须大于等于 1")

    normalize_module = load_normalize_module()
    source_files = collect_video_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"源视频目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"视频级并发数: {max_workers}")
    print(f"覆盖已存在输出: {overwrite}")
    print(f"源视频数: {len(source_files)}")

    if not source_files:
        print("\n没有找到可处理的视频文件。")
        return 0

    processed = 0
    skipped = 0
    failed_jobs = []

    print("\n开始批量标准化编码...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {}
        for task_number, source_path in enumerate(source_files, start=1):
            target_path = build_target_path(source_dir, target_dir, source_path)
            future = executor.submit(
                run_single_normalize_job,
                normalize_module=normalize_module,
                task_number=task_number,
                total_jobs=len(source_files),
                source_path=source_path,
                target_path=target_path,
                overwrite=overwrite,
                output_height=output_height,
                video_bitrate=video_bitrate,
                maxrate=maxrate,
                bufsize=bufsize,
                audio_bitrate=audio_bitrate,
                audio_sample_rate=audio_sample_rate,
                audio_channels=audio_channels,
                gop_size=gop_size,
            )
            future_to_source[future] = source_path

        for future in as_completed(future_to_source):
            source_path = future_to_source[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "source_path": source_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            status = result["status"]
            if status == "processed":
                processed += 1
                print(
                    f"[{result['task_number']}/{len(source_files)}] 标准化完成: "
                    f"{result['target_path']}"
                )
            elif status == "skipped":
                skipped += 1
                print(
                    f"[{result['task_number']}/{len(source_files)}] 跳过已存在: "
                    f"{result['target_path']}"
                )
            else:
                failed_jobs.append(result)
                print(
                    f"[{result.get('task_number', '?')}/{len(source_files)}] 标准化失败: "
                    f"{result['source_path'].name} | {result['error']}"
                )

    print("\n批处理完成")
    print(f"处理成功: {processed}")
    print(f"跳过: {skipped}")
    print(f"失败: {len(failed_jobs)}")

    if failed_jobs:
        print("\n失败列表:")
        for item in failed_jobs:
            print(f"- {item['source_path']} | {item['error']}")
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量压缩标准化原始视频为 720p HEVC + AAC MP4。")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="源视频目录，默认 /Volumes/Dingzhen/STT/The Office BD-original")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="目标目录，默认 /Volumes/Dingzhen/STT/The Office BD-normalized")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="视频级并发数，默认 3")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的标准化视频")
    parser.add_argument("--height", type=int, default=720, help="输出高度，默认 720。")
    parser.add_argument("--video-bitrate", default="1500k", help="VideoToolbox 目标视频码率，默认 1500k。")
    parser.add_argument("--maxrate", default="2200k", help="视频码率上限，默认 2200k。")
    parser.add_argument("--bufsize", default="4000k", help="码率控制 buffer size，默认 4000k。")
    parser.add_argument("--audio-bitrate", default="256k", help="AAC 音频码率，默认 256k。")
    parser.add_argument("--audio-sample-rate", type=int, default=48000, help="音频采样率，默认 48000。")
    parser.add_argument("--audio-channels", type=int, default=2, help="音频声道数，默认 2。")
    parser.add_argument("--gop-size", type=int, default=48, help="关键帧间隔，默认 48。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_batch_normalize(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        output_height=args.height,
        video_bitrate=args.video_bitrate,
        maxrate=args.maxrate,
        bufsize=args.bufsize,
        audio_bitrate=args.audio_bitrate,
        audio_sample_rate=args.audio_sample_rate,
        audio_channels=args.audio_channels,
        gop_size=args.gop_size,
    )
    raise SystemExit(exit_code)
