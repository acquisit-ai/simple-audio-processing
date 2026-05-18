#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_OUTPUT_HEIGHT = 720
DEFAULT_VIDEO_BITRATE = "1700k"
DEFAULT_MAXRATE = "2500k"
DEFAULT_BUFSIZE = "4500k"
DEFAULT_GOP_SIZE = 48


def ensure_binary(binary_name: str) -> str:
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        raise RuntimeError(f"未找到 {binary_name}，请先安装 FFmpeg。")
    return binary_path


def summarize_ffmpeg_error(stderr: str, stdout: str) -> str:
    output = stderr.strip() or stdout.strip()
    if not output:
        return "未知错误"

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    important_markers = (
        "Cannot create compression session",
        "Error while opening encoder",
        "Conversion failed",
        "Unknown encoder",
        "Invalid argument",
    )
    important_lines = [
        line for line in lines
        if any(marker in line for marker in important_markers)
    ]
    summary_lines = important_lines or lines[-3:]
    return " | ".join(summary_lines)


def build_common_input_args(
    ffmpeg_path: str,
    source_video_path: Path,
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        "-map_chapters",
        "-1",
    ]


def build_common_output_args(
    output_video_path: Path,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> list[str]:
    return [
        "-g",
        str(gop_size),
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "hvc1",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]


def normalize_original_video(
    source_video_path: str,
    output_video_path: str,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> None:
    ffmpeg_path = ensure_binary("ffmpeg")
    source_video_file = Path(source_video_path)
    output_video_file = Path(output_video_path)

    if not source_video_file.exists():
        raise FileNotFoundError(f"未找到视频文件: {source_video_file}")

    output_video_file.parent.mkdir(parents=True, exist_ok=True)

    common_input_args = build_common_input_args(
        ffmpeg_path=ffmpeg_path,
        source_video_path=source_video_file,
    )
    common_output_args = build_common_output_args(
        output_video_path=output_video_file,
        gop_size=gop_size,
    )
    videotoolbox_command = [
        *common_input_args,
        "-vf",
        f"scale=-2:{output_height}:flags=lanczos",
        "-c:v",
        "hevc_videotoolbox",
        "-spatial_aq",
        "1",
        "-b:v",
        video_bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        *common_output_args,
    ]
    libx265_command = [
        *common_input_args,
        "-vf",
        f"scale=-2:{output_height}:flags=lanczos",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-b:v",
        video_bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        *common_output_args,
    ]

    result = subprocess.run(videotoolbox_command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        first_error_message = summarize_ffmpeg_error(result.stderr, result.stdout)
        print(
            "⚠️  VideoToolbox HEVC 编码失败，改用 libx265 CPU 编码。"
            f" 原因: {first_error_message}"
        )
        result = subprocess.run(libx265_command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        error_message = summarize_ffmpeg_error(result.stderr, result.stdout)
        raise RuntimeError(f"ffmpeg 标准化编码失败: {error_message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将单个原始视频压缩标准化为 720p HEVC MP4，并保留原始音频流。")
    parser.add_argument("source_video_path", help="原始视频路径")
    parser.add_argument("output_video_path", help="标准化后输出 MP4 路径")
    parser.add_argument("--height", type=int, default=DEFAULT_OUTPUT_HEIGHT, help="输出高度，默认 720。")
    parser.add_argument("--video-bitrate", default=DEFAULT_VIDEO_BITRATE, help="VideoToolbox 目标视频码率，默认 1700k。")
    parser.add_argument("--maxrate", default=DEFAULT_MAXRATE, help="视频码率上限，默认 2500k。")
    parser.add_argument("--bufsize", default=DEFAULT_BUFSIZE, help="码率控制 buffer size，默认 4500k。")
    parser.add_argument("--gop-size", type=int, default=DEFAULT_GOP_SIZE, help="关键帧间隔，默认 48。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_original_video(
        source_video_path=args.source_video_path,
        output_video_path=args.output_video_path,
        output_height=args.height,
        video_bitrate=args.video_bitrate,
        maxrate=args.maxrate,
        bufsize=args.bufsize,
        gop_size=args.gop_size,
    )
