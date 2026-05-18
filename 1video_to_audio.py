#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频音轨提取工具
使用 ffmpeg 直接提取视频原始音轨，不进行转码。
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


CODEC_EXTENSION_MAP = {
    "aac": "m4a",
    "ac3": "ac3",
    "alac": "m4a",
    "dts": "dts",
    "eac3": "eac3",
    "flac": "flac",
    "mp2": "mp2",
    "mp3": "mp3",
    "opus": "opus",
    "pcm_s16le": "wav",
    "pcm_s24le": "wav",
    "pcm_f32le": "wav",
    "truehd": "mka",
    "vorbis": "ogg",
    "wavpack": "wv",
}


def _ensure_binary(binary_name):
    """确保系统中存在所需的可执行文件。"""
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        raise RuntimeError(f"未找到 {binary_name}，请先安装 FFmpeg。")
    return binary_path


def _probe_audio_stream(video_path, stream_index=0):
    """读取音轨信息，用于选择默认输出扩展名。"""
    ffprobe_path = _ensure_binary("ffprobe")
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        f"a:{stream_index}",
        "-show_entries",
        "stream=index,codec_name",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 执行失败: {result.stderr.strip() or result.stdout.strip()}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe 输出无法解析为 JSON") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"文件中没有找到第 {stream_index + 1} 条音轨: {video_path}")

    return streams[0]


def _guess_output_extension(codec_name, output_extension=None):
    """根据音频编码选择更合适的输出扩展名。"""
    if output_extension:
        return output_extension.lstrip(".")
    return CODEC_EXTENSION_MAP.get(codec_name, "mka")


def _build_output_path(video_path, output_path=None, output_dir=None, output_extension=None, stream_index=0):
    """构建输出文件路径。"""
    if output_path is not None:
        return str(Path(output_path))

    video_file = Path(video_path)
    stream_info = _probe_audio_stream(video_path, stream_index=stream_index)
    extension = _guess_output_extension(stream_info.get("codec_name"), output_extension=output_extension)
    destination_dir = Path(output_dir) if output_dir is not None else video_file.parent
    return str(destination_dir / f"{video_file.stem}.{extension}")


def convert_video_to_audio(video_path, output_path=None, audio_format=None, start=None, end=None, stream_index=0):
    """
    直接提取视频原始音轨，不进行转码。

    Args:
        video_path (str): 输入视频文件路径
        output_path (str, optional): 输出音频文件路径，如果不指定则自动生成
        audio_format (str, optional): 输出文件扩展名提示，仅在未指定 output_path 时使用
        start (float, optional): 预留参数。原始音轨直提不支持裁剪
        end (float, optional): 预留参数。原始音轨直提不支持裁剪
        stream_index (int): 要提取的音轨序号，默认提取第一条音轨

    Returns:
        str: 输出音频文件路径
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if start is not None or end is not None:
        raise ValueError("直接提取原音轨不支持 start/end 裁剪参数；如需裁剪，需要重新编码。")

    ffmpeg_path = _ensure_binary("ffmpeg")
    output_path = _build_output_path(
        video_path,
        output_path=output_path,
        output_extension=audio_format,
        stream_index=stream_index,
    )

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始提取原音轨: {video_path}")
    print(f"音轨序号: {stream_index}")
    print(f"输出文件: {output_path}")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-map",
        f"0:a:{stream_index}",
        "-c",
        "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip() or "未知错误"
        print(f"❌ 提取失败: {error_message}")
        raise RuntimeError(f"ffmpeg 提取失败: {error_message}")

    print(f"✅ 提取完成: {output_path}")
    return str(output_path)


def batch_convert_videos_from_lists(input_list, output_list=None, audio_format=None, start=None, end=None, stream_index=0):
    """
    批量提取视频文件音轨（使用输入输出列表）。

    Args:
        input_list (list): 输入视频文件路径列表
        output_list (list, optional): 输出音频文件路径列表，如果不指定则自动生成
        audio_format (str, optional): 输出文件扩展名提示，仅在未指定 output_list 时使用
        start (float, optional): 预留参数。原始音轨直提不支持裁剪
        end (float, optional): 预留参数。原始音轨直提不支持裁剪
        stream_index (int): 要提取的音轨序号，默认提取第一条音轨

    Returns:
        list: 成功提取的音频文件路径列表
    """

    if not input_list:
        print("输入列表为空")
        return []

    if output_list is None:
        output_list = []
        for video_path in input_list:
            auto_output = _build_output_path(
                video_path,
                output_extension=audio_format,
                stream_index=stream_index,
            )
            output_list.append(auto_output)

    if len(input_list) != len(output_list):
        raise ValueError(f"输入列表和输出列表长度不一致: {len(input_list)} vs {len(output_list)}")

    print(f"准备提取 {len(input_list)} 个视频文件的音轨")

    extracted_files = []
    failed_files = []

    for i, (video_path, audio_path) in enumerate(zip(input_list, output_list), 1):
        print(f"\n[{i}/{len(input_list)}] 处理: {video_path}")
        try:
            result = convert_video_to_audio(
                video_path,
                audio_path,
                audio_format=audio_format,
                start=start,
                end=end,
                stream_index=stream_index,
            )
            extracted_files.append(result)
        except Exception as exc:
            print(f"❌ 提取失败 {video_path}: {exc}")
            failed_files.append(video_path)

    print(f"\n{'=' * 50}")
    print("批量提取完成:")
    print(f"✅ 成功: {len(extracted_files)} 个文件")
    if failed_files:
        print(f"❌ 失败: {len(failed_files)} 个文件")
        for failed in failed_files:
            print(f"   - {failed}")

    return extracted_files


def batch_convert_videos(input_dir, output_dir=None, audio_format=None, start=None, end=None, stream_index=0):
    """
    批量提取目录中的所有视频文件音轨。

    Args:
        input_dir (str): 输入视频目录
        output_dir (str, optional): 输出音频目录，如果不指定则使用输入目录
        audio_format (str, optional): 输出文件扩展名提示，仅在未指定 output_dir 时使用
        start (float, optional): 预留参数。原始音轨直提不支持裁剪
        end (float, optional): 预留参数。原始音轨直提不支持裁剪
        stream_index (int): 要提取的音轨序号，默认提取第一条音轨

    Returns:
        list: 成功提取的音频文件路径列表
    """

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"在目录 {input_dir} 中没有找到支持的视频文件")
        return []

    print(f"找到 {len(video_files)} 个视频文件")

    extracted_files = []
    failed_files = []

    for video_file in video_files:
        try:
            audio_file = _build_output_path(
                str(video_file),
                output_dir=output_path,
                output_extension=audio_format,
                stream_index=stream_index,
            )
            result = convert_video_to_audio(
                str(video_file),
                audio_file,
                audio_format=audio_format,
                start=start,
                end=end,
                stream_index=stream_index,
            )
            extracted_files.append(result)
        except Exception as exc:
            print(f"❌ 提取失败 {video_file.name}: {exc}")
            failed_files.append(str(video_file))

    print("\n提取完成:")
    print(f"✅ 成功: {len(extracted_files)} 个文件")
    if failed_files:
        print(f"❌ 失败: {len(failed_files)} 个文件")
        for failed in failed_files:
            print(f"   - {failed}")

    return extracted_files


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="直接提取视频原始音轨，不进行转码。")
    parser.add_argument("inputs", nargs="+", help="一个或多个输入视频文件路径")
    parser.add_argument("-o", "--output", help="单文件输出路径，仅单个输入时可用")
    parser.add_argument("--output-dir", help="批量输出目录")
    parser.add_argument(
        "--ext",
        help="自定义输出文件扩展名，仅影响自动生成的文件名，不会触发转码",
    )
    parser.add_argument(
        "--stream-index",
        type=int,
        default=0,
        help="要提取的音轨序号，默认 0 表示第一条音轨",
    )
    args = parser.parse_args()

    if args.output and len(args.inputs) != 1:
        parser.error("--output 只能和单个输入文件一起使用")

    if len(args.inputs) == 1:
        convert_video_to_audio(
            args.inputs[0],
            output_path=args.output,
            audio_format=args.ext,
            stream_index=args.stream_index,
        )
        return

    output_list = None
    if args.output_dir:
        output_list = [
            _build_output_path(
                input_path,
                output_dir=args.output_dir,
                output_extension=args.ext,
                stream_index=args.stream_index,
            )
            for input_path in args.inputs
        ]

    batch_convert_videos_from_lists(
        args.inputs,
        output_list=output_list,
        audio_format=args.ext,
        stream_index=args.stream_index,
    )


if __name__ == "__main__":
    main()
