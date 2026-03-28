#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path


# ==========================================
# 1. 基础工具
#    - 检查 ffmpeg 是否存在
#    - 读取/写入 JSON
#    - 毫秒转 ffmpeg 可用秒数
# ==========================================
def ensure_binary(binary_name: str) -> str:
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        raise RuntimeError(f"未找到 {binary_name}，请先安装 FFmpeg。")
    return binary_path


def load_json(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"未找到文件: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ms_to_ffmpeg_seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


# ==========================================
# 2. transcript 切片工具
#    - 按切片方案中的 start_index / end_index 提取句子
#    - 保持原始 JSON 顶层结构
#    - 将句子 index 从 0 开始重新编号
#    - 如果存在 tokens，也把每句内部 token 的 index 从 0 开始重排
# ==========================================
def build_sentence_lookup(transcript_data: dict) -> tuple[list[dict], dict[int, int]]:
    ordered_sentences = transcript_data.get("sentences", [])
    sentence_position_map = {
        sentence["index"]: position
        for position, sentence in enumerate(ordered_sentences)
    }
    return ordered_sentences, sentence_position_map


def slice_transcript_for_clip(
    transcript_data: dict,
    start_index: int,
    end_index: int,
) -> dict:
    ordered_sentences, sentence_position_map = build_sentence_lookup(transcript_data)

    if start_index not in sentence_position_map:
        raise ValueError(f"transcript 中不存在 start_index: {start_index}")
    if end_index not in sentence_position_map:
        raise ValueError(f"transcript 中不存在 end_index: {end_index}")
    if start_index > end_index:
        raise ValueError(f"切片 index 非法: {start_index} > {end_index}")

    start_position = sentence_position_map[start_index]
    end_position = sentence_position_map[end_index]
    selected_sentences = ordered_sentences[start_position:end_position + 1]

    clip_transcript = copy.deepcopy(transcript_data)
    clip_sentences = []

    for new_sentence_index, sentence in enumerate(selected_sentences):
        sentence_copy = copy.deepcopy(sentence)
        sentence_copy["index"] = new_sentence_index

        if "tokens" in sentence_copy and isinstance(sentence_copy["tokens"], list):
            for new_token_index, token in enumerate(sentence_copy["tokens"]):
                if isinstance(token, dict):
                    token["index"] = new_token_index

        clip_sentences.append(sentence_copy)

    clip_transcript["sentences"] = clip_sentences

    if "total_sentences" in clip_transcript:
        clip_transcript["total_sentences"] = len(clip_sentences)

    if "total_tokens" in clip_transcript:
        clip_transcript["total_tokens"] = sum(
            len(sentence.get("tokens", []))
            for sentence in clip_sentences
        )

    return clip_transcript


# ==========================================
# 3. 视频切片工具
#    - 默认优先使用 buffered_start_time / buffered_end_time
#    - 若不存在，则退回 start_time / end_time
#    - 使用 ffmpeg 直切并复制原始码流，不做转码
#    - 切点会受关键帧影响，但当前有 buffer，可接受这类误差
# ==========================================
def resolve_clip_times(clip_plan: dict) -> tuple[int, int]:
    start_time = clip_plan.get("buffered_start_time", clip_plan["start_time"])
    end_time = clip_plan.get("buffered_end_time", clip_plan["end_time"])

    if start_time < 0:
        start_time = 0
    if end_time <= start_time:
        raise ValueError(
            f"切片时间非法: start_time={start_time}, end_time={end_time}"
        )

    return start_time, end_time


def cut_video_clip(
    ffmpeg_path: str,
    source_video_path: Path,
    output_video_path: Path,
    start_time_ms: int,
    end_time_ms: int,
) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = end_time_ms - start_time_ms

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        ms_to_ffmpeg_seconds(start_time_ms),
        "-i",
        str(source_video_path),
        "-t",
        ms_to_ffmpeg_seconds(duration_ms),
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        str(output_video_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip() or "未知错误"
        raise RuntimeError(f"ffmpeg 切片失败: {error_message}")


# ==========================================
# 4. 主切片函数
#    - 输入切片方案 JSON、原视频、原始 transcript、输出目录
#    - 按 clip1、clip2... 递增命名
#    - 生成同名的视频片段和 transcript 片段
# ==========================================
def clip_video_and_transcript(
    clipping_plan_path: str,
    video_path: str,
    transcript_path: str,
    output_dir: str,
) -> None:
    clipping_plan_file = Path(clipping_plan_path)
    source_video_file = Path(video_path)
    transcript_file = Path(transcript_path)
    output_directory = Path(output_dir)

    if not source_video_file.exists():
        raise FileNotFoundError(f"未找到视频文件: {source_video_file}")

    ffmpeg_path = ensure_binary("ffmpeg")
    clipping_plan = load_json(clipping_plan_file)
    transcript_data = load_json(transcript_file)
    clips = clipping_plan.get("clips", [])

    if not clips:
        raise ValueError(f"切片方案中没有 clips: {clipping_plan_file}")

    output_directory.mkdir(parents=True, exist_ok=True)
    base_name = source_video_file.stem
    video_suffix = source_video_file.suffix

    print(f"开始处理视频切片: {source_video_file}")
    print(f"切片数量: {len(clips)}")
    print(f"输出目录: {output_directory}")

    for clip_number, clip_plan in enumerate(clips, start=1):
        clip_name = f"{base_name}-clip{clip_number}"
        output_video_path = output_directory / f"{clip_name}{video_suffix}"
        output_transcript_path = output_directory / f"{clip_name}.json"

        start_time_ms, end_time_ms = resolve_clip_times(clip_plan)
        print(
            f"[{clip_number}/{len(clips)}] 切片: {clip_name} | "
            f"视频时间 {start_time_ms}-{end_time_ms} ms | "
            f"句子 {clip_plan['start_index']}-{clip_plan['end_index']}"
        )

        cut_video_clip(
            ffmpeg_path=ffmpeg_path,
            source_video_path=source_video_file,
            output_video_path=output_video_path,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )

        clip_transcript = slice_transcript_for_clip(
            transcript_data=transcript_data,
            start_index=clip_plan["start_index"],
            end_index=clip_plan["end_index"],
        )
        save_json(clip_transcript, output_transcript_path)

    print("✅ 所有视频切片与 transcript 切片已完成。")


# ==========================================
# 5. 命令行入口
#    - 依次传入：切片方案 JSON、视频路径、原始 transcript JSON、输出目录
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据切片方案执行视频与 transcript 切片")
    parser.add_argument("clipping_plan_path", help="切片方案 JSON 文件路径，例如 3clipped/xxx.json")
    parser.add_argument("video_path", help="需要切片的视频路径")
    parser.add_argument("transcript_path", help="原始 transcript JSON 文件路径，例如 2cleaned-data/xxx.json")
    parser.add_argument("output_dir", help="输出文件夹")
    args = parser.parse_args()

    clip_video_and_transcript(
        clipping_plan_path=args.clipping_plan_path,
        video_path=args.video_path,
        transcript_path=args.transcript_path,
        output_dir=args.output_dir,
    )
