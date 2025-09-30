#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频转音频工具
使用 moviepy 将视频文件转换为音频文件
"""

import os
from pathlib import Path
from moviepy import VideoFileClip


def convert_video_to_audio(video_path, output_path=None, audio_format="mp3", start=None, end=None):
    """
    将视频文件转换为音频文件

    Args:
        video_path (str): 输入视频文件路径
        output_path (str, optional): 输出音频文件路径，如果不指定则自动生成
        audio_format (str): 输出音频格式，默认为 "mp3"
        start (float, optional): 开始时间（秒），如果不指定则从头开始
        end (float, optional): 结束时间（秒），如果不指定则到结尾

    Returns:
        str: 输出音频文件路径
    """

    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 如果没有指定输出路径，自动生成
    if output_path is None:
        video_file = Path(video_path)
        output_path = video_file.parent / f"{video_file.stem}.{audio_format}"

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始转换: {video_path}")
    if start is not None or end is not None:
        print(f"裁剪时间: {start or 0}秒 - {end or '结尾'}秒")
    print(f"输出文件: {output_path}")

    try:
        # 加载视频文件
        video = VideoFileClip(video_path)

        # 提取音频
        audio = video.audio

        # 如果指定了时间范围，进行裁剪
        if start is not None or end is not None:
            audio = audio.subclipped(start, end)

        # 保存音频文件
        audio.write_audiofile(str(output_path), logger=None)

        # 清理资源
        audio.close()
        video.close()

        print(f"✅ 转换完成: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        raise


def batch_convert_videos(input_dir, output_dir=None, audio_format="mp3", start=None, end=None):
    """
    批量转换目录中的所有视频文件

    Args:
        input_dir (str): 输入视频目录
        output_dir (str, optional): 输出音频目录，如果不指定则使用输入目录
        audio_format (str): 输出音频格式，默认为 "mp3"
        start (float, optional): 开始时间（秒），如果不指定则从头开始
        end (float, optional): 结束时间（秒），如果不指定则到结尾

    Returns:
        list: 成功转换的音频文件路径列表
    """

    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 设置输出目录
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有视频文件
    video_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"在目录 {input_dir} 中没有找到支持的视频文件")
        return []

    print(f"找到 {len(video_files)} 个视频文件")

    converted_files = []
    failed_files = []

    for video_file in video_files:
        try:
            # 生成输出文件路径
            audio_file = output_path / f"{video_file.stem}.{audio_format}"

            # 转换视频
            result = convert_video_to_audio(str(video_file), str(audio_file), audio_format, start, end)
            converted_files.append(result)

        except Exception as e:
            print(f"❌ 转换失败 {video_file.name}: {e}")
            failed_files.append(str(video_file))

    # 输出结果摘要
    print(f"\n转换完成:")
    print(f"✅ 成功: {len(converted_files)} 个文件")
    if failed_files:
        print(f"❌ 失败: {len(failed_files)} 个文件")
        for failed in failed_files:
            print(f"   - {failed}")

    return converted_files


if __name__ == "__main__":
    # 示例用法：直接调用函数

    # 单文件转换示例    
    video_path = "./original-media/test-portrait.mkv"
    output_path = "./original-media/test-portrait.mp3"
    start = 0  # 开始时间（秒），None表示从头开始
    end = 180  # 结束时间（秒），None表示到结尾

    try:
        convert_video_to_audio(video_path, output_path)
    except Exception as e:
        print(f"转换失败: {e}")

    # 批量转换示例（注释掉，需要时取消注释）
    # try:
    #     batch_convert_videos("./original-media/", "./audios/", "mp3", 0, 60)
    # except Exception as e:
    #     print(f"批量转换失败: {e}")