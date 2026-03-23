#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主要流水线脚本
按顺序调用 1whisper.py, 2data-cleansing.py, 3llm.py 中的函数
从original-media/3min1.mp3 最终得到 3llm/3min1-cleaned-gemini.json

使用方式:
1. 命令行调用:
   python3 main.py <audio_file>
   python3 main.py ./original-media/001.mp3

2. 模块导入调用:
   from main import main, batch_process
   main("./original-media/001.mp3")
   batch_process(["./original-media/001.mp3", "./original-media/002.mp3"])
"""

import os
import sys
import argparse
from pathlib import Path

# 导入各模块的主要函数
import importlib.util

# 动态导入以数字开头的模块
def import_module_by_path(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入各个模块
whisper_module = import_module_by_path("1whisper.py", "whisper_module")
data_cleansing_module = import_module_by_path("2data-cleansing.py", "data_cleansing_module")
llm_module = import_module_by_path("3llm.py", "llm_module")

def main(audio_file="./original-media/3min1.mp3"):
    """
    执行完整的音频转录和分析流水线

    Args:
        audio_file (str): 输入音频文件路径

    Returns:
        dict: 包含所有输出文件路径的字典，如果失败则返回 None
    """
    print("=" * 80)
    print("开始音频处理流水线")
    print("=" * 80)

    # 基于音频文件名自动生成输出路径
    audio_filename = Path(audio_file).stem
    whisper_output = f"1transcript-raw/{audio_filename}.json"
    cleaned_output = f"2cleaned-data/{audio_filename}-cleaned.json"
    llm_output = f"3llm/{audio_filename}-cleaned-gemini.json"

    # 检查输入文件是否存在
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return None

    try:
        # 步骤1: Whisper语音识别
        print("\n步骤1: 使用 Whisper 进行语音识别...")
        print("-" * 50)
        whisper_module.run_whisperx_with_local_file(audio_file, whisper_output)

        # 步骤2: 数据清理
        print("\n步骤2: 清理和结构化数据...")
        print("-" * 50)
        data_cleansing_module.process_assemblyai_to_cleaned(whisper_output, cleaned_output)
        print(f"✓ 清理后的数据已保存: {cleaned_output}")

        # 步骤3: LLM分析
        print("\n步骤3: 使用 LLM 进行文本分析...")
        print("-" * 50)

        # 3a: 提取句子
        print("3a: 提取句子...")
        simplified_data, original_data = llm_module.extract_sentences_only(cleaned_output)

        # 3b: LLM处理
        print("\n3b: 使用 Gemini API 处理...")
        llm_module.process_sentences_with_llm(simplified_data, original_data, llm_output)

        print("\n" + "=" * 80)
        print("✅ 流水线执行完成!")
        print("=" * 80)
        print(f"最终输出文件: {llm_output}")

        # 显示处理链
        print("\n处理链:")
        print(f"  {audio_file}")
        print(f"  ↓ (Whisper)")
        print(f"  {whisper_output}")
        print(f"  ↓ (数据清理)")
        print(f"  {cleaned_output}")
        print(f"  ↓ (LLM分析)")
        print(f"  {llm_output}")

        # 返回输出文件路径
        return {
            "audio_file": audio_file,
            "whisper_output": whisper_output,
            "cleaned_output": cleaned_output,
            "llm_output": llm_output
        }

    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ 流水线执行失败")
        print(f"=" * 80)
        print(f"错误信息: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        return None


def batch_process(audio_files):
    """
    批量处理多个音频文件

    Args:
        audio_files (list): 音频文件路径列表

    Returns:
        dict: 包含成功和失败文件列表的字典
    """
    if not audio_files:
        print("❌ 音频文件列表为空")
        return {"success": [], "failed": []}

    print("=" * 80)
    print(f"批量处理 {len(audio_files)} 个音频文件")
    print("=" * 80)

    success_results = []
    failed_files = []

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'#' * 80}")
        print(f"处理文件 [{i}/{len(audio_files)}]: {audio_file}")
        print(f"{'#' * 80}")

        result = main(audio_file)
        if result:
            success_results.append(result)
        else:
            failed_files.append(audio_file)

    # 输出总结
    print("\n" + "=" * 80)
    print("批量处理完成")
    print("=" * 80)
    print(f"✅ 成功: {len(success_results)} 个文件")
    if failed_files:
        print(f"❌ 失败: {len(failed_files)} 个文件")
        for failed in failed_files:
            print(f"   - {failed}")

    return {
        "success": success_results,
        "failed": failed_files
    }

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="音频处理流水线 - 语音识别、数据清理、LLM分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python3 main.py ./original-media/001.mp3

  # 处理多个文件
  python3 main.py ./original-media/001.mp3 ./original-media/002.mp3

  # 使用默认文件
  python3 main.py
        """
    )

    parser.add_argument(
        "audio_files",
        nargs="*",
        help="输入音频文件路径（支持多个文件）"
    )

    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="批量处理模式（显示批量处理摘要）"
    )

    args = parser.parse_args()

    # 如果没有提供文件，使用默认文件
    if not args.audio_files:
        audio_file = "./original-media/001.mp3"
        print(f"未指定音频文件，使用默认文件: {audio_file}\n")
        main(audio_file)
    # 如果只有一个文件且没有指定批量模式
    elif len(args.audio_files) == 1 and not args.batch:
        main(args.audio_files[0])
    # 批量处理多个文件
    else:
        batch_process(args.audio_files)
