#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主要流水线脚本
按顺序调用 1whisper.py, 2data-cleansing.py, 3llm.py 中的函数
从原始媒体/3min1.mp3 最终得到 3llm/3min1-cleaned-gemini.json
"""

import os
import sys
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

def main():
    """
    执行完整的音频转录和分析流水线
    """
    print("=" * 80)
    print("开始音频处理流水线")
    print("=" * 80)

    # 定义文件路径
    audio_file = "./原始媒体/3min1.mp3"
    whisper_output = "1transcript-raw/3min1.json"
    cleaned_output = "2cleaned-data/3min1-cleaned.json"
    final_output = "3llm/3min1-cleaned-gemini.json"

    # 检查输入文件是否存在
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    try:
        # 步骤1: Whisper语音识别
        print("\n步骤1: 使用 Whisper 进行语音识别...")
        print("-" * 50)
        _, output_json_path = whisper_module.run_whisperx_with_local_file(audio_file, whisper_output)

        # run_whisperx_with_local_file 函数已经自动保存了结果
        print(f"✓ Whisper 输出已保存: {output_json_path}")

        # 步骤2: 数据清理
        print("\n步骤2: 清理和结构化数据...")
        print("-" * 50)
        data_cleansing_module.process_whisperx_to_cleaned(whisper_output, cleaned_output)
        print(f"✓ 清理后的数据已保存: {cleaned_output}")

        # 步骤3: LLM分析
        print("\n步骤3: 使用 LLM 进行文本分析...")
        print("-" * 50)

        # 3a: 提取句子
        print("3a: 提取句子...")
        simplified_data = llm_module.extract_sentences_only(cleaned_output)

        # 3b: LLM处理
        print("\n3b: 使用 Gemini API 处理...")
        llm_module.process_sentences_with_llm(simplified_data, final_output)

        print("\n" + "=" * 80)
        print("✅ 流水线执行完成!")
        print("=" * 80)
        print(f"最终输出文件: {final_output}")

        # 显示处理链
        print("\n处理链:")
        print(f"  {audio_file}")
        print(f"  ↓ (Whisper)")
        print(f"  {whisper_output}")
        print(f"  ↓ (数据清理)")
        print(f"  {cleaned_output}")
        print(f"  ↓ (LLM分析)")
        print(f"  {final_output}")

    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ 流水线执行失败")
        print(f"=" * 80)
        print(f"错误信息: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()