#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /Users/evan/Code/whisper/venv/bin/python whisper.py
import replicate
import json
import os
from dotenv import load_dotenv
from models_config import MODELS

# Load environment variables
load_dotenv()

def run_whisperx_with_local_file(local_audio_path, output_path=None):
    """
    使用本地音频文件 + 硬编码 API token 来调用 Replicate 上的 whisper 模型。
    可通过 USE_MODEL 参数切换不同的模型。

    Args:
        local_audio_path: 音频文件路径
        output_path: 输出文件路径，默认为当前目录下的1transcript-raw文件夹
    """

    print("=" * 60)
    print("开始 Whisper 语音识别任务")
    print("=" * 60)

    # Get API token from environment variable
    REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN environment variable is required")
    print("✓ API Token 已加载")

    # 创建 client
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    print("✓ Replicate 客户端已创建")

    # 硬编码参数：选择使用的模型 (1-5)
    USE_MODEL = 1  # 可选: 1=WhisperX, 2=OpenAI Whisper, 3=Whisper Diarization, 4=Whisper Timestamped, 5=Whisper Diarization Advanced

    if USE_MODEL not in MODELS:
        raise ValueError(f"无效的 USE_MODEL 值: {USE_MODEL}。请使用 {list(MODELS.keys())}。")

    # 从配置中获取模型信息
    model_config = MODELS[USE_MODEL]
    model_version = model_config["version"]
    audio_param_name = model_config["audio_param"]
    output_folder = model_config["name"]

    print(f"\n选择的模型:")
    print(f"  模型 ID: {USE_MODEL}")
    print(f"  模型名称: {model_config['name']}")
    print(f"  模型版本: {model_version}")
    print(f"  输出文件夹: {output_folder}")

    # 构造输入参数
    input_dict = model_config["default_params"].copy()

    print(f"\n音频文件信息:")
    print(f"  文件路径: {local_audio_path}")
    if os.path.exists(local_audio_path):
        file_size = os.path.getsize(local_audio_path) / (1024 * 1024)  # 转换为 MB
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  ✓ 文件存在")
    else:
        print(f"  ✗ 文件不存在!")
        raise FileNotFoundError(f"音频文件不存在: {local_audio_path}")

    print(f"\n模型参数:")
    for key, value in input_dict.items():
        print(f"  {key}: {value}")

    # 打开本地音频文件作为 file object，Replicate Python 客户端支持本地文件输入
    print(f"\n正在调用 Replicate API...")
    print(f"  请稍候，这可能需要几分钟时间...")

    with open(local_audio_path, "rb") as f:
        input_dict[audio_param_name] = f
        # 调用模型（在 with 块内调用，确保文件未关闭）
        output = replicate_client.run(model_version, input=input_dict)

    print(f"  ✓ API 调用完成")

    # 确定输出文件路径
    if output_path is None:
        # 默认保存到当前目录的1transcript-raw文件夹中
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "1transcript-raw", output_folder)
        audio_filename = os.path.basename(local_audio_path)
        output_json_filename = os.path.splitext(audio_filename)[0] + ".json"
        output_json_path = os.path.join(output_dir, output_json_filename)
    else:
        # 使用指定的输出文件路径
        output_json_path = output_path
        output_dir = os.path.dirname(output_json_path)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出文件: {output_json_path}")
    print(f"  ✓ 目录已创建/确认")

    # 保存结果到文件
    save_whisper_result(output, output_json_path)

    return output, output_json_path


def save_whisper_result(result, output_json_path):
    """
    保存Whisper结果到JSON文件

    Args:
        result: Whisper API返回的结果
        output_json_path: 输出文件路径
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果为JSON文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ JSON 文件已保存: {output_json_path}")


def main():
    # 直接使用固定的音频文件路径
    audio_path = "./original-media/3min1.mp3"
    output_path = "1transcript-raw/3min1.json"

    try:
        result, _ = run_whisperx_with_local_file(audio_path, output_path)

        # 显示结果摘要
        print(f"\n结果摘要:")
        if isinstance(result, dict):
            if "transcription" in result:
                text = result["transcription"]
                print(f"  转录文本长度: {len(text)} 字符")
                print(f"  前100个字符: {text[:100]}...")
            elif "text" in result:
                text = result["text"]
                print(f"  转录文本长度: {len(text)} 字符")
                print(f"  前100个字符: {text[:100]}...")
            else:
                print(f"  结果键: {list(result.keys())}")
        elif isinstance(result, str):
            print(f"  转录文本长度: {len(result)} 字符")
            print(f"  前100个字符: {result[:100]}...")

        print(f"\n" + "=" * 60)
        print(f"✅ 任务完成!")
        print(f"=" * 60)

    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"❌ 调用失败")
        print(f"=" * 60)
        print(f"错误信息: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()