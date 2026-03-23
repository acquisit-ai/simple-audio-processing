#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os

import assemblyai as aai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def build_assemblyai_config():
    """
    构建 AssemblyAI 转写配置。

    返回:
        tuple[aai.TranscriptionConfig, dict]: SDK 配置对象和可打印的参数字典
    """

    config = aai.TranscriptionConfig(
        # --- 核心模型与语言 ---
        speech_models=["universal-3-pro"],  # 主模型优先级列表；这里只使用 Universal-3 Pro。
        # speech_model=None,  # 单模型参数；这里留空，因为本脚本使用 speech_models 列表模式。
        language_detection=True,  # 是否开启自动语种检测；混合语种或不确定语种时建议开启。
        # language_code=None,  # 手动指定单一语言；开启自动检测时通常保持为 None。
        # language_codes=None,  # 多语言候选列表；仅在已知可能语言集合时使用。
        # language_confidence_threshold=None,  # 语言检测置信度阈值；低于阈值时可触发报错或回退逻辑。
        # language_detection_options=None,  # 自动语种检测的高级选项；例如期望语言和低置信度回退策略。
        # domain=None,  # 垂直领域模型；例如 medical-v1。通用美剧对白通常保持为 None。
        # speech_threshold=None,  # 音频最低语音占比阈值；过低时可拒绝低语音密度音频。

        # --- U3 Pro 提示增强 ---
        prompt="Transcribe this American TV episode accurately, preserve dialogue and slang, and keep punctuation natural.",  # 自然语言提示词；用于约束转写风格和术语理解。
        # keyterms_prompt=None,  # 关键词增强列表；用于提升特定词汇识别。注意它不能和 prompt 同时使用。
        # keyterms_prompt_options=None,  # 关键词增强高级选项；仅在启用 keyterms_prompt 时使用。
        temperature=0.1,  # 输出随机性；0 最稳定，数值越高生成越发散。
        remove_audio_tags="all",  # 删除 [music]、[laughter] 等音频标签；设为 "all" 表示移除所有此类标签。

        # --- 文本格式化 ---
        punctuate=True,  # 是否自动补全标点；字幕和对白整理通常应开启。
        format_text=True,  # 是否格式化数字、日期、缩写等文本形式。
        # disfluencies=False,  # 是否保留“嗯、啊、呃”等语气词；做干净字幕时通常关闭。
        # filter_profanity=False,  # 是否过滤脏话；美剧对白通常关闭以保留原始表达。
        # custom_spelling=None,  # 自定义拼写映射；用于修正专有名词、角色名、品牌名。
        # word_boost=[],  # 传统词汇增强列表；适用于非 prompt 场景的词汇识别加强。
        # boost_param=None,  # 传统词汇增强强度；与 word_boost 配合使用。

        # --- 摘要与内容分析 ---
        # summarization=False,  # 是否生成摘要；这里关闭，避免与 auto_chapters 冲突。
        # summary_model=None,  # 摘要模型风格；仅在 summarization=True 时使用。
        # summary_type=None,  # 摘要输出样式；仅在 summarization=True 时使用。
        # auto_chapters=True,  # 是否按内容转折自动划分章节。不要开,好像会出问题
        # sentiment_analysis=False,  # 是否做情感分析。
        # entity_detection=False,  # 是否提取人名、地名、组织名等实体。
        # iab_categories=False,  # 是否做 IAB 内容分类标签识别。
        # content_safety=False,  # 是否检测敏感、暴力、仇恨等内容。
        # content_safety_confidence=None,  # 内容安全检测阈值；仅在启用 content_safety 时使用。
        # auto_highlights=False,  # 是否自动提取重点短语和高亮词。

        # --- 说话人与声道 ---
        # speaker_labels=False,  # 是否开启说话人分离；开启后可得到 speaker A/B 等标签。
        # speakers_expected=None,  # 预期说话人数；仅在 speaker_labels=True 时使用。
        # speaker_options=None,  # 说话人分离高级选项；例如最少/最多说话人数。
        # dual_channel=False,  # 是否按双声道分别转写；适合左右声道是独立通话轨道的音频。
        # multichannel=False,  # 是否按多声道分别转写；适合多轨录音源。

        # --- 隐私脱敏 ---
        # redact_pii=False,  # 是否对文本中的个人隐私信息做脱敏。
        # redact_pii_audio=False,  # 是否生成蜂鸣/静音处理后的脱敏音频。
        # redact_pii_audio_quality=None,  # 脱敏音频质量选项；仅在 redact_pii_audio=True 时使用。
        # redact_pii_audio_options=None,  # 脱敏音频附加选项；例如用静音替代蜂鸣。
        # redact_pii_policies=None,  # 需要脱敏的 PII 类型列表；仅在 redact_pii=True 时设置。
        # redact_pii_sub=None,  # 脱敏替换策略；例如 mask 或 entity_name。

        # --- 裁剪与回调 ---
        # audio_start_from=None,  # 从音频的第几毫秒开始转写。
        # audio_end_at=None,  # 在音频的第几毫秒停止转写。
        # webhook_url=None,  # 回调地址；任务完成后向该 URL 发送通知。
        # webhook_auth_header_name=None,  # 回调鉴权请求头名称。
        # webhook_auth_header_value=None,  # 回调鉴权请求头值。

        # --- 进阶功能 ---
        # speech_understanding=None,  # LLM Gateway 能力入口；可用于说话人识别、翻译、定制格式化等高级功能。
    )

    config_dict = config.raw.model_dump()
    return config, config_dict


def run_assemblyai_with_local_file(local_audio_path, output_path=None):
    """
    使用本地音频文件调用 AssemblyAI 转写 API。

    Args:
        local_audio_path: 音频文件路径
        output_path: 输出文件路径，默认为当前目录下的 1transcript-raw/AssemblyAI Universal-3-Pro 文件夹
    """

    print("=" * 60)
    print("开始 AssemblyAI 语音识别任务")
    print("=" * 60)

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY environment variable is required")
    print("✓ API Key 已加载")

    aai.settings.api_key = api_key
    print("✓ AssemblyAI SDK 已初始化")

    config, config_dict = build_assemblyai_config()
    output_folder = "AssemblyAI Universal-3-Pro"

    print(f"\n音频文件信息:")
    print(f"  文件路径: {local_audio_path}")
    if os.path.exists(local_audio_path):
        file_size = os.path.getsize(local_audio_path) / (1024 * 1024)
        print(f"  文件大小: {file_size:.2f} MB")
        print("  ✓ 文件存在")
    else:
        print("  ✗ 文件不存在!")
        raise FileNotFoundError(f"音频文件不存在: {local_audio_path}")

    print(f"\n调用参数:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    print(f"\n正在调用 AssemblyAI API...")
    print("  请稍候，这可能需要几分钟时间...")

    transcriber = aai.Transcriber()
    with open(local_audio_path, "rb") as audio_file:
        transcript = transcriber.transcribe(audio_file, config=config)

    if transcript.error:
        raise RuntimeError(f"AssemblyAI 转写失败: {transcript.error}")

    print("  ✓ API 调用完成")

    if output_path is None:
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "1transcript-raw", output_folder)
        audio_filename = os.path.basename(local_audio_path)
        output_json_filename = os.path.splitext(audio_filename)[0] + ".json"
        output_json_path = os.path.join(output_dir, output_json_filename)
    else:
        output_json_path = output_path
        output_dir = os.path.dirname(output_json_path)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出文件: {output_json_path}")
    print("  ✓ 目录已创建/确认")

    merged_result = save_assemblyai_result(transcript, output_json_path)
    return merged_result, output_json_path


def _save_json(data, output_json_path):
    """将 JSON 数据保存到文件。"""
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def _fetch_sentences_payload(transcript):
    """获取句子级结果，并转换为可写入 JSON 的 Python 数据。"""
    sentences = transcript.get_sentences()
    return [_to_serializable(sentence) for sentence in sentences]


def merge_transcript_result(main_result, sentences_payload):
    """
    合并主 transcript JSON 和句子级结果：
    - 保留主 transcript JSON
    - 删除主 JSON 中的 words 字段
    - 将句子级结果合并到 sentences 字段
    """
    result = dict(main_result or {})
    result.pop("words", None)
    result["sentences"] = list(sentences_payload or [])
    return result


def _build_merged_result(transcript):
    """从 Transcript 对象构建合并后的输出结构。"""
    main_result = transcript.json_response or {}
    sentences_payload = _fetch_sentences_payload(transcript)
    return merge_transcript_result(main_result, sentences_payload)


def save_assemblyai_result(transcript, output_json_path):
    """
    保存合并后的 AssemblyAI 转写结果到 JSON 文件。

    Args:
        transcript: AssemblyAI Transcript 对象
        output_json_path: 输出文件路径
    """
    merged_result = _build_merged_result(transcript)
    _save_json(merged_result, output_json_path)
    print(f"✓ 合并后的 JSON 文件已保存: {output_json_path}")
    return merged_result


def _to_serializable(item):
    """将 SDK 返回对象转换为可写入 JSON 的 Python 数据。"""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return item


def main():
    # 直接使用固定的音频文件路径
    audio_path = "./resource/test.m4a"
    output_path = "1transcript-raw/test.json"

    try:
        result, _ = run_assemblyai_with_local_file(audio_path, output_path)

        print(f"\n结果摘要:")
        if isinstance(result, dict):
            text = result.get("text")
            if text:
                print(f"  转录文本长度: {len(text)} 字符")
                print(f"  前100个字符: {text[:100]}...")
            else:
                print(f"  结果键: {list(result.keys())}")

            summary = result.get("summary")
            if summary:
                print(f"\n摘要预览:")
                print(summary[:300] + ("..." if len(summary) > 300 else ""))

            sentences = result.get("sentences") or []
            if sentences:
                print(f"\n前3个句子级时间戳示例:")
                for sentence in sentences[:3]:
                    start = sentence.get("start")
                    end = sentence.get("end")
                    text_value = sentence.get("text")
                    print(f"  [{start}ms -> {end}ms] {text_value}")

        print(f"\n" + "=" * 60)
        print("✅ 任务完成!")
        print("=" * 60)

    except Exception as exc:
        print(f"\n" + "=" * 60)
        print("❌ 调用失败")
        print("=" * 60)
        print(f"错误信息: {exc}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
