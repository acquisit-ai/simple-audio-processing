import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig

# ==========================================
# 1. 定义结构化输出数据模型
#    - LLMClip: 大模型输出的中间结果，包含 index 和 time，便于自查
#    - Clip: 在 ClipBoundary 基础上补齐精确时间和缓冲时间
#    - LLMClipResponse: 约束模型必须返回 {"clips": [...]} 结构
# ==========================================
class LLMClip(BaseModel):
    clip_id: int = Field(description="Sequential clip identifier.")
    title: str = Field(description="Short user-facing clip title in Simplified Chinese.")
    description: str = Field(description="Concise user-facing clip description in Simplified Chinese.")
    start_index: int = Field(description="Index of the first sentence in the clip.")
    end_index: int = Field(description="Index of the last sentence in the clip.")
    start_time: int = Field(description="Original start timestamp in milliseconds for the first sentence, used only for self-checking.")
    end_time: int = Field(description="Original end timestamp in milliseconds for the last sentence, used only for self-checking.")
    reasoning: str = Field(description="Brief explanation in Simplified Chinese of why these boundaries were chosen.")

class Clip(BaseModel):
    clip_id: int = Field(description="Sequential clip identifier.")
    title: str = Field(description="Short user-facing clip title in Simplified Chinese.")
    description: str = Field(description="Concise user-facing clip description in Simplified Chinese.")
    start_index: int = Field(description="Index of the first sentence in the clip.")
    end_index: int = Field(description="Index of the last sentence in the clip.")
    start_time: int = Field(description="Exact start timestamp in milliseconds.")
    end_time: int = Field(description="Exact end timestamp in milliseconds.")
    buffered_start_time: int = Field(description="Buffered clip start timestamp in milliseconds for actual video cutting.")
    buffered_end_time: int = Field(description="Buffered clip end timestamp in milliseconds for actual video cutting.")
    reasoning: str = Field(description="Brief explanation in Simplified Chinese of why these boundaries were chosen.")

class LLMClipResponse(BaseModel):
    clips: List[LLMClip] = Field(description="Array of clip objects.")

# LLMClip / Clip / LLMClipResponse 中文对照
# clip_id：切片的唯一序号
# title：切片中文标题
# description：切片中文描述
# start_index：切片起始台词的 index
# end_index：切片结束台词的 index
# start_time：切片第一句台词的 start 原始毫秒数，仅供模型自查
# end_time：切片最后一句台词的 end 原始毫秒数，仅供模型自查
# buffered_start_time：带缓冲的切片起始毫秒数，用于实际视频裁切
# buffered_end_time：带缓冲的切片结束毫秒数，用于实际视频裁切
# reasoning：边界取舍原因阐述
# clips：切片数组

# ==========================================
# 2. 初始化运行环境与 Gemini
#    - 从 .env 读取 Google Cloud / Vertex AI 配置
#    - 初始化 google-genai Vertex AI client
#    - 用 response_schema 锁定模型输出 schema
#    - 定义后处理 buffer 参数，供视频裁切阶段复用
# ==========================================
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_GEMINI_THINKING_LEVEL = "high"
GEMINI_MODEL = os.getenv("GEMINI_CLIPPING_MODEL", DEFAULT_GEMINI_MODEL)
GEMINI_THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    DEFAULT_GEMINI_THINKING_LEVEL,
).lower()
VERTEX_LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("GOOGLE_VERTEX_LOCATION")
    or DEFAULT_VERTEX_LOCATION
)

if GEMINI_THINKING_LEVEL not in {"low", "high"}:
    raise ValueError(
        "GEMINI_THINKING_LEVEL must be 'low' or 'high' for Gemini 3 Pro models"
    )


def get_gcloud_project() -> str | None:
    """Read the active gcloud project when GOOGLE_CLOUD_PROJECT is not set."""

    try:
        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    project = completed.stdout.strip()
    if completed.returncode != 0 or not project or project == "(unset)":
        return None
    return project


VERTEX_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCLOUD_PROJECT")
    or get_gcloud_project()
)

if not VERTEX_PROJECT:
    raise ValueError(
        "Google Cloud project is required. Set GOOGLE_CLOUD_PROJECT, "
        "or run: gcloud config set project <PROJECT_ID>"
    )

# 明确使用 Vertex AI。认证走 Application Default Credentials:
# gcloud auth application-default login
client = genai.Client(
    vertexai=True,
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

CLIP_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "clips": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "clip_id": {
                        "type": "INTEGER",
                        "description": "Sequential clip identifier, starting at 1.",
                    },
                    "title": {
                        "type": "STRING",
                        "description": "Short user-facing clip title in Simplified Chinese.",
                    },
                    "description": {
                        "type": "STRING",
                        "description": "Concise user-facing clip description in Simplified Chinese.",
                    },
                    "start_index": {
                        "type": "INTEGER",
                        "description": "Index of the first sentence in the clip.",
                    },
                    "end_index": {
                        "type": "INTEGER",
                        "description": "Index of the last sentence in the clip.",
                    },
                    "start_time": {
                        "type": "INTEGER",
                        "description": "Original start timestamp in milliseconds for the first sentence, used only for self-checking.",
                    },
                    "end_time": {
                        "type": "INTEGER",
                        "description": "Original end timestamp in milliseconds for the last sentence, used only for self-checking.",
                    },
                    "reasoning": {
                        "type": "STRING",
                        "description": "Brief explanation in Simplified Chinese of why these boundaries were chosen.",
                    },
                },
                "required": [
                    "clip_id",
                    "title",
                    "description",
                    "start_index",
                    "end_index",
                    "start_time",
                    "end_time",
                    "reasoning",
                ],
                "propertyOrdering": [
                    "clip_id",
                    "title",
                    "description",
                    "start_index",
                    "end_index",
                    "start_time",
                    "end_time",
                    "reasoning",
                ],
            },
        },
    },
    "required": ["clips"],
    "propertyOrdering": ["clips"],
}


def invoke_structured_gemini(prompt: str) -> LLMClipResponse:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            candidate_count=1,
            response_mime_type="application/json",
            response_schema=CLIP_RESPONSE_SCHEMA,
            thinking_config=ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL,
            ),
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    return LLMClipResponse.model_validate_json(response.text)

# 以下参数用于“精确时间 -> 带缓冲时间”的后处理
# 思路是 gap-based + clamp + 左小右大 + 安全边界
MIN_LEFT_BUFFER_MS = 250
MIN_RIGHT_BUFFER_MS = 400
LEFT_BUFFER_RATIO = 0.55
RIGHT_BUFFER_RATIO = 0.7
SAFETY_MARGIN_MS = 80
DEFAULT_EDGE_LEFT_BUFFER_MS = 900
DEFAULT_EDGE_RIGHT_BUFFER_MS = 1800

# ==========================================
# 3. 定义主提示词与反思提示词
#    - SYSTEM_PROMPT: 负责告诉模型“如何切”
#    - REFLECTION_PROMPT: 负责第二轮复核，尽量修掉明显边界问题
# ==========================================
SYSTEM_PROMPT = """
You are a semantic clipping engine for video preprocessing. Your only task is to read transcript JSON with millisecond timestamps and divide it into rough clips of about 1 to 3 minutes (about 60,000 to 180,000 ms) for English learning.

# Core goals
1. Semantic closure first: each clip should feel as self-contained as possible and should not feel obviously context-dependent when watched alone.
2. Limited overlap is allowed if it helps preserve closure, but avoid excessive repetition between adjacent clips.
3. Linguistic continuity first: do not cut inside a complete sentence, a tight question-answer exchange, or a strongly connected expression chain.
4. Natural openings: avoid starting a clip with a pronoun, conjunction, or fragment that clearly depends on previous context.
5. Duration is a secondary constraint: if duration and semantic integrity conflict, prefer semantic integrity and then stay as close as possible to 3 minutes.
6. Leave-gap cleanup: you may drop segments with very low English-learning value, such as stretches with no useful expressions, phrases, or collocations. Large timeline gaps are acceptable when justified.

# Segmentation cues (soft guidance)
- Split according to content naturally, keep overall clip count reasonable, and preferably keep the total number of clips within about 12 when possible.
- Prefer cuts around scene changes, topic changes, longer pauses, the end of a question-answer turn, or after an action/joke payoff is completed.
- Do not split setup from payoff, question from answer, or suggestion from response across two clips unless necessary.
- By default, try to cover most dialogue with learning value; only drop segments that are clearly low-value, repetitive small talk, or too context-dependent to stand alone.

# English-learning preference (soft guidance)
- Without harming semantic closure, prefer segments containing frequent spoken English, natural question-answering, collocations, phrasal verbs, idioms, emotional expressions, and workplace or everyday expressions.
- If two boundary choices are equally natural, prefer the one that is better for shadowing, retelling, or standalone study.

# Output requirements
Output clip boundaries together with corresponding timestamps:
- title: short user-facing title in Simplified Chinese
- description: concise user-facing description in Simplified Chinese
- start_index: index of the first sentence in the clip
- end_index: index of the last sentence in the clip
- start_time: original start timestamp in milliseconds for the first sentence
- end_time: original end timestamp in milliseconds for the last sentence
- reasoning: short explanation of the boundary choice in Simplified Chinese

All title, description, and reasoning values must be written in Simplified Chinese.
"""

# SYSTEM_PROMPT 中文对照
# 你是一个负责视频流预处理的“语义切片引擎”。你的唯一任务是读取带有毫秒级时间戳的 JSON 台词文本，
# 将其划分为多个 1 到 3 分钟（约 60,000 - 180,000 毫秒）的粗略区块，专供英语学习使用。
#
# 核心目标
# 1. 语义闭环优先：每个切片应尽量自洽，单独观看时不明显悬空。
# 2. 适度重叠以保闭环：相邻切片之间允许有部分时间段重叠，但严禁重叠过多导致学习重复。
# 3. 语言连贯优先：不要在完整句子、紧凑问答或强依赖上下文的表达链条中间切断。
# 4. 起点自然：避免让切片以明显承接上文的代词、连词或残句开头。
# 5. 时长是次级约束：若无法同时满足时长和语义完整，优先保证语义完整，再尽量靠近 3 分钟。
# 6. 留白清洗：丢弃语言学习价值极低的片段，例如没有有价值的词组搭配、短语或表达。允许大段时间轴留白。
#
# 切分线索（软建议）
# - 根据内容合理切分，控制长短，总数量最好控制在 12 个及以内。
# - 优先在场景切换、话题切换、较长停顿、问答回合结束、动作/笑点完成后切分。
# - 不要把 setup 和 payoff、提问和回答、建议和回应拆到两个切片里。
# - 默认尽量覆盖大部分有学习价值的对白；在片段明显低价值、重复寒暄或强依赖前文且难以独立成立时舍弃。
#
# 英语学习导向（软建议）
# - 在不破坏语义闭环的前提下，可优先保留包含高频口语、自然问答、固定搭配、短语动词、习语、情绪表达、职场/日常场景表达的片段。
# - 如果两个切法都同样自然，可优先选择更适合跟读、复述、单独学习的方案。
#
# 输出要求
# 输出切片边界与对应时间：
# - title：面向用户展示的简短中文标题
# - description：面向用户展示的简短中文描述
# - start_index：切片第一句台词的 index
# - end_index：切片最后一句台词的 index
# - start_time：切片第一句台词对应的 start 原始毫秒数
# - end_time：切片最后一句台词对应的 end 原始毫秒数
# - reasoning：中文边界取舍原因阐述
#
# title、description、reasoning 必须全部使用简体中文。

REFLECTION_PROMPT = """
Act as a strict quality reviewer and re-check your previous clipping plan.

Review checklist:
1. Does any clip start with an obviously dangling pronoun or context-dependent opening?
2. Does any clip duration clearly deviate too far from the 60,000-180,000 ms target range?
3. Did you incorrectly cut inside a tight question-answer exchange or another strongly linked dialogue unit?
4. Were you overly conservative and therefore skipped too much dialogue that still has learning value?
5. Is there unnecessary heavy overlap between adjacent clips?
6. Is there a better boundary choice that would preserve more complete everyday expressions, question-answer chains, or collocations for learning?
7. Are title, description, and reasoning all written in Simplified Chinese and aligned with the final clip boundaries?

If you find issues, fix them and output the full corrected result object in the same format: {"clips": [...]}.
If the plan is already solid, output the previous full result object unchanged, still in the format: {"clips": [...]}.
"""

# REFLECTION_PROMPT 中文对照
# 请作为严格的质检员，重新审视你刚才输出的切片方案。
#
# 审查重点：
# 1. 是否有明显的“首句代词悬空”导致语境断裂？
# 2. 是否有切片的时间跨度严重偏离 60,000 - 180,000 毫秒的约束？
# 3. 是否错误切断了紧凑的问答回合？
# 4. 是否因为过度保守而遗漏了大量本可学习的对白？
# 5. 是否有不必要的大量重叠？
# 6. 是否有更适合学习的边界选择，可以保留更完整的常用表达、问答链条或固定搭配？
# 7. title、description、reasoning 是否全部使用简体中文，并且与最终切片边界一致？
#
# 如果发现缺陷，请修正并输出优化后的完整结果对象，格式必须仍然是 {"clips": [...]}。
# 如果逻辑已经闭环，无需优化，请直接原样输出之前的完整结果对象，格式必须仍然是 {"clips": [...]}。


# ==========================================
# 4. 输入预处理
#    - 删除 transcript 中每句的 tokens 字段
#    - 只把句级信息交给模型，减少 token 消耗和无关噪音
# ==========================================
def remove_tokens_from_transcript(transcript_data: dict) -> dict:
    cleaned_data = dict(transcript_data)
    cleaned_sentences = []

    for sentence in transcript_data.get("sentences", []):
        cleaned_sentence = {
            key: value
            for key, value in sentence.items()
            if key != "tokens"
        }
        cleaned_sentences.append(cleaned_sentence)

    cleaned_data["sentences"] = cleaned_sentences
    return cleaned_data


# ==========================================
# 5. 通用数值工具
#    - clamp: 将数值限制在给定区间内
#    - get_left/right_buffer_cap: 按相邻停顿长度动态给出 buffer 上限
#      停顿越大，允许扩出的缓冲越大，最长可到数秒
# ==========================================
def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def get_left_buffer_cap(left_gap: int) -> int:
    if left_gap >= 6000:
        return 3200
    if left_gap >= 3500:
        return 2200
    if left_gap >= 1800:
        return 1400
    return 900


def get_right_buffer_cap(right_gap: int) -> int:
    if right_gap >= 7000:
        return 4200
    if right_gap >= 4000:
        return 3000
    if right_gap >= 2000:
        return 1800
    return 1200


# ==========================================
# 6. 计算带缓冲的切片时间
#    - 输入为精确 start/end 和它们在 transcript 中的位置
#    - 根据前后相邻句子的时间间隙，自适应向外扩展
#    - 左侧扩得更保守，右侧扩得更积极
#    - 用 SAFETY_MARGIN_MS 避免明显吃进相邻对白
#    - 第一段 / 最后一段由于缺少一侧邻居，使用默认边缘 buffer
# ==========================================
def compute_buffered_times(
    exact_start_time: int,
    exact_end_time: int,
    start_position: int,
    end_position: int,
    ordered_sentences: List[dict],
) -> tuple[int, int]:
    buffered_start_time = exact_start_time
    buffered_end_time = exact_end_time

    # 左侧：参考前一句结束时间，决定向前扩多少
    if start_position > 0:
        previous_sentence_end = ordered_sentences[start_position - 1]["end"]
        left_gap = max(0, exact_start_time - previous_sentence_end)
        left_buffer_cap = get_left_buffer_cap(left_gap)
        proposed_left_buffer = clamp(
            int(left_gap * LEFT_BUFFER_RATIO),
            MIN_LEFT_BUFFER_MS,
            left_buffer_cap,
        )
        safe_left_buffer = max(
            0,
            exact_start_time - (previous_sentence_end + SAFETY_MARGIN_MS),
        )
        buffered_start_time = exact_start_time - min(
            proposed_left_buffer,
            safe_left_buffer,
        )
    else:
        # 第一段没有“前一句”，只能做保守默认扩展，并确保不小于 0
        buffered_start_time = max(0, exact_start_time - DEFAULT_EDGE_LEFT_BUFFER_MS)

    # 右侧：参考后一句开始时间，决定向后扩多少
    if end_position < len(ordered_sentences) - 1:
        next_sentence_start = ordered_sentences[end_position + 1]["start"]
        right_gap = max(0, next_sentence_start - exact_end_time)
        right_buffer_cap = get_right_buffer_cap(right_gap)
        proposed_right_buffer = clamp(
            int(right_gap * RIGHT_BUFFER_RATIO),
            MIN_RIGHT_BUFFER_MS,
            right_buffer_cap,
        )
        safe_right_buffer = max(
            0,
            (next_sentence_start - SAFETY_MARGIN_MS) - exact_end_time,
        )
        buffered_end_time = exact_end_time + min(
            proposed_right_buffer,
            safe_right_buffer,
        )
    else:
        # 最后一段没有“后一句”，先给一个默认尾部留白
        # 如果以后接实际视频裁切，再结合真实视频总时长做最终截断
        buffered_end_time = exact_end_time + DEFAULT_EDGE_RIGHT_BUFFER_MS

    return buffered_start_time, buffered_end_time


# ==========================================
# 7. 根据 transcript 回填时间字段
#    - LLM 会返回 start_index / end_index / start_time / end_time
#    - 但最终以原始 transcript 为准，完全忽略 LLM 给出的时间值
#    - 这里根据原始 transcript 回填：
#      1) 精确 start_time / end_time
#      2) 带缓冲的 buffered_start_time / buffered_end_time
# ==========================================
def add_timestamps_to_clips(clips: List[LLMClip], transcript_data: dict) -> List[Clip]:
    ordered_sentences = transcript_data.get("sentences", [])
    sentence_map = {
        sentence["index"]: sentence
        for sentence in ordered_sentences
    }
    sentence_position_map = {
        sentence["index"]: position
        for position, sentence in enumerate(ordered_sentences)
    }

    completed_clips = []
    for clip in clips:
        # 先按 index 找到切片的首句和尾句
        start_sentence = sentence_map[clip.start_index]
        end_sentence = sentence_map[clip.end_index]
        start_position = sentence_position_map[clip.start_index]
        end_position = sentence_position_map[clip.end_index]

        # 再基于精确句边界，计算实际裁视频更自然的缓冲边界
        buffered_start_time, buffered_end_time = compute_buffered_times(
            exact_start_time=start_sentence["start"],
            exact_end_time=end_sentence["end"],
            start_position=start_position,
            end_position=end_position,
            ordered_sentences=ordered_sentences,
        )
        completed_clips.append(
            Clip(
                clip_id=clip.clip_id,
                title=clip.title,
                description=clip.description,
                start_index=clip.start_index,
                end_index=clip.end_index,
                start_time=start_sentence["start"],
                end_time=end_sentence["end"],
                buffered_start_time=buffered_start_time,
                buffered_end_time=buffered_end_time,
                reasoning=clip.reasoning,
            )
        )

    return completed_clips


# ==========================================
# 8. 对模型返回的切片边界做结果校验
#    - 校验 clips 不为空
#    - 校验 clip_id 连续
#    - 校验 start/end index 存在且顺序合法
#    - 校验切片整体顺序不倒退、区间不重复
#    - 时长偏离 1~3 分钟时只给 warning，不直接失败
# ==========================================
def validate_clip_boundaries(clips: List[LLMClip], transcript_data: dict) -> None:
    sentences = transcript_data.get("sentences", [])
    if not sentences:
        raise ValueError("transcript_data.sentences 为空，无法进行切片")

    if not clips:
        raise ValueError("模型返回空 clips，请调整提示词或输入内容后重试")

    sentence_map = {
        sentence["index"]: sentence
        for sentence in sentences
    }

    previous_start_index = None
    previous_end_index = None
    seen_ranges = set()

    for expected_clip_id, clip in enumerate(clips, start=1):
        if clip.clip_id != expected_clip_id:
            raise ValueError(
                f"clip_id 不连续：期望 {expected_clip_id}，实际 {clip.clip_id}"
            )

        if clip.start_index not in sentence_map:
            raise ValueError(f"start_index 不存在于 transcript 中: {clip.start_index}")

        if clip.end_index not in sentence_map:
            raise ValueError(f"end_index 不存在于 transcript 中: {clip.end_index}")

        if clip.start_index > clip.end_index:
            raise ValueError(
                f"切片边界非法：start_index({clip.start_index}) > end_index({clip.end_index})"
            )

        expected_start_time = sentence_map[clip.start_index]["start"]
        expected_end_time = sentence_map[clip.end_index]["end"]
        if clip.start_time != expected_start_time:
            print(
                f"⚠️  切片 {clip.clip_id} 的 start_time={clip.start_time} 与 transcript 回填值 {expected_start_time} 不一致，最终将以 transcript 为准"
            )
        if clip.end_time != expected_end_time:
            print(
                f"⚠️  切片 {clip.clip_id} 的 end_time={clip.end_time} 与 transcript 回填值 {expected_end_time} 不一致，最终将以 transcript 为准"
            )

        if previous_start_index is not None and clip.start_index < previous_start_index:
            raise ValueError("切片顺序异常：start_index 未按时间顺序递增")

        if previous_end_index is not None and clip.end_index < previous_end_index:
            raise ValueError("切片顺序异常：end_index 未按时间顺序递增")

        clip_range = (clip.start_index, clip.end_index)
        if clip_range in seen_ranges:
            raise ValueError(f"检测到重复切片区间：{clip_range}")
        seen_ranges.add(clip_range)

        start_sentence = sentence_map[clip.start_index]
        end_sentence = sentence_map[clip.end_index]
        duration_ms = end_sentence["end"] - start_sentence["start"]
        if duration_ms < 60000 or duration_ms > 180000:
            print(
                f"⚠️  切片 {clip.clip_id} 时长为 {duration_ms}ms，偏离建议区间 60000-180000ms"
            )

        previous_start_index = clip.start_index
        previous_end_index = clip.end_index

# ==========================================
# 9. 主执行管线
#    步骤概览：
#    1) 读取原始 transcript JSON
#    2) 删除 tokens，只保留句级信息给模型
#    3) 第一轮生成切片边界与辅助时间
#    4) 第二轮反思优化边界与辅助时间
#    5) 校验边界合法性，并检查模型时间与 transcript 是否一致
#    6) 回填精确时间与缓冲时间
#    7) 输出到默认 3clipped/ 或用户指定路径
# ==========================================
def process_transcript_pipeline(input_filepath: str, output_filepath: str | None = None):
    input_path = Path(input_filepath)
    
    # 步骤 1：读取原始 transcript JSON
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # 步骤 2：删除 tokens，避免把词级信息和额外噪音送给大模型
    cleaned_transcript_data = remove_tokens_from_transcript(transcript_data)
    transcript_text = json.dumps(cleaned_transcript_data, ensure_ascii=False)
    
    # 步骤 3：第一轮切片
    # 让模型同时输出边界与辅助时间，便于其自查时长和闭环性
    print("🚀 [1/3] 执行初次切片生成...")
    initial_prompt = (
        "Please perform semantic clipping on the following transcript:\n"
        f"{transcript_text}"
    )

    initial_response_obj = invoke_structured_gemini(initial_prompt)
    
    # 步骤 4：第二轮反思优化
    # 把第一次结果回灌给模型，让模型自查边界质量
    print("🔍 [2/3] 执行自我反思与优化...")
    ai_memory_str = initial_response_obj.model_dump_json(indent=2, ensure_ascii=False)
    
    reflection_prompt = (
        "Previous clipping plan:\n"
        f"{ai_memory_str}\n\n"
        f"{REFLECTION_PROMPT}"
    )

    # 再次调用，得到最终的边界结果
    final_response_obj = invoke_structured_gemini(reflection_prompt)

    # 步骤 5：先校验边界是否合法，再做时间回填
    # 注意：即便模型给出的 start_time / end_time 有偏差，最终仍以 transcript 为准
    validate_clip_boundaries(final_response_obj.clips, transcript_data)

    # 步骤 6：根据原始 transcript 回填精确时间和带缓冲时间
    final_clips = add_timestamps_to_clips(final_response_obj.clips, transcript_data)
    
    # 步骤 7：确定输出路径并写入结果
    print("💾 [3/3] 正在落盘保存...")
    if output_filepath is None:
        output_path = Path("3clipped") / input_path.name
    else:
        output_path = Path(output_filepath)

    # 确保目标目录存在，再输出最终 JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"clips": [clip.model_dump() for clip in final_clips]},
            f,
            indent=4,
            ensure_ascii=False,
        )
        
    print(f"✅ 任务完成！文件已保存至: {output_path}")

# ==========================================
# 10. 命令行入口
#     - 必填 input：transcript JSON 路径
#     - 可选 output：输出 JSON 路径
#     - 若不传 output，则默认输出到 3clipped/同名文件
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 transcript 生成视频语义切片")
    parser.add_argument("input", help="输入 transcript JSON 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 JSON 文件路径，默认写入 3clipped/ 下并保持同名",
    )
    args = parser.parse_args()

    process_transcript_pipeline(args.input, args.output)
