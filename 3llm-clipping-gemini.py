import os
import json
import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import (
    Content,
    CreateCachedContentConfig,
    GenerateContentConfig,
    HttpOptions,
    Part,
    ThinkingConfig,
)

# ==========================================
# 1. 定义结构化输出数据模型
#    - LLMClip: 大模型输出的中间结果，包含 index 和 time，便于自查
#    - Clip: 在 ClipBoundary 基础上补齐精确时间和缓冲时间
#    - LLMClipResponse: 约束模型必须返回 {"clips": [...]} 结构
# ==========================================
class EngagementScores(BaseModel):
    drama: int = Field(ge=1, le=10, description="Conflict, tension, awkwardness, or emotional intensity score from 1 to 10.")
    humor: int = Field(ge=1, le=10, description="Comedic or funny value score from 1 to 10.")
    payoff: int = Field(ge=1, le=10, description="Strength of the ending, joke, reversal, or satisfying beat from 1 to 10.")
    standalone: int = Field(ge=1, le=10, description="How well the clip works without surrounding episode context from 1 to 10.")
    reasoning: str = Field(description="Brief Simplified Chinese explanation for the engagement scores.")


class LLMClip(BaseModel):
    clip_id: int = Field(description="Sequential clip identifier.")
    title: str = Field(description="Short user-facing clip title in Simplified Chinese.")
    description: str = Field(description="Concise user-facing clip description in Simplified Chinese.")
    engagement: EngagementScores = Field(description="Audience appeal scores for the clip.")
    start_index: int = Field(description="Index of the first sentence in the clip.")
    end_index: int = Field(description="Index of the last sentence in the clip.")
    start_time: int = Field(description="Original start timestamp in milliseconds for the first sentence, used only for self-checking.")
    end_time: int = Field(description="Original end timestamp in milliseconds for the last sentence, used only for self-checking.")
    reasoning: str = Field(description="Brief explanation in Simplified Chinese of why these boundaries were chosen.")

class Clip(BaseModel):
    clip_id: int = Field(description="Sequential clip identifier.")
    title: str = Field(description="Short user-facing clip title in Simplified Chinese.")
    description: str = Field(description="Concise user-facing clip description in Simplified Chinese.")
    engagement: EngagementScores = Field(description="Audience appeal scores for the clip.")
    start_index: int = Field(description="Index of the first sentence in the clip.")
    end_index: int = Field(description="Index of the last sentence in the clip.")
    start_time: int = Field(description="Exact start timestamp in milliseconds.")
    end_time: int = Field(description="Exact end timestamp in milliseconds.")
    buffered_start_time: int = Field(description="Buffered clip start timestamp in milliseconds for actual video cutting.")
    buffered_end_time: int = Field(description="Buffered clip end timestamp in milliseconds for actual video cutting.")
    duration_time: int = Field(description="Buffered clip duration in milliseconds.")
    reasoning: str = Field(description="Brief explanation in Simplified Chinese of why these boundaries were chosen.")

class LLMClipResponse(BaseModel):
    clips: List[LLMClip] = Field(description="Array of clip objects.")

# LLMClip / Clip / LLMClipResponse 中文对照
# clip_id：切片的唯一序号
# title：切片中文标题
# description：切片中文描述
# engagement：切片吸引力维度评分
# engagement.drama：冲突、尴尬或情绪张力评分，1-10
# engagement.humor：喜剧效果评分，1-10
# engagement.payoff：笑点、反转、情绪落点或结尾满足感评分，1-10
# engagement.standalone：脱离上下文后独立观看成立程度评分，1-10
# engagement.reasoning：吸引力评分的中文解释
# start_index：切片起始台词的 index
# end_index：切片结束台词的 index
# start_time：切片第一句台词的 start 原始毫秒数，仅供模型自查
# end_time：切片最后一句台词的 end 原始毫秒数，仅供模型自查
# buffered_start_time：带缓冲的切片起始毫秒数，用于实际视频裁切
# buffered_end_time：带缓冲的切片结束毫秒数，用于实际视频裁切
# duration_time：带缓冲的切片时长毫秒数，等于 buffered_end_time - buffered_start_time
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
DEFAULT_CACHE_TTL_SECONDS = 20 * 60
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

if GEMINI_THINKING_LEVEL not in {"low", "medium", "high"}:
    raise ValueError(
        "GEMINI_THINKING_LEVEL must be one of: low, medium, high"
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
                    "engagement": {
                        "type": "OBJECT",
                        "description": "Audience appeal scores for this clip.",
                        "properties": {
                            "drama": {
                                "type": "INTEGER",
                                "description": "Conflict, tension, awkwardness, or emotional intensity score from 1 to 10.",
                            },
                            "humor": {
                                "type": "INTEGER",
                                "description": "Comedic or funny value score from 1 to 10.",
                            },
                            "payoff": {
                                "type": "INTEGER",
                                "description": "Strength of the ending, joke, reversal, or satisfying beat from 1 to 10.",
                            },
                            "standalone": {
                                "type": "INTEGER",
                                "description": "How well the clip works without surrounding episode context from 1 to 10.",
                            },
                            "reasoning": {
                                "type": "STRING",
                                "description": "Brief Simplified Chinese explanation for the engagement scores.",
                            },
                        },
                        "required": [
                            "drama",
                            "humor",
                            "payoff",
                            "standalone",
                            "reasoning",
                        ],
                        "propertyOrdering": [
                            "drama",
                            "humor",
                            "payoff",
                            "standalone",
                            "reasoning",
                        ],
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
                    "engagement",
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
                    "engagement",
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


def print_usage_metadata(prefix: str, usage_metadata: object | None) -> None:
    if usage_metadata is None:
        return
    print(f"{prefix} usage_metadata: {usage_metadata}", flush=True)


def create_video_context_cache(
    video_gcs_uri: str,
    video_mime_type: str = "video/mp4",
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> str:
    if not video_gcs_uri.startswith("gs://"):
        raise ValueError("--video-gcs-uri 必须是 gs:// 开头的 Cloud Storage 对象地址")

    print(f"🎞️  创建视频 explicit context cache: {video_gcs_uri}", flush=True)
    cache_display_name = build_cache_display_name(video_gcs_uri)
    cache = client.caches.create(
        model=GEMINI_MODEL,
        config=CreateCachedContentConfig(
            display_name=cache_display_name,
            ttl=f"{cache_ttl_seconds}s",
            system_instruction=SYSTEM_PROMPT,
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_uri(
                            file_uri=video_gcs_uri,
                            mime_type=video_mime_type,
                        )
                    ],
                )
            ],
        ),
    )
    print(f"🎞️  cache display_name: {cache_display_name}", flush=True)
    print(f"🎞️  cache name: {cache.name}", flush=True)
    print_usage_metadata("🎞️  cache", cache.usage_metadata)
    return cache.name


def build_cache_display_name(video_gcs_uri: str) -> str:
    digest = hashlib.sha1(video_gcs_uri.encode("utf-8")).hexdigest()[:12]
    return f"llm-clipping-{digest}"


def delete_context_cache(cache_name: str) -> None:
    print(f"🧹 删除 explicit context cache: {cache_name}", flush=True)
    client.caches.delete(name=cache_name)


def invoke_structured_gemini(
    prompt: str,
    cached_content_name: str | None = None,
) -> LLMClipResponse:
    config_kwargs = {
        "temperature": 0,
        "candidate_count": 1,
        "response_mime_type": "application/json",
        "response_schema": CLIP_RESPONSE_SCHEMA,
        "thinking_config": ThinkingConfig(
            thinking_level=GEMINI_THINKING_LEVEL,
        ),
    }
    if cached_content_name:
        config_kwargs["cached_content"] = cached_content_name
    else:
        config_kwargs["system_instruction"] = SYSTEM_PROMPT

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=GenerateContentConfig(**config_kwargs),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    print_usage_metadata("🤖 Gemini", response.usage_metadata)
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
你是一个用于视频预处理的语义切片引擎。你的唯一任务是读取带毫秒级时间戳的 transcript JSON，并把它切成多个约 1 到 3 分钟（约 60,000 到 180,000 毫秒）的粗切片，供英语学习使用。

# 核心目标
1. 语义闭环优先：每个切片应尽量自洽，单独观看时不明显悬空。
2. 允许少量重叠来保留语义闭环，但不要让相邻切片重复过多。
3. 语言连贯优先：不要切断完整句子、紧密问答、强依赖上下文的表达链条。
4. 起点自然：避免让切片以明显承接上文的代词、连词或残句开头。
5. 时长是次级约束：如果时长和语义完整冲突，优先保证语义完整，再尽量靠近 3 分钟。
6. 可清理低价值留白：可以丢弃英语学习价值很低的片段，例如没有有价值表达、短语或搭配的内容。允许合理的大段时间轴留白。

# 切分线索
- 按内容自然切分，控制总数量；如果可行，尽量把 clips 总数控制在 12 个以内。
- 优先在场景变化、话题变化、较长停顿、问答回合结束、动作或笑点完成后切分。
- 不要把铺垫和笑点、问题和回答、建议和回应拆到两个切片里，除非没有更好的选择。
- 默认尽量覆盖大多数有学习价值的对白；只丢弃明显低价值、重复寒暄或过度依赖上下文且难以独立成立的片段。

# 英语学习偏好
- 在不破坏语义闭环的前提下，优先保留包含高频口语、自然问答、固定搭配、短语动词、习语、情绪表达、职场或日常表达的片段。
- 如果两个边界选择同样自然，优先选择更适合跟读、复述或单独学习的方案。

# 可选视频上下文
- 如果提供了视频，只把视频作为判断语义闭环、场景变化、情绪转折、标题和描述的辅助上下文。
- transcript 的句子 index 是唯一权威来源。
- transcript 的时间戳是唯一权威来源。
- 视频绝不能作为 start_time 或 end_time 的来源。
- 不要从视频中发明新的时间戳或句子 index。
- 每个 clip 的 start_index 和 end_index 必须来自输入 transcript JSON 中实际存在的句子 index。
- 每个 clip 的 start_time 必须严格等于 transcript.sentences[start_index].start。
- 每个 clip 的 end_time 必须严格等于 transcript.sentences[end_index].end。

# 输出要求
输出切片边界和辅助信息：
- title：面向用户展示的简短中文标题
- description：面向用户展示的简短中文描述
- engagement：吸引力维度评分对象
  - drama：1 到 10 的整数；冲突、尴尬或情绪张力
  - humor：1 到 10 的整数；喜剧或好笑程度
  - payoff：1 到 10 的整数；结尾、笑点、反转或情绪落点的强度
  - standalone：1 到 10 的整数；脱离前后剧情后独立观看是否成立
  - reasoning：用简体中文简短解释这些吸引力评分
- start_index：切片第一句台词的 index
- end_index：切片最后一句台词的 index
- start_time：切片第一句台词在 transcript 中的原始 start 毫秒数
- end_time：切片最后一句台词在 transcript 中的原始 end 毫秒数
- reasoning：用简体中文简短解释边界选择原因

# 吸引力评分校准
- 1-3：较弱，独立观看价值低，偏填充或过度依赖上下文。
- 4-6：可用，但吸引力中等。
- 7-8：较强，有明确观看吸引力。
- 9-10：优秀；只给明显非常好笑、有张力、记忆点强或结尾满足感很强的片段。

所有 title、description、reasoning、engagement.reasoning 都必须使用简体中文。
不要虚高 engagement 分数；评分应相对于同一个 transcript 里的其他 clips 做校准。
"""

REFLECTION_PROMPT = """
请作为严格质检员，重新检查上一轮切片方案。

检查清单：
1. 是否有 clip 以明显悬空的代词、连词或依赖前文的开头开始？
2. 是否有 clip 时长严重偏离 60,000 到 180,000 毫秒的目标范围？
3. 是否错误切断了紧密问答、铺垫和笑点、建议和回应，或其他强关联对白单元？
4. 是否过度保守，遗漏了仍有英语学习价值的对白？
5. 相邻 clips 之间是否存在不必要的大量重叠？
6. 是否有更好的边界，可以保留更完整的日常表达、问答链条或固定搭配？
7. title、description、reasoning 是否都使用简体中文，并且与最终 clip 边界一致？
8. engagement 分数是否校准合理、没有虚高，并且与最终 clip 边界一致？
9. start_index 和 end_index 是否都来自 transcript 中实际存在的句子 index？
10. start_time 是否严格等于 transcript.sentences[start_index].start？
11. end_time 是否严格等于 transcript.sentences[end_index].end？
12. 是否错误地把视频时间轴或视频推断结果当成 transcript 时间戳使用？

如果发现问题，请修正并输出完整修正后的结果对象，格式仍然必须是 {"clips": [...]}。
如果方案已经可靠，请原样输出上一轮完整结果对象，格式仍然必须是 {"clips": [...]}。
"""


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
                engagement=clip.engagement,
                start_index=clip.start_index,
                end_index=clip.end_index,
                start_time=start_sentence["start"],
                end_time=end_sentence["end"],
                buffered_start_time=buffered_start_time,
                buffered_end_time=buffered_end_time,
                duration_time=buffered_end_time - buffered_start_time,
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


def get_index_constraint_text(transcript_data: dict) -> str:
    sentences = transcript_data.get("sentences", [])
    if not sentences:
        raise ValueError("transcript_data.sentences 为空，无法生成 index 约束")

    indexes = [sentence["index"] for sentence in sentences]
    return (
        "# Transcript index constraints\n"
        f"- valid_min_index: {min(indexes)}\n"
        f"- valid_max_index: {max(indexes)}\n"
        f"- valid_index_count: {len(indexes)}\n"
        "- start_index and end_index must be existing sentence indexes from the provided transcript JSON.\n"
        "- Never output an index smaller than valid_min_index or larger than valid_max_index.\n"
        "- Never infer extra sentence indexes from the video.\n"
    )


def validate_or_repair_clip_boundaries(
    response_obj: LLMClipResponse,
    transcript_data: dict,
    transcript_text: str,
    index_constraint_text: str,
    cached_content_name: str | None = None,
) -> LLMClipResponse:
    try:
        validate_clip_boundaries(response_obj.clips, transcript_data)
        return response_obj
    except ValueError as exc:
        validation_error = str(exc)
        print(f"⚠️  切片边界校验失败，执行一次自动修复: {validation_error}", flush=True)

    repair_prompt = (
        "上一轮切片结果没有通过代码侧确定性校验。\n"
        "请修复完整 JSON 结果，使它符合要求的 schema 和所有 transcript index 约束。\n"
        "不得从视频中发明句子 index。\n"
        "不得把视频时间轴或视频推断结果当成 transcript 时间戳使用。\n"
        "如果某个边界超出 transcript 范围，请选择语义上最接近且实际存在的 transcript 句子 index，并同步更新 title、description、engagement、start_time、end_time 和 reasoning，使它们与修正后的边界一致。\n\n"
        f"{index_constraint_text}\n"
        "校验错误：\n"
        f"{validation_error}\n\n"
        "上一轮无效切片结果：\n"
        f"{response_obj.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
        "Transcript JSON：\n"
        f"{transcript_text}"
    )

    repaired_response_obj = invoke_structured_gemini(
        repair_prompt,
        cached_content_name=cached_content_name,
    )
    validate_clip_boundaries(repaired_response_obj.clips, transcript_data)
    return repaired_response_obj

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
def process_transcript_pipeline(
    input_filepath: str,
    output_filepath: str | None = None,
    video_gcs_uri: str | None = None,
    video_mime_type: str = "video/mp4",
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
):
    input_path = Path(input_filepath)
    
    # 步骤 1：读取原始 transcript JSON
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # 步骤 2：删除 tokens，避免把词级信息和额外噪音送给大模型
    cleaned_transcript_data = remove_tokens_from_transcript(transcript_data)
    transcript_text = json.dumps(cleaned_transcript_data, ensure_ascii=False)
    index_constraint_text = get_index_constraint_text(transcript_data)

    cached_content_name = None
    if video_gcs_uri:
        cached_content_name = create_video_context_cache(
            video_gcs_uri=video_gcs_uri,
            video_mime_type=video_mime_type,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    
    try:
        # 步骤 3：第一轮切片
        # 让模型同时输出边界与辅助时间，便于其自查时长和闭环性
        print("🚀 [1/3] 执行初次切片生成...")
        initial_prompt = (
            f"{index_constraint_text}\n"
            "Please perform semantic clipping on the following transcript:\n"
            f"{transcript_text}"
        )

        initial_response_obj = invoke_structured_gemini(
            initial_prompt,
            cached_content_name=cached_content_name,
        )
        
        # 步骤 4：第二轮反思优化
        # 把第一次结果回灌给模型，让模型自查边界质量
        print("🔍 [2/3] 执行自我反思与优化...")
        ai_memory_str = initial_response_obj.model_dump_json(indent=2, ensure_ascii=False)
        
        reflection_prompt = (
            f"{index_constraint_text}\n"
            "Previous clipping plan:\n"
            f"{ai_memory_str}\n\n"
            f"{REFLECTION_PROMPT}"
        )

        # 再次调用，得到最终的边界结果
        final_response_obj = invoke_structured_gemini(
            reflection_prompt,
            cached_content_name=cached_content_name,
        )

        # 步骤 5：先校验边界是否合法，再做时间回填
        # 注意：即便模型给出的 start_time / end_time 有偏差，最终仍以 transcript 为准
        final_response_obj = validate_or_repair_clip_boundaries(
            response_obj=final_response_obj,
            transcript_data=transcript_data,
            transcript_text=transcript_text,
            index_constraint_text=index_constraint_text,
            cached_content_name=cached_content_name,
        )

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
    finally:
        if cached_content_name:
            try:
                delete_context_cache(cached_content_name)
            except Exception as exc:
                print(
                    f"⚠️  explicit context cache 删除失败: {cached_content_name} | {type(exc).__name__}: {exc}",
                    flush=True,
                )

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
    parser.add_argument(
        "--video-gcs-uri",
        default=None,
        help="可选：GCS 视频对象地址，例如 gs://bucket/path/video.mp4；提供后会创建 explicit context cache 并作为多模态上下文",
    )
    parser.add_argument(
        "--video-mime-type",
        default="video/mp4",
        help="可选：视频 MIME type，默认 video/mp4",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=DEFAULT_CACHE_TTL_SECONDS,
        help="可选：explicit context cache TTL 秒数，默认 1200 秒（20 分钟）",
    )
    args = parser.parse_args()

    process_transcript_pipeline(
        args.input,
        args.output,
        video_gcs_uri=args.video_gcs_uri,
        video_mime_type=args.video_mime_type,
        cache_ttl_seconds=args.cache_ttl_seconds,
    )
