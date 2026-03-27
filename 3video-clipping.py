import os
import json
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 1. 定义结构化输出数据模型
#    - ClipBoundary: 只描述大模型需要决定的“句子边界”
#    - Clip: 在 ClipBoundary 基础上补齐精确时间和缓冲时间
#    - ClipBoundaryResponse: 约束模型必须返回 {"clips": [...]} 结构
# ==========================================
class ClipBoundary(BaseModel):
    clip_id: int = Field(description="切片的唯一序号")
    start_index: int = Field(description="切片起始台词的 index")
    end_index: int = Field(description="切片结束台词的 index")
    reasoning: str = Field(description="简述边界判定与取舍逻辑")

class Clip(ClipBoundary):
    start_time: int = Field(description="切片的起始绝对毫秒数")
    end_time: int = Field(description="切片的结束绝对毫秒数")
    buffered_start_time: int = Field(description="带缓冲的切片起始毫秒数，用于实际视频裁切")
    buffered_end_time: int = Field(description="带缓冲的切片结束毫秒数，用于实际视频裁切")

class ClipBoundaryResponse(BaseModel):
    clips: List[ClipBoundary] = Field(description="切片数组")

# ==========================================
# 2. 初始化运行环境与 LLM
#    - 从 .env 读取 OPENAI_API_KEY
#    - 初始化 ChatOpenAI
#    - 用 with_structured_output 锁定模型输出 schema
#    - 定义后处理 buffer 参数，供视频裁切阶段复用
# ==========================================
# 从 .env 加载环境变量，便于本地开发直接读取 API Key
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4", # 必须使用支持严格结构化输出的模型
    temperature=0.1 # 极低温度，确保逻辑确定性
)

# 绑定结构化输出模型，要求 LLM 严格按 ClipBoundaryResponse 返回
structured_llm = llm.with_structured_output(ClipBoundaryResponse)

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
你是一个负责视频流预处理的“语义切片引擎”。你的唯一任务是读取带有毫秒级时间戳的 JSON 台词文本，将其划分为多个 1 到 3 分钟（约 60,000 - 180,000 毫秒）的粗略区块，专供英语学习使用。

# 核心目标
1. 语义闭环优先：每个切片应尽量自洽，单独观看时不明显悬空。
2. 语言连贯优先：不要在完整句子、紧凑问答或强依赖上下文的表达链条中间切断。
3. 起点自然：避免让切片以明显承接上文的代词、连词或残句开头。
4. 时长是次级约束：若无法同时满足时长和语义完整，优先保证语义完整，再尽量靠近 3 分钟。

# 切分线索
- 优先在场景切换、话题切换、较长停顿、问答回合结束、动作/笑点完成后切分。
- 不要把 setup 和 payoff、提问和回答、建议和回应拆到两个切片里。
- 默认尽量覆盖大部分有学习价值的对白；只在片段明显低价值、重复寒暄或强依赖前文且难以独立成立时才舍弃。

# 英语学习导向（软建议）
- 在不破坏语义闭环的前提下，可优先保留包含高频口语、自然问答、固定搭配、短语动词、习语、情绪表达、职场/日常场景表达的片段。
- 如果两个切法都同样自然，可优先选择更适合跟读、复述、单独学习的方案。

# 输出要求
只输出句子边界：
- start_index：切片第一句台词的 index
- end_index：切片最后一句台词的 index
- reasoning：简述边界取舍
- 不要输出 start_time 或 end_time，它们会由程序自动回填
"""

REFLECTION_PROMPT = """
请作为严格的质检员，重新审视你刚才输出的切片方案。
审查重点：
1. 是否有明显的“首句代词悬空”导致语境断裂？
2. 是否有切片的时间跨度严重偏离 60,000 - 180,000 毫秒的硬性约束？
3. 是否错误切断了紧凑的问答回合？
4. 是否因为过度保守而遗漏了大量本可学习的对白？
5. 是否有不必要的大量重叠？
6. 是否有更适合学习的边界选择，可以保留更完整的常用表达、问答链条或固定搭配？

如果发现缺陷，请修正并输出优化后的完整结果对象，格式必须仍然是 {"clips": [...]}。
如果逻辑已经闭环，无需优化，请直接原样输出之前的完整结果对象，格式必须仍然是 {"clips": [...]}。
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
#    - LLM 只返回 start_index / end_index
#    - 这里根据原始 transcript 回填：
#      1) 精确 start_time / end_time
#      2) 带缓冲的 buffered_start_time / buffered_end_time
# ==========================================
def add_timestamps_to_clips(clips: List[ClipBoundary], transcript_data: dict) -> List[Clip]:
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
def validate_clip_boundaries(clips: List[ClipBoundary], transcript_data: dict) -> None:
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
#    3) 第一轮生成切片边界
#    4) 第二轮反思优化边界
#    5) 校验边界合法性
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
    # 只让模型决定“按哪些句子边界切”，不让它生成时间戳
    print("🚀 [1/3] 执行初次切片生成...")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请对以下台词进行语义切片：\n{transcript_text}")
    ]
    
    # structured_llm 会直接返回 ClipBoundaryResponse 对象
    initial_response_obj = structured_llm.invoke(messages)
    
    # 步骤 4：第二轮反思优化
    # 把第一次结果作为 AIMessage 回灌，让模型自查边界质量
    print("🔍 [2/3] 执行自我反思与优化...")
    ai_memory_str = initial_response_obj.model_dump_json(indent=2, ensure_ascii=False)
    
    messages.append(AIMessage(content=ai_memory_str))
    messages.append(HumanMessage(content=REFLECTION_PROMPT))
    
    # 再次调用，得到最终的边界结果
    final_response_obj = structured_llm.invoke(messages)

    # 步骤 5：先校验边界是否合法，再做时间回填
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
