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
# 1. 定义极其严密的 Pydantic 数据结构
# ==========================================
class ClipBoundary(BaseModel):
    clip_id: int = Field(description="切片的唯一序号")
    start_index: int = Field(description="切片起始台词的 index")
    end_index: int = Field(description="切片结束台词的 index")
    reasoning: str = Field(description="简述边界判定与取舍逻辑")

class Clip(ClipBoundary):
    start_time: int = Field(description="切片的起始绝对毫秒数")
    end_time: int = Field(description="切片的结束绝对毫秒数")

class ClipBoundaryResponse(BaseModel):
    clips: List[ClipBoundary] = Field(description="切片数组")

# ==========================================
# 2. 初始化核心参数与 LLM 引擎
# ==========================================
# 从 .env 加载环境变量
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4", # 必须使用支持严格结构化输出的模型
    temperature=0.1 # 极低温度，确保逻辑确定性
)

# 绑定 Pydantic 模型，物理锁死输出结构
structured_llm = llm.with_structured_output(ClipBoundaryResponse)

# ==========================================
# 3. 核心 Prompt 定义
# ==========================================
SYSTEM_PROMPT = """
你是一个负责视频流预处理的“语义切片引擎”。你的唯一任务是读取带有毫秒级时间戳的 JSON 台词文本，将其划分为多个 1 到 3 分钟（约 60,000 - 180,000 毫秒）的粗略区块，专供英语学习使用。

# 切片优先级
1. 绝对独立与不相关：每个切片必须是一个自洽、完整的语境闭环。
2. 适度重叠以保闭环：相邻切片之间允许有部分时间段重叠，但严禁重叠过多导致学习重复。
3. 语言连贯优先：绝不能在完整的句子、意群或紧凑的问答中间切断。
4. 大胆舍弃（留白清洗）：直接丢弃语言学习价值极低的片段。允许大段的时间轴留白。

# 输出要求
你只需要输出句子边界：
- start_index：切片第一句台词的 index 原始数值。
- end_index：切片最后一句台词的 index 原始数值。
不要输出 start_time 或 end_time，它们会由下游程序根据原始数据自动回填。
"""

REFLECTION_PROMPT = """
请作为严格的质检员，重新审视你刚才输出的切片方案。
审查重点：
1. 是否有明显的“首句代词悬空”导致语境断裂？
2. 是否有切片的时间跨度严重偏离 60,000 - 180,000 毫秒的硬性约束？
3. 是否错误切断了紧凑的问答回合？

如果发现缺陷，请修正并输出优化后的完整 JSON 数组。
如果逻辑已经完美闭环，无需优化，请直接原样输出之前的 JSON 数组。
"""


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


def add_timestamps_to_clips(clips: List[ClipBoundary], transcript_data: dict) -> List[Clip]:
    sentence_map = {
        sentence["index"]: sentence
        for sentence in transcript_data.get("sentences", [])
    }

    completed_clips = []
    for clip in clips:
        start_sentence = sentence_map[clip.start_index]
        end_sentence = sentence_map[clip.end_index]
        completed_clips.append(
            Clip(
                clip_id=clip.clip_id,
                start_index=clip.start_index,
                end_index=clip.end_index,
                start_time=start_sentence["start"],
                end_time=end_sentence["end"],
                reasoning=clip.reasoning,
            )
        )

    return completed_clips

# ==========================================
# 4. 主执行管线
# ==========================================
def process_transcript_pipeline(input_filepath: str, output_filepath: str | None = None):
    input_path = Path(input_filepath)
    
    # 步骤 1：读取 JSON 文件
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    cleaned_transcript_data = remove_tokens_from_transcript(transcript_data)
    transcript_text = json.dumps(cleaned_transcript_data, ensure_ascii=False)
    
    # 步骤 2：构建初始上下文并调用大模型（First Pass）
    print("🚀 [1/3] 执行初次切片生成...")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请对以下台词进行语义切片：\n{transcript_text}")
    ]
    
    # structured_llm 直接返回 Pydantic 对象
    initial_response_obj = structured_llm.invoke(messages)
    
    # 步骤 3：反思机制（Reflection Pass）
    print("🔍 [2/3] 执行自我反思与优化...")
    # 必须将大模型的输出转回 JSON 字符串，并作为 AIMessage 注入历史栈，建立完整因果链
    ai_memory_str = initial_response_obj.model_dump_json(indent=2, ensure_ascii=False)
    
    messages.append(AIMessage(content=ai_memory_str))
    messages.append(HumanMessage(content=REFLECTION_PROMPT))
    
    # 再次调用，输出最终优化的对象
    final_response_obj = structured_llm.invoke(messages)
    final_clips = add_timestamps_to_clips(final_response_obj.clips, transcript_data)
    
    # 步骤 4：保存结果
    print("💾 [3/3] 正在落盘保存...")
    if output_filepath is None:
        output_path = Path("3clipped") / input_path.name
    else:
        output_path = Path(output_filepath)

    output_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"clips": [clip.model_dump() for clip in final_clips]},
            f,
            indent=4,
            ensure_ascii=False,
        )
        
    print(f"✅ 任务完成！文件已保存至: {output_path}")

# ==========================================
# 启动入口
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
