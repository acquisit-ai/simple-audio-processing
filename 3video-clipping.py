import os
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 1. 定义极其严密的 Pydantic 数据结构
# ==========================================
class Clip(BaseModel):
    clip_id: int = Field(description="切片的唯一序号")
    start_index: int = Field(description="切片起始台词的 index")
    end_index: int = Field(description="切片结束台词的 index")
    start_time: int = Field(description="切片的起始绝对毫秒数")
    end_time: int = Field(description="切片的结束绝对毫秒数")
    reasoning: str = Field(description="简述边界判定与取舍逻辑")

class ClipResponse(BaseModel):
    clips: List[Clip] = Field(description="切片数组")

# ==========================================
# 2. 初始化核心参数与 LLM 引擎
# ==========================================
# 请在环境变量中配置 OPENAI_API_KEY
llm = ChatOpenAI(
    model="gpt-5.4", # 必须使用支持严格结构化输出的模型
    temperature=0.1 # 极低温度，确保逻辑确定性
)

# 绑定 Pydantic 模型，物理锁死输出结构
structured_llm = llm.with_structured_output(ClipResponse)

# ==========================================
# 3. 核心 Prompt 定义
# ==========================================
SYSTEM_PROMPT = """
你是一个负责视频流预处理的“语义切片引擎”。你的唯一任务是读取带有毫秒级时间戳的 JSON 台词文本，将其划分为多个 1 到 3 分钟（约 60,000 - 180,000 毫秒）的粗略区块，专供英语学习使用。

# 切片优先级
1. 绝对独立与不相关：每个切片必须是一个自洽、完整的语境闭环。
2. 适度重叠以保闭环：相邻切片之间允许有部分时间段重叠，但严禁重叠过多导致学习重复。
3. 语言连贯优先：绝不能在完整的句子、意群或紧凑的问答中间切断。
4. 首句防悬空（代词限制）：起始句必须有明确主语。避免以无前置指代的代词（He, She, It等）开局。
5. 大胆舍弃（留白清洗）：直接丢弃语言学习价值极低的片段。允许大段的时间轴留白。

# 时间戳取值规则
直接提取原始数值，严禁任何数学加减法计算：
- start_time：切片第一句台词的 start 原始数值。
- end_time：切片最后一句台词的 end 原始数值。
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

# ==========================================
# 4. 主执行管线
# ==========================================
def process_transcript_pipeline(input_filepath: str):
    input_path = Path(input_filepath)
    
    # 步骤 1：读取 JSON 文件
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    transcript_text = json.dumps(transcript_data, ensure_ascii=False)
    
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
    
    # 步骤 4：保存结果
    print("💾 [3/3] 正在落盘保存...")
    output_dir = Path("3clipped")
    output_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
    
    output_filepath = output_dir / input_path.name # 保持同名
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # 将最终的 Pydantic 对象序列化并保存
        f.write(final_response_obj.model_dump_json(indent=4, ensure_ascii=False))
        
    print(f"✅ 任务完成！文件已保存至: {output_filepath}")

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的 transcript json 路径
    source_file = "transcript_01.json" 
    process_transcript_pipeline(source_file)