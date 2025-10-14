from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Define Pydantic data structures

class SemanticElement(BaseModel):
    id: int = Field(description="ID，现在随机生成")
    baseForm: str = Field(description="单词原形")
    dictionary: str = Field(description="词典释义，对该语义元素的解释")

class SubtitleToken(BaseModel):
    text: str = Field(description="token 的文本内容")
    explanation: str = Field(description="根据上下文对该 token 的中文解释/注释")
    semanticElement: SemanticElement = Field(description="在词典中的语义元素信息")

class Sentence(BaseModel):
    text: str = Field(description="句子的完整英文原文")
    explanation: Optional[str] = Field(description="对整个句子的中文解释", default=None)
    tokens: List[SubtitleToken] = Field(description="构成该句子的 token 列表")

# Top-level model to wrap the sentences list
class AnalysisResult(BaseModel):
    """用于包装句子列表的顶层 Pydantic 模型"""
    sentences: List[Sentence] = Field(description="分析后的句子列表")


def analyze_english_text_to_sentences(text_to_analyze: str) -> types.GenerateContentResponse:
    try:
        client = genai.Client(api_key=API_KEY)

        prompt = f"""
        请将以下英文文本进行结构化分析。请严格遵循以下指示：
        1.  按顺序为每个句子提供一个整体的中文翻译或解释（explanation）。
        2.  按顺序将每个句子进一步分解为有意义的语言元素分片（SubtitleToken）可以是单词，对于简单常用的单词，也可分为短语固定搭配。
            **以英语学习为目的拆分**：对于难度稍高的词组（如"be addicted to"），应该作为一个token，在explanation中同时解释短语含义和核心单词，semanticElement.baseForm应为核心词的原形（如"addicted"）。

            正面例子（应该这样拆分）：
            - "by the way" -> 保持为一个token（固定短语）
            - "deal with" -> 保持为一个token（固定搭配）
            - "at the same time" -> 保持为一个token（固定短语）
            - "looking forward to" -> 保持为一个token（固定搭配）
            - "be addicted to" -> 保持为一个token，explanation: "对...上瘾；addicted表示沉迷的、上瘾的"，baseForm: "addicted"
            - "get rid of" -> 保持为一个token，explanation: "摆脱、除去；rid表示使摆脱"，baseForm: "rid"

            反面例子（不该这样拆分）：
            - "I am happy" -> 不要拆成"I am"作为一个token，应该分别为"I"、"am"、"happy"三个token
            - "the book" -> 不要保持为一个token，应该分别为"the"、"book"两个token
            - "very good" -> 不要保持为一个token，应该分别为"very"、"good"两个token
        3.  为每个分片（token）提供符合上下文语境的中文解释（explanation），标点符号为空。
        4.  对于超简单，最常用单词，比如if, is, are, the, but, or, a, and等，允许explanation为空，你自行决定。
        5.  为每个token生成一个semanticElement对象，包含：
           - id: 随机生成的数字ID
           - baseForm: 单词的原形（如running -> run，studies -> study，标点符号为空）
           - dictionary: 词典释义 (标点符号为空)
        6.  任何标点符号都应被视为一个独立的token，explanation和semanticElement所有字段为空。
        7.  严格按照我提供的 JSON schema 格式，输出一个包含句子列表的 JSON 对象。

        需要分析的英文文本如下：
        ---
        {text_to_analyze}
        ---
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # Use the top-level model as schema
                response_schema=AnalysisResult,
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            )
        )
        return response

    except Exception as e:
        print(f"调用 API 时发生错误: {e}")
        raise

# Main program
if __name__ == "__main__":
    english_text_input = """[
      {
        "index": 0,
        "text": "Number one most racist country in Europe by a landslide, by absolutely no competition is the Czech Republic."
      },
      {
        "index": 1,
        "text": "Bro, I don't know what kind of race theory they're teaching them in Czech schools, but I genuinely did not expect it to be this bad."
      },
      {
        "index": 2,
        "text": "If Czechia had more black people, which it doesn't have any now, but if they did, they would make Jim Crow look like a progressive Swedish government."
      }
    ]"""

    print(f"正在分析文本: \"{english_text_input}\"")
    print("-" * 30)

    try:
        api_response = analyze_english_text_to_sentences(english_text_input)

        print("--- 原始 JSON 输出 ---")
        parsed_json = json.loads(api_response.text)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

        print("\n--- 解析后的 Pydantic 对象遍历 ---")
        
        # Get sentences list from parsed result
        parsed_result: AnalysisResult = api_response.parsed
        if parsed_result and parsed_result.sentences:
            sentence_list = parsed_result.sentences
            for i, sentence in enumerate(sentence_list):
                print(f"\n句子 {i}: \"{sentence.text}\"")
                print(f"  句子解释: {sentence.explanation}")
                print("  Tokens 详解:")
                for j, token in enumerate(sentence.tokens):
                    print(f"    - {j}. '{token.text}': {token.explanation}")
                    print(f"      语义元素 ID: {token.semanticElement.id}, 原形: '{token.semanticElement.baseForm}', 词典: '{token.semanticElement.dictionary}'")
        else:
            print("未能成功解析响应或响应中不包含句子。")
            print("原始响应内容:", api_response.text)

    except Exception as e:
        print(f"\n程序运行失败。")