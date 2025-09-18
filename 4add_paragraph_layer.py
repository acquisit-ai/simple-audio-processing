#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def add_paragraph_layer(input_path: str, output_path: str = None) -> str:
    """
    将现有JSON结构转换为包含Paragraph层的新结构

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选，如果不提供则自动生成）

    Returns:
        str: 输出文件路径
    """

    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建新结构
    new_structure = {
        "language": data["language"],
        "total_sentences": data["total_sentences"],
        "total_tokens": data["total_tokens"],
        "total_paragraphs": len(data["sentences"]),  # 每个句子作为一个段落
        "paragraphs": []
    }

    # 将每个句子包装成一个段落
    for i, sentence in enumerate(data["sentences"]):
        paragraph = {
            "index": i,
            "total_sentences": 1,  # 每个段落只有一个句子
            "sentences": [sentence]  # 将句子放入列表
        }
        new_structure["paragraphs"].append(paragraph)

    # 生成输出文件路径
    if output_path is None:
        # 如果未指定输出路径，使用默认规则
        input_path_obj = Path(input_path)
        output_dir = Path("4final")
        base_name = input_path_obj.stem  # 不含扩展名的文件名
        output_path = output_dir / f"{base_name}-final.json"
    else:
        output_path = Path(output_path)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存到输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_structure, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {input_path} -> {output_path}")
    print(f"总段落数: {new_structure['total_paragraphs']}")
    print(f"总句子数: {new_structure['total_sentences']}")
    print(f"总token数: {new_structure['total_tokens']}")

    return str(output_path)

if __name__ == "__main__":
    # 直接指定输入输出路径
    input_file = "3llm/3min2-cleaned-gemini.json"
    output_file = "4final/3min2-cleaned-gemini-final.json"

    add_paragraph_layer(input_file, output_file)