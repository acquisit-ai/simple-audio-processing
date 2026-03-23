import json
from pathlib import Path
from typing import List, TypedDict


class SubtitleToken(TypedDict):
    index: int
    text: str
    start: float
    end: float


class Sentence(TypedDict):
    index: int
    text: str
    start: float
    end: float
    tokens: List[SubtitleToken]


class Chapter(TypedDict):
    index: int
    headline: str
    summary: str
    gist: str
    start: float
    end: float


class Paragraph(TypedDict):
    sentences: List[Sentence]
    chapters: List[Chapter]


def _build_token(index: int, word: dict) -> SubtitleToken:
    """将 AssemblyAI word 结构转换为统一 token 结构。"""
    return {
        "index": index,
        "text": word["text"],
        "start": word["start"],
        "end": word["end"],
    }


def _build_sentence(index: int, sentence_data: dict) -> Sentence:
    """将 AssemblyAI sentence 结构转换为统一句子结构。"""
    tokens = [
        _build_token(word_idx, word)
        for word_idx, word in enumerate(sentence_data.get("words", []))
        if "start" in word and "end" in word
    ]

    return {
        "index": index,
        "text": sentence_data["text"].strip(),
        "start": sentence_data["start"],
        "end": sentence_data["end"],
        "tokens": tokens,
    }


def _build_chapter(index: int, chapter_data: dict) -> Chapter:
    """将 AssemblyAI chapter 结构转换为统一章节结构。"""
    return {
        "index": index,
        "headline": chapter_data.get("headline", ""),
        "summary": chapter_data.get("summary", ""),
        "gist": chapter_data.get("gist", ""),
        "start": chapter_data["start"],
        "end": chapter_data["end"],
    }


def _load_assemblyai_result(input_path: str) -> dict:
    """读取并校验 AssemblyAI 合并输出。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "sentences" not in data:
        raise ValueError("输入文件不符合 AssemblyAI 合并输出格式：缺少 sentences 字段")

    return data


def process_assemblyai_to_cleaned(input_path: str, output_path: str = None) -> Paragraph:
    """
    读取 AssemblyAI 输出 JSON，并转换为清理后的统一结构。

    注意：函数名为兼容现有调用链保留，但当前只支持 AssemblyAI 输出。
    """
    data = _load_assemblyai_result(input_path)

    sentences = [
        _build_sentence(idx, sentence_data)
        for idx, sentence_data in enumerate(data.get("sentences", []))
    ]
    chapters = [
        _build_chapter(idx, chapter_data)
        for idx, chapter_data in enumerate(data.get("chapters", []))
    ]

    paragraph: Paragraph = {
        "sentences": sentences,
        "chapters": chapters,
    }

    if output_path is None:
        input_file = Path(input_path)
        output_path = f"2cleaned-data/{input_file.stem}-cleaned.json"

    save_cleaned_data(paragraph, output_path)
    return paragraph


def save_cleaned_data(paragraph: Paragraph, output_path: str) -> None:
    """保存清理后的数据到 JSON 文件。"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paragraph, f, indent=2, ensure_ascii=False)

    sentence_count = len(paragraph["sentences"])
    chapter_count = len(paragraph["chapters"])
    total_tokens = sum(len(sentence["tokens"]) for sentence in paragraph["sentences"])

    print(f"Data cleaned and saved to: {output_path}")
    print(f"Number of sentences: {sentence_count}")
    print(f"Number of chapters: {chapter_count}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    input_file = "1transcript-raw/test.json"
    output_file = "2cleaned-data/test-cleaned.json"
    process_assemblyai_to_cleaned(input_file, output_file)
