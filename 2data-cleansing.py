import json
from pathlib import Path
from typing import List, TypedDict

# Type definitions
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

class Paragraph(TypedDict):
    language: str
    sentences: List[Sentence]

def process_whisperx_to_cleaned(input_path: str, output_path: str = None) -> None:
    # Read WhisperX output
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract language
    language = data.get('detected_language', 'en')

    # Process segments to sentences
    sentences = []
    for idx, segment in enumerate(data.get('segments', [])):
        # Create sentence
        sentence: Sentence = {
            'index': idx,
            'text': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end'],
            'tokens': []
        }

        # Process words to tokens
        words = segment.get('words', [])
        for word_idx, word in enumerate(words):
            # Skip words without timing info
            if 'start' not in word or 'end' not in word:
                continue

            token: SubtitleToken = {
                'index': word_idx,
                'text': word['word'],
                'start': word['start'],
                'end': word['end']
            }
            sentence['tokens'].append(token)

        sentences.append(sentence)

    # Create final Paragraph structure
    paragraph: Paragraph = {
        'language': language,
        'sentences': sentences
    }

    # Determine output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = f"cleaned-data/{input_file.stem}-cleaned.json"

    # Save to output file
    save_cleaned_data(paragraph, output_path, language, sentences)

    return paragraph


def save_cleaned_data(paragraph: Paragraph, output_path: str, language: str, sentences: List[Sentence]) -> None:
    """
    保存清理后的数据到JSON文件

    Args:
        paragraph: 清理后的段落数据
        output_path: 输出文件路径
        language: 检测到的语言
        sentences: 句子列表
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(paragraph, f, indent=2, ensure_ascii=False)

    print(f"Data cleaned and saved to: {output_path}")
    print(f"Language: {language}")
    print(f"Number of sentences: {len(sentences)}")
    total_tokens = sum(len(s['tokens']) for s in sentences)
    print(f"Total tokens: {total_tokens}")


if __name__ == '__main__':
    input_file = '1transcript-raw/3min1.json'
    output_file = '2cleaned-data/3min1-cleaned.json'

    # 使用默认输出路径（自动生成）
    process_whisperx_to_cleaned(input_file, output_file)