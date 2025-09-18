"""
英文字幕批量分析工具

功能说明：
1. 从清理后的字幕数据中提取简化版句子（仅保留index和text）
2. 使用Gemini API并行批量分析英文句子，生成中文解释和词汇分析
3. 自动处理失败重试，确保数据完整性
4. 按原始顺序合并所有分析结果并保存

使用方法：
直接运行 python 3llm.py 即可自动处理 cleaned-data/3min1-cleaned.json

输出文件：
- 3llm/3min1-cleaned-gemini.json (完整分析结果)
"""

import json
import sys
import time
from pathlib import Path
from typing import List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入gemini.py中的分析函数
sys.path.append(str(Path(__file__).parent))
from gemini import analyze_english_text_to_sentences

# 配置常量
DEFAULT_BATCH_SIZE = 10  # 每批次处理的句子数量
DEFAULT_MAX_WORKERS = 5  # 最大并行线程数

# 输入数据类型定义
class Sentence(TypedDict):
    index: int  # 句子编号
    text: str   # 句子文本

class Paragraph(TypedDict):
    sentences: List[Sentence]  # 句子列表

def analyze_batch_wrapper(batch_info: Tuple[int, List[dict]], stop_event: threading.Event) -> Tuple[int, List[dict], float]:
    """
    批次分析包装函数，调用Gemini API分析一批句子。

    参数:
        batch_info: (批次索引, 句子列表) 的元组
        stop_event: 用于停止处理的事件

    返回:
        (批次索引, 分析结果列表, 批次用时) 的元组
    """
    batch_idx, batch = batch_info
    batch_num = batch_idx + 1

    # 检查是否已经被请求停止
    if stop_event.is_set():
        return (batch_idx, [], 0.0)

    start_time = time.time()
    print(f"  开始处理批次 {batch_num} (句子 {batch[0]['index']+1}-{batch[-1]['index']+1})")

    # 将批次转换为JSON字符串供分析
    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

    try:
        # 再次检查停止事件
        if stop_event.is_set():
            return (batch_idx, [], 0.0)

        # 调用gemini.py中的分析函数
        result = analyze_english_text_to_sentences(batch_json)

        end_time = time.time()
        batch_time = end_time - start_time

        if result and result.parsed and result.parsed.sentences:
            # 将Pydantic模型转换为字典
            analyzed = [sentence.model_dump() for sentence in result.parsed.sentences]
            print(f"  ✓ 批次 {batch_num} 完成: {len(analyzed)} 个句子 (用时: {batch_time:.2f}秒)")
            return (batch_idx, analyzed, batch_time)
        else:
            print(f"  ✗ 批次 {batch_num} 失败 (用时: {batch_time:.2f}秒)")
            # 设置停止事件，立即终止其他线程
            stop_event.set()
            return (batch_idx, [], batch_time)
    except Exception as e:
        end_time = time.time()
        batch_time = end_time - start_time
        print(f"  ✗ 批次 {batch_num} 错误: {e} (用时: {batch_time:.2f}秒)")
        # 设置停止事件，立即终止其他线程
        stop_event.set()
        return (batch_idx, [], batch_time)

def process_sentences_with_llm(sentences_data: dict, output_path: str = None, batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
    """
    使用Gemini API并行批量处理清理后的句子。

    参数:
        sentences_data: 简化后的句子数据字典
        output_path: 可选输出路径，不提供则自动生成'-gemini'后缀
        batch_size: 每批次处理的句子数量（默认10个）
        max_workers: 最大并行线程数（默认5个）
    """
    # 从内存中获取句子数据
    sentences = sentences_data.get('sentences', [])
    total_sentences = len(sentences)
    print(f"总计需要处理的句子数: {total_sentences}")

    # 准备带索引的批次
    batches = []
    for i in range(0, total_sentences, batch_size):
        batch = sentences[i:i + batch_size]
        batches.append((i // batch_size, batch))

    total_batches = len(batches)
    print(f"总批次数: {total_batches}")
    print(f"使用 {min(max_workers, total_batches)} 个并行工作线程处理...")

    # 开始总体计时
    total_start_time = time.time()

    # 创建停止事件
    stop_event = threading.Event()

    # 并行处理批次
    results_dict = {}
    batch_times = []

    with ThreadPoolExecutor(max_workers=min(max_workers, total_batches)) as executor:
        # 提交所有批次处理任务
        futures = {executor.submit(analyze_batch_wrapper, batch, stop_event): batch[0] for batch in batches}

        # 收集完成的结果
        completed = 0
        for future in as_completed(futures):
            batch_idx, analyzed, batch_time = future.result()
            results_dict[batch_idx] = analyzed
            batch_times.append(batch_time)
            completed += 1
            print(f"进度: {completed}/{total_batches} 批次已完成")

            # 如果有批次失败，立即取消所有未完成的任务
            if len(analyzed) == 0 and batch_time > 0:  # 失败但不是被停止事件取消的
                print(f"\n⚠️  检测到批次失败，正在取消剩余任务...")
                # 取消所有未完成的futures
                for f in futures:
                    f.cancel()
                break

    # 检查是否有批次失败
    failed_batches = []
    for i in range(total_batches):
        if i not in results_dict or len(results_dict[i]) == 0:
            failed_batches.append(i + 1)

    if failed_batches:
        print(f"\n❌ 分析失败!")
        print(f"  失败的批次: {failed_batches}")
        print(f"  失败总数: {len(failed_batches)}/{total_batches}")
        print(f"  由于处理不完整，文件未保存。")
        return

    # 计算总体时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # 按正确顺序合并结果
    all_analyzed_sentences = []
    for i in range(total_batches):
        all_analyzed_sentences.extend(results_dict[i])

    print(f"\n按正确顺序合并结果...")

    # 创建最终结果结构
    final_result = {
        "language": "en",
        "total_sentences": len(all_analyzed_sentences),
        "sentences": all_analyzed_sentences
    }

    # 确定输出路径
    if output_path is None:
        output_path = "3llm/3min1-cleaned-gemini.json"

    # 保存分析后的数据
    save_llm_result(final_result, output_path, all_analyzed_sentences, total_time, batch_times, total_batches)


def save_llm_result(final_result: dict, output_path: str, all_analyzed_sentences: list, total_time: float, batch_times: list, total_batches: int) -> None:
    """
    保存LLM分析结果到JSON文件

    Args:
        final_result: 最终结果字典
        output_path: 输出文件路径
        all_analyzed_sentences: 所有分析后的句子
        total_time: 总处理时间
        batch_times: 各批次用时列表
        total_batches: 总批次数
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    # 计算平均批次时间
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

    print(f"\n✓ 分析完成!")
    print(f"  输出已保存到: {output_path}")
    print(f"  总共分析句子数: {len(all_analyzed_sentences)}")
    print(f"\n⏱️  时间统计:")
    print(f"  总处理时间: {total_time:.2f}秒")
    print(f"  平均批次时间: {avg_batch_time:.2f}秒")
    print(f"  处理批次数: {total_batches}")


def extract_sentences_only(input_path: str) -> dict:
    """
    从句子中仅提取index和text字段，返回内存中的数据。

    参数:
        input_path: 清理后的JSON文件路径

    返回:
        简化后的数据字典
    """
    # 读取清理后的数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 仅提取句子的index和text字段
    simplified_sentences = []
    for sentence in data.get('sentences', []):
        simplified_sentence: Sentence = {
            'index': sentence['index'],
            'text': sentence['text']
        }
        simplified_sentences.append(simplified_sentence)

    # 创建简化的段落结构
    paragraph: Paragraph = {
        'sentences': simplified_sentences
    }

    print(f"已提取句子数量: {len(simplified_sentences)}")

    return paragraph

if __name__ == '__main__':
    # 直接执行，无需命令行参数
    input_file = '2cleaned-data/3min2-cleaned.json'
    output_file = '3llm/3min2-cleaned-gemini.json'

    # 步骤1: 在内存中提取句子
    print("步骤1: 提取句子...")
    simplified_data = extract_sentences_only(input_file)

    # 步骤2: 使用LLM处理
    print("\n步骤2: 使用Gemini API处理...")
    process_sentences_with_llm(simplified_data, output_file)