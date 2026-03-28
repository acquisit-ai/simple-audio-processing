import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ==========================================
# 1. 批处理配置
#    - 默认扫描 2cleaned-data/ 下的所有 JSON 文件
#    - 默认输出到 3clipped/ 下并保持同名
#    - 默认使用 5 个并发工作线程
#    - 单文件失败时自动重试 1 次
# ==========================================
DEFAULT_INPUT_DIR = Path("2cleaned-data")
DEFAULT_OUTPUT_DIR = Path("3clipped")
DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_RETRIES = 1
CLIPPING_SCRIPT = Path("3llm-clipping.py")


# ==========================================
# 2. 单文件任务执行器
#    - 每个任务单独启动一个 Python 子进程
#    - 这样可以避免多个线程共享同一个 LLM 客户端实例
#    - 返回结构化结果，方便主线程汇总成功与失败
# ==========================================
def process_one_file(input_path: Path, output_dir: Path) -> dict:
    output_path = output_dir / input_path.name
    command = [
        sys.executable,
        str(CLIPPING_SCRIPT),
        str(input_path),
        "--output",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        return {
            "input_path": input_path,
            "output_path": output_path,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {
            "input_path": input_path,
            "output_path": output_path,
            "returncode": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


# ==========================================
# 3. 带重试的任务包装器
#    - 单文件最多执行 1 次重试
#    - 任意一次成功就直接返回
#    - 如果最终失败，保留最后一次错误输出
# ==========================================
def process_one_file_with_retry(
    input_path: Path,
    output_dir: Path,
    task_number: int,
    total_files: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    print(f"[{task_number}/{total_files}] 开始处理: {input_path.name}", flush=True)

    last_result = None

    for attempt in range(max_retries + 1):
        result = process_one_file(input_path, output_dir)
        result["attempt"] = attempt + 1
        result["max_attempts"] = max_retries + 1
        result["task_number"] = task_number
        result["total_files"] = total_files

        if result["returncode"] == 0:
            result["retried"] = attempt > 0
            return result

        last_result = result

    last_result["retried"] = max_retries > 0
    return last_result


# ==========================================
# 4. 收集输入文件
#    - 只处理输入目录下的 .json 文件
#    - 按文件名排序，便于日志和复跑时保持稳定顺序
# ==========================================
def collect_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"未找到输入目录: {input_dir}")

    input_files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"输入目录下没有 JSON 文件: {input_dir}")

    return input_files


# ==========================================
# 5. 进度输出工具
#    - 统一格式化批处理进度，便于观察当前完成比例
#    - 显示成功数、失败数、重试后成功数和剩余数
# ==========================================
def format_progress_line(
    completed_count: int,
    total_files: int,
    success_count: int,
    failed_count: int,
    retried_success_count: int,
) -> str:
    progress_pct = (completed_count / total_files) * 100 if total_files else 0
    remaining_count = total_files - completed_count
    return (
        f"[{completed_count}/{total_files} | {progress_pct:.1f}%] "
        f"成功 {success_count} | 失败 {failed_count} | "
        f"重试后成功 {retried_success_count} | 剩余 {remaining_count}"
    )


# ==========================================
# 6. 批量执行主流程
#    - 用 ThreadPoolExecutor 启动最多 5 个并发任务
#    - 每个任务独立调用现有的 3llm-clipping.py
#    - 主线程持续汇总进度、失败原因和最终统计
# ==========================================
def process_all_files(
    input_dir: Path,
    output_dir: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> int:
    input_files = collect_input_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(input_files)
    success_results = []
    failed_results = []

    print(f"待处理文件数: {total_files}", flush=True)
    print(f"输出目录: {output_dir}", flush=True)
    print(f"并发线程数: {max_workers}", flush=True)
    print(f"失败重试次数: {max_retries}", flush=True)
    print("已提交全部任务，开始并发处理...\n", flush=True)

    task_info_map = {
        input_path: {
            "task_number": index,
            "total_files": total_files,
        }
        for index, input_path in enumerate(input_files, start=1)
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for input_path in input_files:
            task_info = task_info_map[input_path]
            future = executor.submit(
                process_one_file_with_retry,
                input_path,
                output_dir,
                task_info["task_number"],
                task_info["total_files"],
                max_retries,
            )
            future_map[future] = input_path

        completed_count = 0
        retried_success_count = 0
        for future in as_completed(future_map):
            completed_count += 1
            result = future.result()
            input_path = result["input_path"]
            task_info = task_info_map[input_path]

            if result["returncode"] == 0:
                success_results.append(result)
                if result.get("retried"):
                    retried_success_count += 1
                    print(
                        f"[{task_info['task_number']}/{task_info['total_files']}] "
                        f"完成处理: {input_path.name} | 重试后成功 (第 {result['attempt']} 次) | "
                        f"{format_progress_line(completed_count, total_files, len(success_results), len(failed_results), retried_success_count)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{task_info['task_number']}/{task_info['total_files']}] "
                        f"完成处理: {input_path.name} | 成功 | "
                        f"{format_progress_line(completed_count, total_files, len(success_results), len(failed_results), retried_success_count)}",
                        flush=True,
                    )
            else:
                failed_results.append(result)
                print(
                    f"[{task_info['task_number']}/{task_info['total_files']}] "
                    f"完成处理: {input_path.name} | 失败 (已尝试 {result['attempt']}/{result['max_attempts']} 次) | "
                    f"{format_progress_line(completed_count, total_files, len(success_results), len(failed_results), retried_success_count)}",
                    flush=True,
                )
                if result["stderr"].strip():
                    print(result["stderr"].strip(), flush=True)
                elif result["stdout"].strip():
                    print(result["stdout"].strip(), flush=True)

    print("\n批处理完成", flush=True)
    print(f"成功: {len(success_results)}", flush=True)
    print(f"失败: {len(failed_results)}", flush=True)
    print(f"重试后成功: {retried_success_count}", flush=True)

    if failed_results:
        print("\n失败文件列表:", flush=True)
        for result in failed_results:
            print(f"- {result['input_path'].name}", flush=True)
        return 1

    return 0


# ==========================================
# 7. 命令行入口
#    - 可自定义输入目录、输出目录和线程数
#    - 默认值即覆盖当前项目的批处理需求
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并发批量执行视频语义切片")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="输入目录，默认 2cleaned-data",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认 3clipped",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="线程池大小，默认 5",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="单文件失败后的重试次数，默认 1",
    )
    args = parser.parse_args()

    exit_code = process_all_files(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    raise SystemExit(exit_code)
