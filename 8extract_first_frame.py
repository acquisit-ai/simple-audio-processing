import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional


SUPPORTED_VIDEO_SUFFIXES = {".mp4"}
DEFAULT_SOURCE_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-clips")
DEFAULT_TARGET_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-cover")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def get_cwebp_path() -> Optional[str]:
    return shutil.which("cwebp")


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def collect_supported_video_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    video_files = [path for path in directory.iterdir() if is_supported_video_file(path)]
    return sorted(video_files, key=natural_sort_key)


def collect_existing_webp_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()

    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".webp"
    }


def extract_first_frame_to_webp(input_path: str, output_path: str) -> None:
    ensure_ffmpeg_exists()

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_file),
        "-frames:v",
        "1",
        str(output_file),
    ]

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        cwebp_path = get_cwebp_path()
        if cwebp_path is None:
            raise RuntimeError(f"ffmpeg failed:\n{stderr}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_png_path = Path(temp_dir) / f"{output_file.stem}.png"
            png_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                str(input_file),
                "-frames:v",
                "1",
                str(temp_png_path),
            ]
            try:
                run_cmd(png_cmd)
                run_cmd([
                    cwebp_path,
                    str(temp_png_path),
                    "-o",
                    str(output_file),
                ])
            except subprocess.CalledProcessError as fallback_exc:
                fallback_stderr = fallback_exc.stderr or ""
                raise RuntimeError(
                    "ffmpeg failed to write webp directly, and PNG->cwebp fallback failed:\n"
                    f"{stderr}\n{fallback_stderr}"
                ) from fallback_exc

    if not output_file.exists():
        raise RuntimeError(f"Output webp not created: {output_path}")


def run_batch_extract(
    source_dir: Path,
    target_dir: Path,
    extractor: Callable[[str, str], None] = extract_first_frame_to_webp,
) -> dict[str, int]:
    source_files = collect_supported_video_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_names = collect_existing_webp_names(target_dir)

    processed = 0
    skipped = 0
    failed = 0

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"源视频数: {len(source_files)}")
    print(f"目标已存在 webp 数: {len(existing_names)}")

    if not source_files:
        print("没有找到可处理的视频文件。")
        return {
            "total_source_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }

    for index, source_path in enumerate(source_files, start=1):
        target_name = f"{source_path.stem}.webp"
        target_path = target_dir / target_name
        if target_name in existing_names:
            skipped += 1
            print(f"[{index}/{len(source_files)}] 跳过: {source_path.name}")
            continue

        try:
            print(f"[{index}/{len(source_files)}] 开始处理: {source_path.name}")
            extractor(str(source_path), str(target_path))
            existing_names.add(target_name)
            processed += 1
            print(f"[{index}/{len(source_files)}] 处理完成: {source_path.name}")
        except Exception as exc:
            failed += 1
            print(
                f"[{index}/{len(source_files)}] 处理失败: {source_path.name} | "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n批处理完成")
    print(f"处理成功: {processed}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")

    return {
        "total_source_files": len(source_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量提取视频第一帧为 WebP。")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="源视频目录，只读取当前目录下的 MP4 文件，默认 /Volumes/Dingzhen/STT/The Office BD-clips。",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="目标目录；若已存在同名 .webp 则跳过，默认 /Volumes/Dingzhen/STT/The Office BD-cover。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_batch_extract(args.source_dir, args.target_dir)
    raise SystemExit(1 if summary["failed"] > 0 else 0)
