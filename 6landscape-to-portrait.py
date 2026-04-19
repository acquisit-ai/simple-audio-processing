import argparse
import json
import math
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


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
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH")


def get_video_resolution(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]
    result = run_cmd(cmd)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")

    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    return width, height


def make_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def get_ffmpeg_encoders() -> str:
    result = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    return result.stdout + result.stderr


def choose_video_encoder() -> tuple[str, list[str]]:
    system = platform.system().lower()
    encoders = get_ffmpeg_encoders()

    # macOS: VideoToolbox
    if system == "darwin" and "h264_videotoolbox" in encoders:
        return "h264_videotoolbox", ["-b:v", "8M"]

    # Windows: NVIDIA NVENC if available
    if system == "windows" and "h264_nvenc" in encoders:
        return "h264_nvenc", ["-cq", "23", "-preset", "p5"]

    # fallback
    return "libx264", ["-crf", "20", "-preset", "medium"]


def build_filter(output_w: int, output_h: int, blur_sigma: int = 35) -> str:
    # 背景：放大填满 -> 裁切 -> 模糊
    # 前景：按比例缩放到完整放进竖屏画布
    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]"
        f"scale={output_w}:{output_h}:force_original_aspect_ratio=increase,"
        f"crop={output_w}:{output_h},"
        f"gblur=sigma={blur_sigma},"
        f"setsar=1[bg];"
        f"[fgsrc]"
        f"scale={output_w}:{output_h}:force_original_aspect_ratio=decrease,"
        f"setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"format=yuv420p[v]"
    )


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


def collect_existing_video_names(directory: Path) -> set[str]:
    return {path.name for path in collect_supported_video_files(directory)}


def convert_landscape_to_vertical(input_path: str, output_path: str) -> None:
    ensure_ffmpeg_exists()

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    src_w, src_h = get_video_resolution(str(input_file))

    if src_w <= src_h:
        raise ValueError(
            f"Input video is not landscape: {src_w}x{src_h}. "
            "This function only handles landscape videos."
        )

    if src_w > 1080:
        out_w = 1080
        out_h = 1920
    else:
        out_w = make_even(src_w)
        out_h = make_even(math.ceil(src_w / 9 * 16))

    encoder, encoder_args = choose_video_encoder()
    filter_complex = build_filter(out_w, out_h)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        encoder,
        *encoder_args,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        # GPU 编码失败时自动回退到 libx264
        if encoder != "libx264":
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                str(input_file),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_file),
            ]
            run_cmd(fallback_cmd)
        else:
            raise RuntimeError(f"ffmpeg failed:\n{stderr}") from e


def run_batch_convert(
    source_dir: Path,
    target_dir: Path,
    converter: Callable[[str, str], None] = convert_landscape_to_vertical,
) -> dict[str, int]:
    source_files = collect_supported_video_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_names = collect_existing_video_names(target_dir)

    processed = 0
    skipped = 0
    failed = 0

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"源视频数: {len(source_files)}")
    print(f"目标已存在视频数: {len(existing_names)}")

    if not source_files:
        print("没有找到可处理的视频文件。")
        return {
            "total_source_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }

    for index, source_path in enumerate(source_files, start=1):
        target_path = target_dir / source_path.name
        if source_path.name in existing_names:
            skipped += 1
            print(f"[{index}/{len(source_files)}] 跳过: {source_path.name}")
            continue

        try:
            print(f"[{index}/{len(source_files)}] 开始处理: {source_path.name}")
            converter(str(source_path), str(target_path))
            existing_names.add(source_path.name)
            processed += 1
            print(f"[{index}/{len(source_files)}] 处理完成: {source_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(source_files)}] 处理失败: {source_path.name} | {type(exc).__name__}: {exc}")

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
    parser = argparse.ArgumentParser(description="批量将横屏视频转换为竖屏视频。")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="源视频目录，只读取当前目录下的支持视频文件。",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="目标目录；若已存在同名视频则跳过。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_batch_convert(args.source_dir, args.target_dir)
    raise SystemExit(1 if summary["failed"] > 0 else 0)
