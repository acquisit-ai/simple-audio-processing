import argparse
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


def get_ffmpeg_encoders() -> str:
    result = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    return result.stdout + result.stderr


def choose_video_encoder(use_gpu: bool = True) -> tuple[str, list[str]]:
    if not use_gpu:
        return "libx264", ["-preset", "medium"]

    system = platform.system().lower()
    encoders = get_ffmpeg_encoders()

    if system == "darwin" and "h264_videotoolbox" in encoders:
        return "h264_videotoolbox", ["-tag:v", "avc1"]

    if "h264_nvenc" in encoders:
        return "h264_nvenc", ["-preset", "fast"]

    return "libx264", ["-preset", "medium"]


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


def build_hls_output_dir(source_path: Path, target_root: Path) -> Path:
    return target_root / f"{source_path.stem}_hls"


def is_hls_bundle_complete(output_dir: Path) -> bool:
    return (output_dir / "playlist.m3u8").exists() and (output_dir / "init.mp4").exists()


def build_hls_ffmpeg_command(
    input_file: Path,
    output_dir: Path,
    video_codec: str,
    encoder_opts: list[str],
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_file),
        "-c:v",
        video_codec,
        *encoder_opts,
        "-c:a",
        "aac",
        "-b:v",
        "3000k",
        "-b:a",
        "128k",
        "-hls_time",
        "6",
        "-hls_playlist_type",
        "vod",
        "-hls_segment_type",
        "fmp4",
        "-hls_fmp4_init_filename",
        "init.mp4",
        "-hls_segment_filename",
        str(output_dir / "segment_%03d.m4s"),
        str(output_dir / "playlist.m3u8"),
    ]


def convert_video_to_hls(input_path: str, output_dir: str, use_gpu: bool = True) -> None:
    ensure_ffmpeg_exists()

    input_file = Path(input_path)
    output_path = Path(output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    video_codec, encoder_opts = choose_video_encoder(use_gpu=use_gpu)
    cmd = build_hls_ffmpeg_command(input_file, output_path, video_codec, encoder_opts)

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if use_gpu and video_codec != "libx264":
            fallback_codec, fallback_opts = choose_video_encoder(use_gpu=False)
            fallback_cmd = build_hls_ffmpeg_command(
                input_file,
                output_path,
                fallback_codec,
                fallback_opts,
            )
            try:
                run_cmd(fallback_cmd)
            except subprocess.CalledProcessError as fallback_exc:
                fallback_stderr = fallback_exc.stderr or ""
                raise RuntimeError(
                    "ffmpeg failed with GPU encoder and CPU fallback:\n"
                    f"{stderr}\n{fallback_stderr}"
                ) from fallback_exc
        else:
            raise RuntimeError(f"ffmpeg failed:\n{stderr}") from exc

    if not is_hls_bundle_complete(output_path):
        raise RuntimeError(f"Incomplete HLS bundle: {output_dir}")


def run_batch_convert(
    source_dir: Path,
    target_dir: Path,
    use_gpu: bool = True,
    converter: Callable[[str, str, bool], None] = convert_video_to_hls,
) -> dict[str, int]:
    source_files = collect_supported_video_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"源视频数: {len(source_files)}")

    if not source_files:
        print("没有找到可处理的视频文件。")
        return {
            "total_source_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }

    for index, source_path in enumerate(source_files, start=1):
        output_dir = build_hls_output_dir(source_path, target_dir)
        if is_hls_bundle_complete(output_dir):
            skipped += 1
            print(f"[{index}/{len(source_files)}] 跳过: {source_path.name}")
            continue

        try:
            print(f"[{index}/{len(source_files)}] 开始处理: {source_path.name}")
            converter(str(source_path), str(output_dir), use_gpu)
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
    parser = argparse.ArgumentParser(description="批量将视频转换为 fMP4 HLS bundle。")
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
        help="目标目录；每个视频输出为一个 <stem>_hls 目录。",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="优先使用硬件编码；失败时自动回退到 CPU。",
    )
    gpu_group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="只使用 CPU 编码。",
    )
    parser.set_defaults(use_gpu=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_batch_convert(args.source_dir, args.target_dir, use_gpu=args.use_gpu)
    raise SystemExit(1 if summary["failed"] > 0 else 0)
