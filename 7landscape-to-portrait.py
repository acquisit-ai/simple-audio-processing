import argparse
import json
import math
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable


DEFAULT_SOURCE_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-clips")
DEFAULT_TARGET_DIR = Path("/Volumes/Dingzhen/STT/The Office BD-portrait")
DEFAULT_MAX_WORKERS = 3
SUPPORTED_VIDEO_SUFFIXES = {".mp4"}
DEFAULT_OUTPUT_HEIGHT = 1280
DEFAULT_VIDEO_BITRATE = "1500k"
DEFAULT_MAXRATE = "2200k"
DEFAULT_BUFSIZE = "4000k"
DEFAULT_AUDIO_BITRATE = "256k"
DEFAULT_AUDIO_SAMPLE_RATE = 48000
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_GOP_SIZE = 48


def build_videotoolbox_encoder_args(
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> list[str]:
    return [
        "-b:v", video_bitrate,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-g", str(gop_size),
        "-tag:v", "hvc1",
        "-spatial_aq", "1",
    ]


def build_libx265_encoder_args(
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> list[str]:
    return [
        "-b:v", video_bitrate,
        "-preset", "medium",
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-g", str(gop_size),
        "-tag:v", "hvc1",
    ]


VIDEOTOOLBOX_ENCODER_ARGS = build_videotoolbox_encoder_args()
LIBX265_ENCODER_ARGS = build_libx265_encoder_args()


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


def choose_video_encoder(
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> tuple[str, list[str]]:
    system = platform.system().lower()
    encoders = get_ffmpeg_encoders()

    # macOS: VideoToolbox
    if system == "darwin" and "hevc_videotoolbox" in encoders:
        return "hevc_videotoolbox", build_videotoolbox_encoder_args(
            video_bitrate=video_bitrate,
            maxrate=maxrate,
            bufsize=bufsize,
            gop_size=gop_size,
        )

    # Windows: NVIDIA NVENC if available
    if system == "windows" and "hevc_nvenc" in encoders:
        return "hevc_nvenc", [
            "-b:v", video_bitrate,
            "-maxrate", maxrate,
            "-bufsize", bufsize,
            "-g", str(gop_size),
            "-tag:v", "hvc1",
        ]

    # fallback
    return "libx265", build_libx265_encoder_args(
        video_bitrate=video_bitrate,
        maxrate=maxrate,
        bufsize=bufsize,
        gop_size=gop_size,
    )


def build_output_size(
    source_width: int,
    source_height: int,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
) -> tuple[int, int]:
    output_width = make_even(math.ceil(output_height * 9 / 16))
    return output_width, make_even(output_height)


def build_filter(output_w: int, output_h: int, blur_sigma: int = 35) -> str:
    # 背景：放大填满 -> 裁切 -> 模糊
    # 前景：按比例缩放到完整放进竖屏画布
    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]"
        f"scale={output_w}:{output_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={output_w}:{output_h},"
        f"gblur=sigma={blur_sigma},"
        f"setsar=1[bg];"
        f"[fgsrc]"
        f"scale={output_w}:{output_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
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


def convert_landscape_to_vertical(
    input_path: str,
    output_path: str,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE,
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    audio_channels: int = DEFAULT_AUDIO_CHANNELS,
    gop_size: int = DEFAULT_GOP_SIZE,
) -> None:
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

    out_w, out_h = build_output_size(src_w, src_h, output_height=output_height)

    encoder, encoder_args = choose_video_encoder(
        video_bitrate=video_bitrate,
        maxrate=maxrate,
        bufsize=bufsize,
        gop_size=gop_size,
    )
    filter_complex = build_filter(out_w, out_h)
    audio_encoder_args = [
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-ac", str(audio_channels),
    ]

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
        *audio_encoder_args,
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        # GPU 编码失败时自动回退到 libx265
        if encoder != "libx265":
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
                "libx265",
                *build_libx265_encoder_args(
                    video_bitrate=video_bitrate,
                    maxrate=maxrate,
                    bufsize=bufsize,
                    gop_size=gop_size,
                ),
                *audio_encoder_args,
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
    max_workers: int = DEFAULT_MAX_WORKERS,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    maxrate: str = DEFAULT_MAXRATE,
    bufsize: str = DEFAULT_BUFSIZE,
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE,
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    audio_channels: int = DEFAULT_AUDIO_CHANNELS,
    gop_size: int = DEFAULT_GOP_SIZE,
    converter: Callable[[str, str, int, str, str, str, str, int, int, int], None] = convert_landscape_to_vertical,
) -> dict[str, int]:
    if max_workers < 1:
        raise ValueError("max_workers 必须大于等于 1")

    source_files = collect_supported_video_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_names = collect_existing_video_names(target_dir)

    processed = 0
    skipped = 0
    failed = 0

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"源 MP4 数: {len(source_files)}")
    print(f"目标已存在 MP4 数: {len(existing_names)}")
    print(f"并发数: {max_workers}")

    if not source_files:
        print("没有找到可处理的 MP4 文件。")
        return {
            "total_source_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }

    jobs = []
    for index, source_path in enumerate(source_files, start=1):
        target_path = target_dir / source_path.name
        if source_path.name in existing_names:
            skipped += 1
            print(f"[{index}/{len(source_files)}] 跳过: {source_path.name}")
            continue

        jobs.append((index, source_path, target_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                converter,
                str(source_path),
                str(target_path),
                output_height,
                video_bitrate,
                maxrate,
                bufsize,
                audio_bitrate,
                audio_sample_rate,
                audio_channels,
                gop_size,
            ): (index, source_path, target_path)
            for index, source_path, target_path in jobs
        }

        for future in as_completed(future_to_job):
            index, source_path, target_path = future_to_job[future]
            try:
                future.result()
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
        default=DEFAULT_SOURCE_DIR,
        help="源视频目录，只读取当前目录下的 MP4 文件，默认 /Volumes/Dingzhen/STT/The Office BD-clips。",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="目标目录；若已存在同名视频则跳过，默认 /Volumes/Dingzhen/STT/The Office BD-portrait。",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="并发处理数，默认 3。")
    parser.add_argument("--height", type=int, default=DEFAULT_OUTPUT_HEIGHT, help="输出竖屏高度，默认 1280。")
    parser.add_argument("--video-bitrate", default=DEFAULT_VIDEO_BITRATE, help="VideoToolbox HEVC 目标视频码率，默认 1500k。")
    parser.add_argument("--maxrate", default=DEFAULT_MAXRATE, help="视频码率上限，默认 2200k。")
    parser.add_argument("--bufsize", default=DEFAULT_BUFSIZE, help="码率控制 buffer size，默认 4000k。")
    parser.add_argument("--audio-bitrate", default=DEFAULT_AUDIO_BITRATE, help="AAC 音频码率，默认 256k。")
    parser.add_argument("--audio-sample-rate", type=int, default=DEFAULT_AUDIO_SAMPLE_RATE, help="音频采样率，默认 48000。")
    parser.add_argument("--audio-channels", type=int, default=DEFAULT_AUDIO_CHANNELS, help="音频声道数，默认 2。")
    parser.add_argument("--gop-size", type=int, default=DEFAULT_GOP_SIZE, help="关键帧间隔，默认 48。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_batch_convert(
        args.source_dir,
        args.target_dir,
        max_workers=args.max_workers,
        output_height=args.height,
        video_bitrate=args.video_bitrate,
        maxrate=args.maxrate,
        bufsize=args.bufsize,
        audio_bitrate=args.audio_bitrate,
        audio_sample_rate=args.audio_sample_rate,
        audio_channels=args.audio_channels,
        gop_size=args.gop_size,
    )
    raise SystemExit(1 if summary["failed"] > 0 else 0)
