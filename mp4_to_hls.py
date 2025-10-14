#!/usr/bin/env python3
import os
import subprocess
import sys

def convert_to_fmp4_hls(input_file, output_dir=None, use_gpu=False):
    """
    将 MP4 视频转换为 fMP4-HLS 格式

    Args:
        input_file: 输入的 MP4 文件路径
        output_dir: 输出目录，默认为输入文件同目录下的 hls 文件夹
        use_gpu: 是否使用 GPU 加速（macOS 使用 VideoToolbox，其他系统使用 NVIDIA）
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    # 设置输出目录
    if output_dir is None:
        base_dir = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(base_dir, f"{filename}_hls")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件路径
    output_playlist = os.path.join(output_dir, "playlist.m3u8")

    # 根据是否使用 GPU 选择编码器
    if use_gpu:
        # 检测操作系统，选择合适的硬件编码器
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            video_codec = "h264_videotoolbox"  # Apple VideoToolbox
            # VideoToolbox 只需要 tag，不支持 profile/level 参数
            encoder_opts = [
                "-tag:v", "avc1"            # 兼容性标签（HLS 需要）
            ]
        else:  # Linux/Windows - 使用 NVIDIA
            video_codec = "h264_nvenc"
            encoder_opts = ["-preset", "fast"]  # NVENC 预设

        print(f"使用 GPU 加速: {video_codec}")
    else:
        video_codec = "libx264"
        encoder_opts = ["-preset", "medium"]  # x264 预设
        print("使用 CPU 编码")

    # FFmpeg 命令：转换为 fMP4-HLS
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", video_codec,         # 视频编码器
        *encoder_opts,               # 编码器选项
        "-c:a", "aac",               # 音频编码器
        "-b:v", "3000k",             # 视频比特率（移动端流媒体优化）
        "-b:a", "128k",              # 音频比特率
        "-hls_time", "6",            # 每个分片的时长（秒）
        "-hls_playlist_type", "vod", # 播放列表类型
        "-hls_segment_type", "fmp4", # 使用 fMP4 格式
        "-hls_fmp4_init_filename", "init.mp4",  # 初始化分片文件名
        "-hls_segment_filename", os.path.join(output_dir, "segment_%03d.m4s"),
        output_playlist
    ]

    print(f"开始转换: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"执行命令: {' '.join(ffmpeg_cmd)}\n")

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"\n转换成功!")
        print(f"播放列表文件: {output_playlist}")
    except subprocess.CalledProcessError as e:
        print(f"\n转换失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    input_video = "original-media/1.mp4"
    # 设置 use_gpu=True 启用 GPU 加速（macOS 使用 VideoToolbox，Linux/Windows 使用 NVENC）
    convert_to_fmp4_hls(input_video, use_gpu=True)
    input_video = "original-media/2.mp4"
    # 设置 use_gpu=True 启用 GPU 加速（macOS 使用 VideoToolbox，Linux/Windows 使用 NVENC）
    convert_to_fmp4_hls(input_video, use_gpu=True)    
    input_video = "original-media/3.mp4"
    # 设置 use_gpu=True 启用 GPU 加速（macOS 使用 VideoToolbox，Linux/Windows 使用 NVENC）
    convert_to_fmp4_hls(input_video, use_gpu=True)    
    input_video = "original-media/4.mp4"
    # 设置 use_gpu=True 启用 GPU 加速（macOS 使用 VideoToolbox，Linux/Windows 使用 NVENC）
    convert_to_fmp4_hls(input_video, use_gpu=True)