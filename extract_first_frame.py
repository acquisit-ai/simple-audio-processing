#!/usr/bin/env python3
"""
从HLS视频流中提取第一帧并保存为WebP格式
支持传统 .ts 和现代 fMP4 (.m4s) 格式
最小化网络和计算代价的实现

【设计理念：稳定性优先】

本脚本采用"让ffmpeg直接读取m3u8 URL"的方案，相比"固定下载1MB数据"的方案具有以下优势：

1. 自动处理fMP4格式复杂性
   - 固定1MB方案：手动下载segment_000.m4s前1MB → 缺少init.mp4 → 解码失败
   - 本方案：ffmpeg自动下载init.mp4 + 智能读取segment → 稳定解码

2. 自适应数据量
   - 固定1MB方案：低码率视频浪费带宽，高码率视频数据不足
   - 本方案：ffmpeg流式读取，找到第一帧立即停止，适配任意码率

3. 零配置，广泛兼容
   - 无需针对不同视频调整参数
   - 支持任意GOP大小、任意码率、任意分辨率

使用示例：
    python3 extract_first_frame.py
    python3 extract_first_frame.py -u https://example.com/video.m3u8 -o thumb.webp -q 95
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from urllib.parse import urlparse


def extract_video_name_from_url(m3u8_url):
    """
    从m3u8 URL中提取视频名称，用于自动命名输出文件

    示例：
        https://.../demo-hls/001_hls/playlist.m3u8 → 001_hls
        https://.../videos/my_video/index.m3u8 → my_video
        https://.../demo.m3u8 → demo

    Args:
        m3u8_url: m3u8播放列表的URL

    Returns:
        str: 提取的视频名称
    """
    # 解析URL路径
    parsed = urlparse(m3u8_url)
    path = parsed.path.rstrip('/')

    # 移除文件名（playlist.m3u8, index.m3u8等）
    path_parts = [p for p in path.split('/') if p]

    if len(path_parts) >= 2:
        # 取倒数第二个部分作为视频名称（通常是目录名）
        # 例如：001_hls/playlist.m3u8 → 001_hls
        video_name = path_parts[-2]
    elif len(path_parts) == 1:
        # 如果只有文件名，提取文件名（去掉扩展名）
        # 例如：demo.m3u8 → demo
        video_name = path_parts[0].replace('.m3u8', '')
    else:
        # 降级方案：使用时间戳
        import time
        video_name = f"video_{int(time.time())}"

    # 清理文件名（移除不安全字符）
    video_name = video_name.replace(' ', '_')

    return video_name


def ensure_webp_directory():
    """
    确保webp目录存在，如果不存在则创建

    Returns:
        Path: webp目录的Path对象
    """
    webp_dir = Path('webp')
    webp_dir.mkdir(exist_ok=True)
    return webp_dir


def extract_first_frame_from_hls(m3u8_url, output_path, quality=80):
    """
    使用ffmpeg直接从HLS的m3u8 URL中提取第一帧并保存为WebP

    【核心稳定性优势】

    相比"固定下载前1MB数据"的方案，本方案完全避免了以下不稳定问题：

    ❌ 固定1MB方案的致命缺陷：
    1. fMP4格式依赖：segment_000.m4s单独无法解码，必须配合init.mp4
       - 固定下载1MB segment数据，缺少init.mp4 → 解码失败

    2. 数据量不可预测：
       - 低码率视频：第一帧可能在50KB位置 → 1MB浪费950KB
       - 高码率视频：第一帧可能在1.5MB位置 → 1MB数据不足，解码失败
       - I帧位置取决于编码器设置(GOP大小)，无法用固定值覆盖所有情况

    3. 需要临时文件：
       - 下载到本地 → ffmpeg读取 → 删除临时文件
       - 增加磁盘I/O开销和错误处理复杂度

    ✅ 本方案的稳定性保障：

    1. ffmpeg自动处理fMP4格式：
       - 解析m3u8，发现 #EXT-X-MAP:URI="init.mp4"
       - 自动下载init.mp4 (通常1-2KB)
       - 自动与segment数据合并解码

    2. 智能流式读取，自适应数据量：
       ffmpeg内部流程：
       ├─ 发送HTTP Range请求到segment_000.m4s
       ├─ 边接收数据边解码（流式处理）
       ├─ 找到第一个I帧 → 解码成功
       └─ 立即关闭TCP连接（自动停止接收剩余数据）

       实际网络开销示例：
       - init.mp4: 1.4KB (必须完整下载)
       - segment_000.m4s: 2.4MB总大小，实际只读取到第一帧位置就停止
         (具体读取量取决于视频码率，通常50KB-500KB)

    3. 零临时文件：
       网络流 → ffmpeg内存处理 → 直接输出webp

    4. 完整异常处理：
       - subprocess.CalledProcessError: 捕获ffmpeg执行失败
       - FileNotFoundError: 捕获ffmpeg未安装
       - 输出文件校验: 确保webp文件生成成功

    5. 广泛兼容性：
       - ✓ 传统HLS (.ts片段)
       - ✓ 现代fMP4 HLS (.m4s片段 + init.mp4)
       - ✓ 任意码率视频 (50kbps - 50Mbps+)
       - ✓ 任意GOP设置 (关键帧间隔1秒-10秒)

    Args:
        m3u8_url: HLS播放列表的URL
        output_path: 输出的WebP文件路径
        quality: WebP质量 (0-100，默认80为高质量)

    Returns:
        bool: 提取成功返回True，失败返回False
    """
    print(f"🎯 HLS URL: {m3u8_url}")
    print(f"📁 输出路径: {output_path}")
    print(f"🎨 WebP质量: {quality}")
    print("-" * 60)

    # ffmpeg命令参数说明：
    #
    # 核心参数：
    # -i {m3u8_url}: 输入HLS流（直接从URL读取）
    #   ↓ ffmpeg会自动：
    #   1. 解析m3u8播放列表
    #   2. 下载init.mp4初始化片段 (fMP4格式必需)
    #   3. 流式读取第一个segment，找到第一个I帧后立即停止
    #   这是稳定性的关键：让ffmpeg处理所有复杂性，而非手动猜测需要下载多少字节
    #
    # -frames:v 1: 只提取1帧视频
    #   ↓ 告诉ffmpeg解码到第一帧后立即停止
    #   这确保了最小的网络和计算开销
    #
    # -q:v {quality}: WebP质量（0-100，建议80-95）
    # -loglevel error: 只显示错误信息
    # -y: 覆盖已存在的文件
    ffmpeg_command = [
        'ffmpeg',
        '-i', m3u8_url,              # 直接从m3u8 URL读取（关键！）
        '-frames:v', '1',             # 只提取第一帧（最小开销）
        '-q:v', str(quality),         # WebP质量
        '-loglevel', 'error',         # 减少日志输出
        '-y',                         # 覆盖输出文件
        output_path
    ]

    try:
        print("⏳ 正在提取第一帧...")

        # 执行ffmpeg命令
        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # 检查输出文件是否创建成功
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print("-" * 60)
            print(f"✅ 第一帧提取成功！")
            print(f"📊 文件大小: {file_size / 1024:.2f} KB")
            print(f"📍 保存位置: {os.path.abspath(output_path)}")
            return True
        else:
            print("❌ 输出文件未创建")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg执行失败:")
        error_msg = e.stderr.decode().strip()
        if error_msg:
            print(error_msg)
        else:
            print("未知错误")
        return False

    except FileNotFoundError:
        print("❌ 未找到ffmpeg，请先安装ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        print("   Windows: 从 https://ffmpeg.org/download.html 下载")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从HLS视频流中提取第一帧并保存为WebP格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认URL，自动命名并保存到webp目录
  python3 extract_first_frame.py
  # 输出: webp/001_hls.webp

  # 指定自定义URL，自动从URL提取名称
  python3 extract_first_frame.py -u https://example.com/my_video/playlist.m3u8
  # 输出: webp/my_video.webp

  # 手动指定输出路径和质量
  python3 extract_first_frame.py -u URL -o custom/path.webp -q 95
        """
    )

    parser.add_argument(
        '-u', '--url',
        default='https://storage.googleapis.com/demo-vedios-cyberdinzhen/demo-hls/009_hls/playlist.m3u8',
        help='HLS视频的m3u8 URL (默认: demo视频)'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='输出文件路径 (默认: 自动创建webp目录并根据URL命名，如webp/001_hls.webp)'
    )

    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=80,
        choices=range(0, 101),
        metavar='[0-100]',
        help='WebP质量 0-100 (默认: 80)'
    )

    args = parser.parse_args()

    # 如果未指定输出路径，自动生成
    if args.output is None:
        # 确保webp目录存在
        webp_dir = ensure_webp_directory()

        # 从URL提取视频名称
        video_name = extract_video_name_from_url(args.url)

        # 生成输出路径
        output_path = webp_dir / f"{video_name}.webp"
        print("=" * 60)
        print("🎬 HLS视频第一帧提取工具")
        print("=" * 60)
        print(f"📂 自动创建输出目录: {webp_dir}/")
        print(f"📝 自动命名: {video_name}.webp (从URL提取)")
        print("=" * 60)
    else:
        output_path = args.output
        print("=" * 60)
        print("🎬 HLS视频第一帧提取工具")
        print("=" * 60)

    success = extract_first_frame_from_hls(
        args.url,
        str(output_path),
        args.quality
    )

    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
