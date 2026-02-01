#!/bin/bash
# 快速开始脚本 - 使用Qwen3-VL-8B对游戏视频进行描述

echo "=========================================="
echo "游戏视频描述生成工具 - 快速开始"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查是否安装了必要的包
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null || {
    echo "警告: torch 未安装，正在安装依赖..."
    pip install -r requirements_video_description.txt
}

# 设置默认参数
VIDEO_DIR="processed_bilibili_cs2"
OUTPUT_FILE="video_descriptions.jsonl"
MODEL_NAME="Qwen/Qwen2-VL-8B-Instruct"
MAX_VIDEOS=5

# 检查参数
if [ "$1" == "full" ]; then
    echo "使用完整版本..."
    python3 describe_videos_with_qwen.py \
        --video_dir "$VIDEO_DIR" \
        --output_file "$OUTPUT_FILE" \
        --model_name "$MODEL_NAME" \
        --use_subtitle_segments \
        --min_segment_duration 5.0 \
        --max_segment_duration 30.0 \
        --max_videos "$MAX_VIDEOS"
else
    echo "使用简化版本（快速测试）..."
    python3 describe_videos_simple.py \
        --video_dir "$VIDEO_DIR" \
        --output_file "${OUTPUT_FILE%.jsonl}_simple.jsonl" \
        --model_name "$MODEL_NAME" \
        --max_videos "$MAX_VIDEOS"
fi

echo "=========================================="
echo "处理完成！"
echo "=========================================="



