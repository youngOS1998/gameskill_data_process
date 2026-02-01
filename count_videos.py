#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计指定文件夹下的视频文件数量
"""

import os
from pathlib import Path

def count_videos(folder_path):
    """
    统计文件夹下的视频文件数量
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        tuple: (视频文件总数, 视频文件列表)
    """
    # 常见的视频文件扩展名
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
                       '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'}
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return 0, []
    
    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return 0, []
    
    video_files = []
    
    # 遍历文件夹中的所有文件
    for file_path in folder.iterdir():
        if file_path.is_file():
            # 检查文件扩展名
            if file_path.suffix.lower() in video_extensions:
                video_files.append(file_path.name)
    
    return len(video_files), sorted(video_files)

def main():
    # 默认统计当前目录下的 processed_bilibili_cs2 文件夹
    folder_path = './processed_bilibili_cs2'
    
    count, video_files = count_videos(folder_path)
    
    print(f"文件夹: {folder_path}")
    print(f"视频文件总数: {count}")
    print(f"\n视频文件列表:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")

if __name__ == '__main__':
    main()
