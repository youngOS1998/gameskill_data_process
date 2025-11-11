"""
游戏解说视频数据转换脚本
将fp_clips.jsonl转换为Qwen2.5-VL训练格式
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse


def format_timestamp(seconds: float) -> str:
    """将秒数转换为HH:MM:SS格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_video_filename(video_id: str, start_time: float, end_time: float) -> str:
    """根据视频ID和时间戳生成视频文件名"""
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    # 如果video_id已经以下划线结尾，只添加一个下划线
    if video_id.endswith('_'):
        return f"{video_id}_{start_ms}_{end_ms}.mp4"
    else:
        return f"{video_id}__{start_ms}_{end_ms}.mp4"


def create_game_instruction_prompts() -> List[str]:
    """创建多样化的游戏指导提示词"""
    prompts = [
        "请观看这个游戏视频片段，并给出专业的游戏解说和分析。",
        "请分析这个游戏片段中的战术策略和操作技巧。",
        "请详细解说这个游戏视频中的关键操作和决策。",
        "请从专业角度分析这个游戏片段的亮点和技巧。",
        "请观看视频并解释游戏中的战术运用和团队配合。",
        "请分析这个游戏片段中的操作细节和策略思路。",
        "请解说这个游戏视频中的精彩操作和战术分析。",
        "请从游戏专业角度分析这个片段的技巧和策略。",
        "请详细分析这个游戏视频中的操作技巧和战术运用。",
        "请观看并解说这个游戏片段中的关键操作和决策过程。"
    ]
    return prompts


def convert_fp_clips_to_qwen_format(
    input_file: str,
    output_file: str,
    video_base_path: str = "videos_val",
    use_context: bool = True,
    max_context_length: int = 200
) -> None:
    """
    将fp_clips.jsonl转换为Qwen2.5-VL训练格式
    
    Args:
        input_file: 输入的fp_clips.jsonl文件路径
        output_file: 输出的训练数据文件路径
        video_base_path: 视频文件的基础路径
        use_context: 是否使用之前的解说文字作为上下文
        max_context_length: 上下文的最大长度
    """
    
    # 获取提示词列表
    instruction_prompts = create_game_instruction_prompts()
    
    converted_data = []
    processed_count = 0
    skipped_count = 0
    
    print(f"开始转换数据: {input_file}")
    print(f"视频基础路径: {video_base_path}")
    print(f"使用上下文: {use_context}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取数据
                video_id = data.get('video', '')
                content = data.get('content', [])
                previous = data.get('previous', '')
                title = data.get('title', '')
                category = data.get('category', '')
                
                # 验证数据格式
                if not isinstance(content, list):
                    print(f"警告: 第{line_num}行数据格式不正确，跳过")
                    skipped_count += 1
                    continue
                
                start_time, end_time, commentary = content
                
                # 生成视频文件名
                video_filename = generate_video_filename(video_id, start_time, end_time)
                video_path = video_filename  # 直接使用文件名，不包含路径前缀
                
                # 构建对话内容
                if use_context and previous and len(previous) > 10:
                    # 截断过长的上下文
                    if len(previous) > max_context_length:
                        previous = "..." + previous[len(previous) - max_context_length:]
                    
                    # 构建包含上下文的提示
                    instruction = f"基于之前的解说内容：\"{previous}\"\n\n请观看这个游戏视频片段，并给出专业的游戏解说和分析。"
                else:
                    # 随机选择一个基础提示词
                    import random
                    instruction = random.choice(instruction_prompts)
                
                # 构建Qwen2.5-VL格式的数据
                qwen_data = {
                    "video": "videos_val/" + video_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<video>\n{instruction}"
                        },
                        {
                            "from": "gpt", 
                            "value": commentary
                        }
                    ]
                }
                
                # 添加元数据（可选）
                if title or category:
                    qwen_data["metadata"] = {
                        "title": title,
                        "category": category,
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time
                    }
                
                converted_data.append(qwen_data)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第{line_num}行JSON解析失败: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"错误: 第{line_num}行处理失败: {e}")
                skipped_count += 1
                continue
    
    # 保存转换后的数据
    print(f"\n开始保存数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n转换完成!")
    print(f"成功处理: {processed_count} 条数据")
    print(f"跳过数据: {skipped_count} 条")
    print(f"输出文件: {output_file}")


def validate_video_files(data_file: str, video_base_path: str) -> None:
    """验证视频文件是否存在"""
    print(f"\n验证视频文件存在性...")
    
    missing_files = []
    existing_files = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                video_path = data.get('video', '')
                
                if video_path:
                    full_path = os.path.join(video_base_path, video_path)
                    if os.path.exists(full_path):
                        existing_files.append(video_path)
                    else:
                        missing_files.append(video_path)
                        
            except Exception as e:
                continue
    
    print(f"存在的视频文件: {len(existing_files)}")
    print(f"缺失的视频文件: {len(missing_files)}")
    
    if missing_files:
        print(f"\n缺失的视频文件示例 (前10个):")
        for i, file in enumerate(missing_files[:10]):
            print(f"  {i+1}. {file}")
        
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")


def main():
    parser = argparse.ArgumentParser(description='将游戏解说数据转换为Qwen2.5-VL训练格式')
    parser.add_argument('--input', '-i', default='fp_clips.jsonl', 
                       help='输入的fp_clips.jsonl文件路径')
    parser.add_argument('--output', '-o', default='game_commentary_dataset.jsonl',
                       help='输出的训练数据文件路径')
    parser.add_argument('--video-path', '-v', default='videos_val',
                       help='视频文件的基础路径')
    parser.add_argument('--no-context', action='store_true',
                       help='不使用之前的解说文字作为上下文')
    parser.add_argument('--max-context', type=int, default=200,
                       help='上下文的最大长度')
    parser.add_argument('--validate', action='store_true',
                       help='验证视频文件是否存在')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 转换数据
    convert_fp_clips_to_qwen_format(
        input_file=args.input,
        output_file=args.output,
        video_base_path=args.video_path,
        use_context=not args.no_context,
        max_context_length=args.max_context
    )
    
    # 验证视频文件
    if args.validate:
        validate_video_files(args.output, args.video_path)


if __name__ == "__main__":
    main()