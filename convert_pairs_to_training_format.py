"""
将 clip_pairs_test.jsonl 转换为 Qwen3-VL-8B 微调训练格式

输入格式 (clip_pairs_test.jsonl):
{
  "previous_clip": {
    "clip_path": "...",
    "description": "...",  // 之前的解说内容
    ...
  },
  "current_clip": {
    "clip_path": "...",  // 当前视频
    "description": "...",  // 作为 gpt 的 value
    ...
  }
}

输出格式 (gameskill_1_train.jsonl):
{
  "video": "...",
  "data_path": "...",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\n视频标题: ...\n类别: ...\n之前的解说内容: ...\n请观看视频并给出解说: ..."
    },
    {
      "from": "gpt",
      "value": "..."  // current_clip.description
    }
  ],
  "metadata": {...}
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def get_args():
    parser = argparse.ArgumentParser(
        description='将 clip_pairs 转换为 Qwen3-VL 训练格式'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='clip_pairs_all.jsonl',
        help='输入的 clip_pairs 文件（JSONL格式）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='gameskill_train_all.jsonl',
        help='输出的训练数据文件（JSONL格式）'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/root/autodl-tmp/Projects/qwen_gameskill/qwen-vl-finetune/dataset',
        help='数据路径（用于训练配置）'
    )
    parser.add_argument(
        '--video_base_path',
        type=str,
        default='video_clips_6_15s',
        help='视频文件的基础路径（用于调整视频路径）'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='cs2_web_data',
        help='视频类别'
    )
    parser.add_argument(
        '--max_previous_length',
        type=int,
        default=500,
        help='之前的解说内容最大长度（超过会截断）'
    )
    parser.add_argument(
        '--skip_no_previous',
        action='store_true',
        help='跳过没有前一段描述的数据（previous_clip.description 为空）'
    )
    return parser.parse_args()


def format_previous_description(description: Optional[str], max_length: int = 500) -> str:
    """
    格式化之前的解说内容
    
    Args:
        description: 之前的解说内容
        max_length: 最大长度
    
    Returns:
        格式化后的文本，如果为空则返回 "..."
    """
    if not description or not description.strip():
        return "..."
    
    description = description.strip()
    
    # 如果超过最大长度，截断并添加前缀
    if len(description) > max_length:
        description = "..." + description[-max_length:]
    
    return description


def convert_pair_to_training_format(
    pair_data: Dict[str, Any],
    data_path: str,
    video_base_path: str,
    category: str,
    max_previous_length: int,
    skip_no_previous: bool
) -> Optional[Dict[str, Any]]:
    """
    将一对数据转换为训练格式
    
    Args:
        pair_data: 输入的数据对
        data_path: 数据路径
        video_base_path: 视频基础路径
        category: 视频类别
        max_previous_length: 之前解说内容的最大长度
        skip_no_previous: 是否跳过没有前一段描述的数据
    
    Returns:
        转换后的训练数据，如果跳过则返回 None
    """
    previous_clip = pair_data.get('previous_clip')
    current_clip = pair_data.get('current_clip')
    
    if not current_clip:
        return None
    
    # 检查是否有前一段
    if not previous_clip:
        if skip_no_previous:
            return None
        previous_description = "..."
    else:
        previous_description = previous_clip.get('description')
        if not previous_description and skip_no_previous:
            return None
        previous_description = format_previous_description(previous_description, max_previous_length)
    
    # 获取当前视频信息
    current_video_path = current_clip.get('clip_path', '')
    current_description = current_clip.get('description', '')
    video_id = current_clip.get('video_id', '')
    start_time = current_clip.get('start_time', 0.0)
    end_time = current_clip.get('end_time', 0.0)
    duration = current_clip.get('duration', end_time - start_time)
    
    # 如果当前视频没有描述，跳过
    if not current_description or not current_description.strip():
        return None
    
    # 调整视频路径格式（如果需要）
    # 如果路径已经包含 video_base_path，直接使用；否则添加
    if video_base_path and not current_video_path.startswith(video_base_path):
        # 从 clip_path 中提取文件名，然后组合路径
        from pathlib import Path
        clip_filename = Path(current_video_path).name
        video_path = f"{video_base_path}/{clip_filename}"
    else:
        video_path = current_video_path
    
    # 生成视频标题（使用 video_id）
    title = f"{video_id}_" if video_id else ""
    
    # 构建 human 提示词
    instruction = "详细描述视频内容，并给出专业的游戏解说和分析，包括战术策略、操作技巧和关键决策。"
    
    # human_value = f"<video>\n视频标题: {title}\n类别: {category}"
    human_value = f"<video>\n这是cs2的游戏画面，你现在是我的游戏AI助手，请给我一些指导建议"
    if previous_description and previous_description != "...":
        human_value += f"\n上一个连续视频片段的建议内容: {previous_description}"
    human_value += f"\n请观看这个游戏视频片段: {instruction}"
    
    # 构建对话
    conversations = [
        {
            "from": "human",
            "value": human_value
        },
        {
            "from": "gpt",
            "value": current_description.strip()
        }
    ]
    
    # 构建 metadata
    metadata = {
        "title": title,
        "category": category,
        "video_start": start_time,
        "video_end": end_time,
        "duration": duration,
        "source_file": f"{video_id}.json" if video_id else "unknown.json"
    }
    
    # 构建最终数据
    training_data = {
        "video": video_path,
        "data_path": data_path,
        "conversations": conversations,
        "metadata": metadata
    }
    
    return training_data


def main():
    args = get_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"数据路径: {args.data_path}")
    print(f"视频基础路径: {args.video_base_path}")
    print(f"视频类别: {args.category}")
    print(f"跳过无前一段描述: {args.skip_no_previous}")
    print("=" * 60)
    
    # 统计信息
    total_pairs = 0
    converted_count = 0
    skipped_count = 0
    skipped_no_previous = 0
    skipped_no_current_desc = 0
    
    # 处理数据
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                pair_data = json.loads(line)
                total_pairs += 1
                
                # 转换为训练格式
                training_data = convert_pair_to_training_format(
                    pair_data,
                    args.data_path,
                    args.video_base_path,
                    args.category,
                    args.max_previous_length,
                    args.skip_no_previous
                )
                
                if training_data is None:
                    skipped_count += 1
                    # 判断跳过的原因
                    previous_clip = pair_data.get('previous_clip')
                    current_clip = pair_data.get('current_clip')
                    
                    if not previous_clip or not previous_clip.get('description'):
                        skipped_no_previous += 1
                    elif not current_clip or not current_clip.get('description'):
                        skipped_no_current_desc += 1
                    continue
                
                # 写入输出文件
                f_out.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # 每处理1000条输出一次进度
                if converted_count % 1000 == 0:
                    print(f"已处理 {converted_count} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"总数据对数: {total_pairs}")
    print(f"成功转换: {converted_count}")
    print(f"跳过数量: {skipped_count}")
    if skipped_no_previous > 0:
        print(f"  其中无前一段描述: {skipped_no_previous}")
    if skipped_no_current_desc > 0:
        print(f"  其中无当前描述: {skipped_no_current_desc}")
    print(f"输出文件: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
