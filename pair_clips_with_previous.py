"""
根据已处理的视频片段描述，找到每个片段的前一段视频，并整合成数据对。

支持两种输入格式：
1. JSON 数组格式 (.json): [{"clip_path": "...", ...}, ...]
2. JSONL 格式 (.jsonl): 每行一个 JSON 对象

使用方法:
    python pair_clips_with_previous.py \
        --input_file clips_final_standard.json \
        --clips_dir video_clips_6_15s \
        --output_file clip_pairs.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def get_args():
    parser = argparse.ArgumentParser(
        description='将已处理的视频片段与前一段整合成数据对'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='clips_final_standard_all.json',
        help='已处理的片段描述文件（支持JSON数组格式或JSONL格式）'
    )
    parser.add_argument(
        '--clips_dir',
        type=str,
        default='/root/autodl-tmp/Projects/Qwen3-VL/qwen-vl-finetune/dataset/video_clips_6_15s',
        help='视频片段目录'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='clip_pairs_all.jsonl',
        help='输出的数据对文件（JSONL格式）'
    )
    parser.add_argument(
        '--include_missing',
        action='store_true',
        help='是否包含前一段不存在的片段（只输出当前段）'
    )
    return parser.parse_args()


def parse_clip_filename(filename: str) -> Dict[str, Any]:
    """
    解析视频片段文件名，提取信息
    格式: {video_id}__clip{number}_{start}-{end}.mp4
    """
    import re
    name_without_ext = Path(filename).stem
    
    pattern = r'^(.+?)__clip(\d+)_([\d.]+)-([\d.]+)$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        video_id = match.group(1)
        clip_number = int(match.group(2))
        start_time = float(match.group(3))
        end_time = float(match.group(4))
        
        return {
            'video_id': video_id,
            'clip_number': clip_number,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
    else:
        return {
            'video_id': name_without_ext,
            'clip_number': 0,
            'start_time': 0.0,
            'end_time': 0.0,
            'duration': 0.0
        }


def generate_previous_clip_filename(video_id: str, clip_number: int) -> str:
    """
    根据当前片段信息生成前一段的文件名
    
    Args:
        video_id: 视频ID
        clip_number: 当前片段编号
    
    Returns:
        前一段的文件名（不含路径）
    """
    if clip_number <= 1:
        return None
    
    prev_clip_number = clip_number - 1
    # 格式: {video_id}__clip{number:04d}_{start}-{end}.mp4
    # 由于我们不知道前一段的具体时间，需要从目录中查找
    return None  # 返回None表示需要从目录中查找


def find_previous_clip_file(clips_dir: Path, video_id: str, current_clip_number: int) -> Optional[Path]:
    """
    在目录中查找前一段视频文件
    
    Args:
        clips_dir: 视频片段目录
        video_id: 视频ID
        current_clip_number: 当前片段编号
    
    Returns:
        前一段视频文件的Path，如果不存在则返回None
    """
    if current_clip_number <= 1:
        return None
    
    prev_clip_number = current_clip_number - 1
    # 查找格式: {video_id}__clip{number:04d}_*.mp4
    pattern = f"{video_id}__clip{prev_clip_number:04d}_*.mp4"
    
    matches = list(clips_dir.glob(pattern))
    if matches:
        # 应该只有一个匹配
        return matches[0]
    return None


def load_processed_clips(input_file: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    加载已处理的片段描述
    支持两种格式：
    1. JSON 数组格式: [{"clip_path": "...", ...}, ...]
    2. JSONL 格式: 每行一个 JSON 对象
    
    Returns:
        {(video_id, clip_number): clip_data} 的字典
    """
    processed = {}
    
    if not input_file.exists():
        print(f"警告: 输入文件不存在: {input_file}")
        return processed
    
    # 根据文件扩展名判断格式，或尝试解析
    file_ext = input_file.suffix.lower()
    is_json_array = (file_ext == '.json')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if is_json_array:
                # JSON 数组格式
                try:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        print(f"警告: JSON文件不是数组格式，尝试按JSONL格式解析...")
                        # 回退到 JSONL 格式
                        f.seek(0)
                        is_json_array = False
                    else:
                        # 成功解析为数组
                        for item_num, data in enumerate(data_list, 1):
                            if not isinstance(data, dict):
                                continue
                            
                            video_id = data.get('video_id')
                            clip_number = data.get('clip_number')
                            
                            if video_id and clip_number is not None:
                                key = (video_id, clip_number)
                                processed[key] = data
                        
                        print(f"成功加载 JSON 数组格式，共 {len(data_list)} 条记录")
                        return processed
                except json.JSONDecodeError as e:
                    print(f"警告: JSON数组解析失败: {e}，尝试按JSONL格式解析...")
                    f.seek(0)
                    is_json_array = False
            
            # JSONL 格式（每行一个 JSON 对象）
            if not is_json_array:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            continue
                        
                        video_id = data.get('video_id')
                        clip_number = data.get('clip_number')
                        
                        if video_id and clip_number is not None:
                            key = (video_id, clip_number)
                            processed[key] = data
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                        continue
                
                print(f"成功加载 JSONL 格式，共 {len(processed)} 条记录")
    
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        import traceback
        traceback.print_exc()
    
    return processed


def main():
    args = get_args()
    
    input_file = Path(args.input_file)
    clips_dir = Path(args.clips_dir)
    output_file = Path(args.output_file)
    
    if not clips_dir.exists():
        print(f"错误: 视频片段目录不存在: {clips_dir}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"视频片段目录: {clips_dir}")
    print(f"输出文件: {output_file}")
    print(f"包含前一段缺失的片段: {args.include_missing}")
    print("=" * 60)
    
    # 加载已处理的片段
    print("\n正在加载已处理的片段...")
    processed_clips = load_processed_clips(input_file)
    print(f"已加载 {len(processed_clips)} 个已处理的片段")
    
    # 统计信息
    total_pairs = 0
    pairs_with_previous = 0
    pairs_without_previous = 0
    skipped_first_clip = 0
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历已处理的片段
        for (video_id, clip_number), current_clip_data in sorted(processed_clips.items()):
            # 跳过第一个片段（没有前一段）
            if clip_number <= 1:
                skipped_first_clip += 1
                continue
            
            # 查找前一段文件
            prev_clip_path = find_previous_clip_file(clips_dir, video_id, clip_number)
            
            if prev_clip_path is None or not prev_clip_path.exists():
                # 前一段不存在
                if args.include_missing:
                    # 只输出当前段
                    pair_data = {
                        'previous_clip': None,
                        'current_clip': current_clip_data,
                        'has_previous': False
                    }
                    f_out.write(json.dumps(pair_data, ensure_ascii=False) + '\n')
                    pairs_without_previous += 1
                    total_pairs += 1
                continue
            
            # 查找前一段是否已处理（有描述）
            prev_clip_number = clip_number - 1
            prev_clip_key = (video_id, prev_clip_number)
            prev_clip_data = processed_clips.get(prev_clip_key)
            
            if prev_clip_data is None:
                # 前一段存在但未处理，只使用文件路径信息
                prev_clip_filename = prev_clip_path.name
                prev_clip_info = parse_clip_filename(prev_clip_filename)
                prev_clip_data = {
                    'clip_path': str(prev_clip_path),
                    'clip_filename': prev_clip_filename,
                    'video_id': prev_clip_info['video_id'],
                    'clip_number': prev_clip_info['clip_number'],
                    'start_time': prev_clip_info['start_time'],
                    'end_time': prev_clip_info['end_time'],
                    'duration': prev_clip_info['duration'],
                    'description': None  # 未处理，没有描述
                }
            
            # 构建数据对
            pair_data = {
                'previous_clip': prev_clip_data,
                'current_clip': current_clip_data,
                'has_previous': True,
                'previous_has_description': prev_clip_data.get('description') is not None
            }
            
            f_out.write(json.dumps(pair_data, ensure_ascii=False) + '\n')
            pairs_with_previous += 1
            total_pairs += 1
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"总数据对数: {total_pairs}")
    print(f"  包含前一段的数据对: {pairs_with_previous}")
    print(f"  前一段缺失的数据对: {pairs_without_previous}")
    print(f"  跳过的第一个片段: {skipped_first_clip}")
    print(f"输出文件: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
