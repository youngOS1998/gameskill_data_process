"""
将 clip_descriptions_copy.jsonl（画面描述）与 clip_pairs_all.jsonl（游戏建议）合并为训练数据。

- clip_pairs_all.jsonl: 每行为 (previous_clip, current_clip)，current_clip.description 为游戏建议
- clip_descriptions_copy.jsonl: 每行为单个 clip，description 为画面描述

合并后每条训练数据的 GPT 输出格式：先输出画面描述，再输出游戏建议。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def get_args():
    parser = argparse.ArgumentParser(
        description='合并画面描述与游戏建议为 Qwen3-VL 训练格式'
    )
    parser.add_argument(
        '--pairs_file',
        type=str,
        default='clip_pairs_all.jsonl',
        help='clip_pairs 文件（含游戏建议）'
    )
    parser.add_argument(
        '--descriptions_file',
        type=str,
        default='clip_descriptions_copy.jsonl',
        help='clip 画面描述文件'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='gameskill_merged_train.jsonl',
        help='合并后的训练数据文件'
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
        help='视频文件基础路径'
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
        help='之前的解说内容最大长度'
    )
    parser.add_argument(
        '--desc_label',
        type=str,
        default='【画面描述】',
        help='画面描述部分的标题'
    )
    parser.add_argument(
        '--suggestion_label',
        type=str,
        default='【游戏建议】',
        help='游戏建议部分的标题'
    )
    parser.add_argument(
        '--skip_no_description',
        action='store_true',
        help='若当前 clip 在 descriptions 中无画面描述则跳过'
    )
    return parser.parse_args()


def load_descriptions_index(descriptions_path: Path) -> Dict[Tuple[str, int], str]:
    """
    加载 clip_descriptions_copy.jsonl，建立 (video_id, clip_number) -> description 的索引。
    """
    index = {}
    if not descriptions_path.exists():
        return index

    with open(descriptions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                video_id = data.get('video_id')
                clip_number = data.get('clip_number')
                description = data.get('description') or ''
                if video_id is not None and clip_number is not None:
                    index[(video_id, clip_number)] = description.strip()
            except json.JSONDecodeError:
                continue
    return index


def format_previous_description(description: Optional[str], max_length: int) -> str:
    if not description or not str(description).strip():
        return "..."
    s = str(description).strip()
    if len(s) > max_length:
        s = "..." + s[-max_length:]
    return s


def build_gpt_value(scene_description: str, game_suggestion: str,
                    desc_label: str, suggestion_label: str) -> str:
    """拼接 GPT 输出：画面描述 + 游戏建议。"""
    parts = []
    if scene_description and scene_description.strip():
        parts.append(f"{desc_label}\n{scene_description.strip()}")
    if game_suggestion and game_suggestion.strip():
        parts.append(f"{suggestion_label}\n{game_suggestion.strip()}")
    if not parts:
        return ""
    return "\n\n".join(parts)


def main():
    args = get_args()

    pairs_path = Path(args.pairs_file)
    descriptions_path = Path(args.descriptions_file)
    output_path = Path(args.output_file)

    if not pairs_path.exists():
        print(f"错误: 文件不存在 {pairs_path}")
        return
    if not descriptions_path.exists():
        print(f"错误: 文件不存在 {descriptions_path}")
        return

    print("正在加载画面描述索引...")
    desc_index = load_descriptions_index(descriptions_path)
    print(f"已加载 {len(desc_index)} 条画面描述")

    total = 0
    written = 0
    skipped_no_desc = 0
    skipped_no_suggestion = 0
    skipped_empty_gpt = 0

    with open(pairs_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            previous_clip = pair.get('previous_clip') or {}
            current_clip = pair.get('current_clip')
            if not current_clip:
                continue

            total += 1
            video_id = current_clip.get('video_id')
            clip_number = current_clip.get('clip_number')
            game_suggestion = (current_clip.get('description') or '').strip()

            scene_description = desc_index.get((video_id, clip_number), '').strip()
            if not scene_description and args.skip_no_description:
                skipped_no_desc += 1
                continue
            if not scene_description:
                scene_description = ''

            gpt_value = build_gpt_value(
                scene_description,
                game_suggestion,
                args.desc_label,
                args.suggestion_label,
            )
            if not gpt_value:
                skipped_empty_gpt += 1
                continue
            if not game_suggestion:
                skipped_no_suggestion += 1
                # 仍然输出只有画面描述的情况，除非用户要求只保留两者都有的
                # 这里选择仍然写入

            previous_desc = format_previous_description(
                previous_clip.get('description'),
                args.max_previous_length,
            )
            video_path = current_clip.get('clip_path', '')
            if args.video_base_path and not video_path.startswith(args.video_base_path):
                video_path = f"{args.video_base_path}/{Path(video_path).name}"
            title = f"{video_id}_" if video_id else ""
            instruction = "详细描述视频内容，并给出专业的游戏解说和分析，包括战术策略、操作技巧和关键决策。"
            human_value = (
                f"<video>\n这是cs2的游戏画面，你现在是我的游戏AI助手，请给我一些指导建议"
                + (f"\n上一个连续视频片段的建议内容: {previous_desc}" if previous_desc and previous_desc != "..." else "")
                + f"\n请观看这个游戏视频片段，{instruction}"
            )

            start_time = current_clip.get('start_time', 0.0)
            end_time = current_clip.get('end_time', 0.0)
            duration = current_clip.get('duration', end_time - start_time)

            record = {
                "video": video_path,
                "data_path": args.data_path,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value},
                ],
                "metadata": {
                    "title": title,
                    "category": args.category,
                    "video_start": start_time,
                    "video_end": end_time,
                    "duration": duration,
                    "source_file": f"{video_id}.json" if video_id else "unknown.json",
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1

            if written % 5000 == 0 and written > 0:
                print(f"已写入 {written} 条...")

    print("=" * 60)
    print("处理完成")
    print(f"读取数据对: {total}")
    print(f"写入条数: {written}")
    print(f"因无画面描述跳过: {skipped_no_desc}")
    print(f"因无游戏建议: {skipped_no_suggestion}")
    print(f"因 GPT 内容为空跳过: {skipped_empty_gpt}")
    print(f"输出文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
