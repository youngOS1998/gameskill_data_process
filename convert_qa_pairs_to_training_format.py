"""
将 clip_qa_pairs.jsonl 转换为 Qwen3-VL 微调所需的对话格式（与 gameskill_merged_train.jsonl 结构一致）。

输入格式（clip_qa_pairs.jsonl）示例：
{
  "question": "在该手枪局片段中，G2进攻方采用的双人协同推进战术有何战术优势？",
  "answer": "双人协同推进通过一人掩护、一人探头的方式，有效降低暴露风险...",
  "source_clip_path": "video_clips_6_15s/av1005709654__clip0001_0.00-12.57.mp4",
  "video_id": "av1005709654",
  "clip_number": 1,
  "start_time": 0.0,
  "end_time": 12.57
}

输出格式（与 gameskill_merged_train.jsonl 一致）示例：
{
  "video": "video_clips_6_15s/av1005709654__clip0001_0.00-12.57.mp4",
  "data_path": "/root/autodl-tmp/Projects/qwen_gameskill/qwen-vl-finetune/dataset",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\\n这是cs2的游戏画面，你现在是我的游戏AI助手。\\n请根据这个游戏视频片段回答下面的问题：在该手枪局片段中，G2进攻方采用的双人协同推进战术有何战术优势？"
    },
    {
      "from": "gpt",
      "value": "双人协同推进通过一人掩护、一人探头的方式，有效降低暴露风险..."
    }
  ],
  "metadata": {
    "title": "av1005709654_",
    "category": "cs2_web_data",
    "video_start": 0.0,
    "video_end": 12.57,
    "duration": 12.57,
    "source_file": "av1005709654.json"
  }
}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_args():
    parser = argparse.ArgumentParser(
        description="将 clip_qa_pairs.jsonl 转成 Qwen3-VL 训练用对话格式"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="clip_qa_pairs.jsonl",
        help="输入的 QA 文件（JSONL 格式）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gameskill_qa_train.jsonl",
        help="输出训练数据文件（JSONL 格式）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/autodl-tmp/Projects/qwen_gameskill/qwen-vl-finetune/dataset",
        help="数据根路径（与现有 gameskill_merged_train.jsonl 一致）",
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="video_clips_6_15s",
        help="视频相对路径前缀（如果 source_clip_path 不包含则自动补上）",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="cs2_web_data",
        help="metadata.category 字段，保持与现有数据一致",
    )
    return parser.parse_args()


def build_video_path(source_clip_path: str, video_base_path: str) -> str:
    """
    处理 source_clip_path，输出最终写入训练数据的 video 字段。
    """
    if not source_clip_path:
        return ""

    # 已经是以 video_base_path 开头，直接返回
    if video_base_path and source_clip_path.startswith(video_base_path):
        return source_clip_path

    # 否则只取文件名，前面补上 video_base_path
    name = Path(source_clip_path).name
    return f"{video_base_path}/{name}" if video_base_path else source_clip_path


def convert_qa_record(
    qa: Dict[str, Any],
    data_path: str,
    video_base_path: str,
    category: str,
) -> Optional[Dict[str, Any]]:
    """
    将一条 QA 记录转换为训练格式。
    """
    question = (qa.get("question") or "").strip()
    answer = (qa.get("answer") or "").strip()
    source_clip_path = qa.get("source_clip_path") or qa.get("clip_path") or ""
    video_id = (qa.get("video_id") or "").strip()
    start_time = qa.get("start_time", 0.0)
    end_time = qa.get("end_time", 0.0)

    if not question or not answer or not source_clip_path:
        # 关键字段缺失则跳过
        return None

    video_path = build_video_path(source_clip_path, video_base_path)

    # title 和 source_file 按现有训练数据风格构造
    title = f"{video_id}_" if video_id else ""
    duration = end_time - start_time if (end_time and start_time is not None) else 0.0

    # human 提示：与已有数据保持风格，只是把“描述视频并给出分析”改成“回答问题”
    human_value = (
        "<video>\n"
        "这是cs2的游戏画面，你现在是我的游戏AI助手。\n"
        f"请根据这个游戏视频片段回答下面的问题：{question}"
    )

    conversations = [
        {
            "from": "human",
            "value": human_value,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

    metadata = {
        "title": title,
        "category": category,
        "video_start": start_time,
        "video_end": end_time,
        "duration": duration,
        "source_file": f"{video_id}.json" if video_id else "unknown.json",
    }

    training_obj = {
        "video": video_path,
        "data_path": data_path,
        "conversations": conversations,
        "metadata": metadata,
    }

    return training_obj


def main():
    args = get_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"错误：输入文件不存在：{input_path}")
        return

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"data_path: {args.data_path}")
    print(f"video_base_path: {args.video_base_path}")
    print(f"category: {args.category}")
    print("=" * 60)

    total = 0
    converted = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_no} 行 JSON 解析失败，已跳过：{e}")
                skipped += 1
                continue

            total += 1
            record = convert_qa_record(
                obj,
                data_path=args.data_path,
                video_base_path=args.video_base_path,
                category=args.category,
            )
            if record is None:
                skipped += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

            if converted % 5000 == 0:
                print(f"已转换 {converted} 条（共读取 {total} 条，跳过 {skipped} 条）")

    print("\n" + "=" * 60)
    print("转换完成")
    print(f"总读取条数: {total}")
    print(f"成功转换: {converted}")
    print(f"跳过条数: {skipped}")
    print(f"输出文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

