"""
将 gameskill_merged_train.jsonl 与 gameskill_qa_train.jsonl 合并为一个完整训练文件。

两个输入文件格式一致（均为 Qwen3-VL 训练用 JSONL），逐行合并到同一输出文件。
支持指定先后顺序、可选打乱顺序（便于训练时混合两种数据）。
"""

import argparse
import json
import random
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="合并 gameskill_merged_train.jsonl 与 gameskill_qa_train.jsonl"
    )
    parser.add_argument(
        "--merged_file",
        type=str,
        default="gameskill_merged_train.jsonl",
        help="描述+建议 训练数据（gameskill_merged_train.jsonl）",
    )
    parser.add_argument(
        "--qa_file",
        type=str,
        default="gameskill_qa_train.jsonl",
        help="问答对训练数据（gameskill_qa_train.jsonl）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gameskill_full_train.jsonl",
        help="合并后的完整训练文件",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["merged_first", "qa_first"],
        default="merged_first",
        help="合并顺序：merged_first=先写 merged 再写 qa；qa_first=先写 qa 再写 merged",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="合并后打乱所有行（建议训练时使用）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="打乱时使用的随机种子（仅 --shuffle 时有效）",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list:
    """读取 JSONL，每行一个 JSON 对象，返回对象列表。"""
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return lines


def main():
    args = get_args()

    merged_path = Path(args.merged_file)
    qa_path = Path(args.qa_file)
    output_path = Path(args.output_file)

    if not merged_path.exists():
        print(f"错误：文件不存在 {merged_path}")
        return
    if not qa_path.exists():
        print(f"错误：文件不存在 {qa_path}")
        return

    print("正在读取...")
    merged = read_jsonl(merged_path)
    qa = read_jsonl(qa_path)
    print(f"  {merged_path.name}: {len(merged)} 条")
    print(f"  {qa_path.name}: {len(qa)} 条")

    if args.order == "merged_first":
        combined = merged + qa
    else:
        combined = qa + merged

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(combined)
        print(f"已打乱（seed={args.seed}）")

    total = len(combined)
    with output_path.open("w", encoding="utf-8") as f:
        for obj in combined:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("合并完成")
    print(f"总条数: {total}")
    print(f"输出: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
