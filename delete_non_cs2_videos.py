"""
根据 cs2_filter_results.jsonl 中的判断结果，
删除 reason 中判定为“否”的视频文件及其对应的 .json 标注文件。

使用方法（在 /root/autodl-tmp/gameskill_data 下运行）:

    python delete_non_cs2_videos.py \
        --result_file cs2_filter_results.jsonl \
        --base_dir .

默认会从每行 JSON 的字段:
  - video_path: 视频相对路径，例如 "processed_bilibili_cs2/xxx.mp4"
  - reason: 包含一行 "- 是否是CS2或CSGO：是/否"
中检查是否包含 "是否是CS2或CSGO：否"，如果是则删除对应视频。
"""

import argparse
import json
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="根据 cs2_filter_results.jsonl 删除 reason 判定为“否”的视频文件及对应的 .json 文件"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="cs2_filter_results_2.jsonl",
        help="过滤结果文件（JSONL，每行一个 JSON）",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="视频根目录（通常是项目根目录，如 /root/autodl-tmp/gameskill_data）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将删除的文件，不实际删除",
    )
    return parser.parse_args()


def should_delete_by_reason(reason: str) -> bool:
    """
    根据 reason 文本判断是否要删除:
    只要包含 “是否是CS2或CSGO：否” 就认为需要删除。
    """
    if not isinstance(reason, str):
        return False
    return "是否是CS2或CSGO：否" in reason


def main():
    args = get_args()

    result_path = Path(args.result_file)
    if not result_path.is_file():
        print(f"结果文件不存在: {result_path}")
        return

    base_dir = Path(args.base_dir).resolve()
    print(f"结果文件: {result_path}")
    print(f"基准目录: {base_dir}")
    print(f"dry_run 模式: {args.dry_run}")

    total = 0
    to_delete_video = 0
    to_delete_json = 0
    deleted_video = 0
    deleted_json = 0
    missing_video = 0
    missing_json = 0

    # 为了安全，统计完后再真正删除
    delete_video_paths = []
    delete_json_paths = []

    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # 跳过坏行
                continue

            total += 1
            reason = data.get("reason", "")
            if not should_delete_by_reason(reason):
                continue

            video_path = data.get("video_path") or data.get("video_filename")
            if not video_path:
                continue

            # video_path 在你的结果里通常是 "processed_bilibili_cs2/xxx.mp4"
            abs_video_path = (base_dir / video_path).resolve()
            delete_video_paths.append(abs_video_path)

            # 对应的 json 文件：同目录同名但后缀为 .json
            if abs_video_path.suffix.lower() == ".mp4":
                abs_json_path = abs_video_path.with_suffix(".json")
                delete_json_paths.append(abs_json_path)

    # 去重
    delete_video_paths = sorted(set(delete_video_paths))
    delete_json_paths = sorted(set(delete_json_paths))
    to_delete_video = len(delete_video_paths)
    to_delete_json = len(delete_json_paths)

    print(f"\n在 {total} 条记录中，找到 {to_delete_video} 个 reason 判定为“否”的视频。")
    print(f"对应需要处理的 json 文件约 {to_delete_json} 个（按同名 .json 推算）。")

    # 先处理视频文件
    for path in delete_video_paths:
        if not path.is_file():
            print(f"[视频不存在] {path}")
            missing_video += 1
            continue

        if args.dry_run:
            print(f"[将删除视频] {path}")
        else:
            try:
                os.remove(path)
                print(f"[已删除视频] {path}")
                deleted_video += 1
            except Exception as e:
                print(f"[删除视频失败] {path} - {e}")

    # 再处理对应的 json 文件
    for path in delete_json_paths:
        if not path.is_file():
            print(f"[JSON 不存在] {path}")
            missing_json += 1
            continue

        if args.dry_run:
            print(f"[将删除 JSON] {path}")
        else:
            try:
                os.remove(path)
                print(f"[已删除 JSON] {path}")
                deleted_json += 1
            except Exception as e:
                print(f"[删除 JSON 失败] {path} - {e}")

    if args.dry_run:
        print("\ndry_run 模式下未真正删除任何文件。")

    print("\n统计信息：")
    print(f"  总记录数: {total}")
    print(f"  需删除的视频: {to_delete_video}")
    print(f"  实际删除成功的视频: {deleted_video}")
    print(f"  结果文件中有记录但磁盘上不存在的视频: {missing_video}")
    print(f"  需删除的 JSON: {to_delete_json}")
    print(f"  实际删除成功的 JSON: {deleted_json}")
    print(f"  结果中推算存在但磁盘上找不到的 JSON: {missing_json}")


if __name__ == "__main__":
    main()


