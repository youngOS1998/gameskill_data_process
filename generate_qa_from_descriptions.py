"""
基于视频描述数据，使用 Qwen3-VL-32B 生成游戏理解相关的问答对，用于训练 Qwen3-VL-4B 的游戏理解能力。
输入：clip_descriptions_copy.jsonl（每条包含 description 等字段）
输出：问答对 jsonl，每条格式 {"question": "...", "answer": "...", "source_clip_path": "...", ...}
仿照 describe_clips_with_qwen_parallel.py 的并行与模型调用方式。
"""

import json
import os
import argparse
import re
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from multiprocessing import Process, Queue, set_start_method


def get_args():
    parser = argparse.ArgumentParser(description='根据视频描述用 Qwen3-VL-32B 生成游戏问答对')
    parser.add_argument('--input_file', type=str, default='clip_descriptions_copy.jsonl',
                        help='描述数据 jsonl 路径')
    parser.add_argument('--output_file', type=str, default='clip_qa_pairs.jsonl',
                        help='输出问答对 jsonl 路径')
    parser.add_argument('--model_name', type=str,
                        default='/root/autodl-tmp/models/models--Qwen--Qwen3-VL-32B-Instruct',
                        help='Qwen3-VL 模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_descriptions', type=int, default=None,
                        help='最多处理多少条描述（测试用，None 表示全部）')
    parser.add_argument('--questions_per_description', type=int, default=4,
                        help='每条描述生成的问题数量（建议 3-5）')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='单次生成的最大 token 数')
    parser.add_argument('--inference_workers', type=int, default=4,
                        help='推理并行进程数（多 GPU 时可增大）')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                        help='GPU ID 列表，逗号分隔，如 "0,1,2"')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每批处理的描述数（当前为逐条生成，保留参数兼容）')
    return parser.parse_args()


def load_model(model_name: str, device: str, verbose: bool = True, use_device_map: bool = True):
    """加载 Qwen3-VL 模型（用于纯文本生成）"""
    if verbose:
        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {device}")
    try:
        if ':' in device:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if ('cuda' in device and torch.cuda.is_available()) else torch.float32
        if ':' in device:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            model = model.to(device)
        elif use_device_map and device == 'cuda':
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            model = model.to(device)
        model.eval()
        if 'cuda' in device:
            torch.cuda.empty_cache()
        if verbose:
            print("模型加载完成")
        return model, processor
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


# def build_qa_prompt(description: str, num_questions: int = 4) -> str:
#     """构建用于生成问答对的用户提示（纯文本）。"""
#     return f"""你是一位精通 FPS 与战术竞技游戏的专家。下面是一段游戏视频片段的文字描述，请基于该描述生成 {num_questions} 个与游戏理解相关的问题，并为每个问题提供简洁、专业的解答。要求：

# 1. 问题类型需覆盖：游戏关键信息捕捉、战术分析、操作技巧、局势判断、地图/点位理解、团队配合、经济与装备选择、可学习的技巧等。
# 2. 问题与答案必须严格基于给定描述中的内容，不要编造描述中未出现的信息。
# 3. 答案用 1–3 句话说明即可，语言专业但易懂，适合作为游戏 AI 助手的训练数据。

# 请严格按以下格式输出（不要省略编号，不要使用 Markdown 标题）：
# 问题1：<这里写第一个问题>
# 答案1：<这里写第一个答案>

# 问题2：<这里写第二个问题>
# 答案2：<这里写第二个答案>

# 问题3：<这里写第三个问题>
# 答案3：<这里写第三个答案>

# 问题4：<这里写第四个问题>
# 答案4：<这里写第四个答案>

# 【视频描述】
# {description}
# """

def build_qa_prompt(description: str, num_questions: int = 4) -> str:
    """构建用于生成问答对的用户提示（纯文本）。"""
    return f"""你是一位精通 FPS 与战术竞技游戏的专家。下面是一段游戏视频片段的文字描述，请基于该描述生成 {num_questions} 个与游戏理解相关的问题，并为每个问题提供简洁、专业的解答。要求：

1. 问题类型需覆盖：游戏关键信息捕捉，比如问题可以是：“这个视频片段中，玩家使用了什么武器？”、“玩家在这个视频片段中，做了什么战术动作？”、“玩家在这个视频片段中，遇到了什么情况？”等。
2. 问题与答案必须严格基于给定描述中的内容，不要编造描述中未出现的信息。
3. 答案用 1–3 句话说明即可，语言专业但易懂，适合作为游戏 AI 助手的训练数据。

请严格按以下格式输出（不要省略编号，不要使用 Markdown 标题）：
问题1：<这里写第一个问题>
答案1：<这里写第一个答案>

问题2：<这里写第二个问题>
答案2：<这里写第二个答案>

问题3：<这里写第三个问题>
答案3：<这里写第三个答案>

问题4：<这里写第四个问题>
答案4：<这里写第四个答案>

【视频描述】
{description}
"""


def parse_qa_from_response(response: str) -> List[Tuple[str, str]]:
    """从模型输出中解析出 (问题, 答案) 对。"""
    pairs = []
    # 匹配 "问题N：..." 与 "答案N：..." 或 "问题 N：" 等变体
    q_pattern = re.compile(r'问题\s*(\d+)\s*[：:]\s*(.+?)(?=问题\s*\d+\s*[：:]|答案\s*\d+\s*[：:]|$)', re.DOTALL)
    a_pattern = re.compile(r'答案\s*(\d+)\s*[：:]\s*(.+?)(?=问题\s*\d+\s*[：:]|答案\s*\d+\s*[：:]|$)', re.DOTALL)
    q_matches = list(q_pattern.finditer(response))
    a_matches = list(a_pattern.finditer(response))
    q_by_num = {int(m.group(1)): m.group(2).strip() for m in q_matches}
    a_by_num = {int(m.group(1)): m.group(2).strip() for m in a_matches}
    for num in sorted(set(q_by_num) & set(a_by_num)):
        q = q_by_num[num].strip()
        a = a_by_num[num].strip()
        if q and a:
            pairs.append((q, a))
    # 若上面没匹配到，尝试更宽松的 "Q:" "A:" 或 "问：" "答："
    if not pairs:
        loose_q = re.compile(r'(?:问|Q)[题]?\s*[：:]\s*(.+?)(?=(?:答|A)[案]?\s*[：:]|$)', re.DOTALL)
        loose_a = re.compile(r'(?:答|A)[案]?\s*[：:]\s*(.+?)(?=(?:问|Q)[题]?\s*[：:]|$)', re.DOTALL)
        qs = loose_q.findall(response)
        as_ = loose_a.findall(response)
        for q, a in zip(qs, as_):
            q, a = q.strip(), a.strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def generate_qa_text_only(model, processor, description: str, num_questions: int, max_new_tokens: int) -> str:
    """
    仅使用文本输入调用 Qwen3-VL，生成问答内容（无图像）。
    """
    prompt = build_qa_prompt(description, num_questions)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 纯文本：无图像/视频，直接用 tokenizer 编码
    tokenizer = processor.tokenizer
    max_len = getattr(tokenizer, 'model_max_length', None) or 32768
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(max_len, 32768)
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


# 进程内模型缓存
_worker_model = None
_worker_processor = None


def inference_worker_qa(task_queue: Queue, result_queue: Queue, gpu_id: int, model_name: str,
                       num_questions: int, max_new_tokens: int):
    """单进程 worker：从队列取 (clip_path, meta, description)，生成问答并放入 result_queue。"""
    global _worker_model, _worker_processor
    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        print(f"[Worker] 在 GPU {gpu_id} 上加载模型...")
        _worker_model, _worker_processor = load_model(model_name, device, verbose=True, use_device_map=False)
        print(f"[Worker] GPU {gpu_id} 模型加载完成")
        while True:
            task = task_queue.get()
            if task is None:
                break
            clip_path, meta, description = task
            try:
                torch.cuda.set_device(gpu_id)
                raw = generate_qa_text_only(
                    _worker_model, _worker_processor,
                    description, num_questions, max_new_tokens
                )
                pairs = parse_qa_from_response(raw)
                result_queue.put((clip_path, meta, pairs, None))
            except Exception as e:
                import traceback
                err = f"{str(e)}\n{traceback.format_exc()}"
                result_queue.put((clip_path, meta, [], err))
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    except Exception as e:
        import traceback
        result_queue.put(("ERROR", None, [], f"{e}\n{traceback.format_exc()}"))


def load_descriptions(path: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载描述 jsonl，返回 [{clip_path, description, ...}, ...]"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                desc = data.get('description') or data.get('desc') or ''
                if not desc:
                    continue
                rows.append(data)
                if max_items and len(rows) >= max_items:
                    break
            except json.JSONDecodeError:
                continue
    return rows


def main():
    args = get_args()
    if args.inference_workers > 1 and torch.cuda.is_available():
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    print("=" * 60)
    print("根据视频描述生成游戏理解问答对（Qwen3-VL-32B）")
    print("=" * 60)
    print(f"输入: {args.input_file}")
    print(f"输出: {args.output_file}")
    print(f"模型: {args.model_name}")
    print(f"每条描述生成问题数: {args.questions_per_description}")
    print(f"最大生成 token: {args.max_new_tokens}")
    print(f"推理进程数: {args.inference_workers}")
    print(f"GPU: {args.gpu_ids}")
    if args.max_descriptions:
        print(f"最多处理: {args.max_descriptions} 条")
    print("=" * 60)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在 {args.input_file}")
        return

    rows = load_descriptions(args.input_file, args.max_descriptions)
    if not rows:
        print("没有可用的描述数据")
        return

    # 已处理集合（支持断点续跑）
    processed = set()
    output_path = Path(args.output_file)
    if output_path.exists():
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    clip = obj.get('source_clip_path') or obj.get('clip_path')
                    if clip:
                        processed.add(clip)
                except json.JSONDecodeError:
                    pass
        print(f"已处理 {len(processed)} 条（将跳过）")

    todo = [r for r in rows if (r.get('clip_path') or r.get('clip_filename')) not in processed]
    if not todo:
        print("没有待处理的描述")
        return

    print(f"待处理: {len(todo)} 条")

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    if args.inference_workers > 1:
        if len(gpu_ids) < args.inference_workers:
            args.inference_workers = len(gpu_ids)
        gpu_ids = gpu_ids[:args.inference_workers]

    task_queue = Queue()
    result_queue = Queue()
    for r in todo:
        clip_path = r.get('clip_path') or r.get('clip_filename') or ''
        meta = {
            'video_id': r.get('video_id'),
            'clip_number': r.get('clip_number'),
            'start_time': r.get('start_time'),
            'end_time': r.get('end_time'),
        }
        task_queue.put((clip_path, meta, (r.get('description') or r.get('desc') or '').strip()))
    for _ in range(args.inference_workers):
        task_queue.put(None)

    processes = []
    if args.inference_workers > 1:
        for i, gid in enumerate(gpu_ids):
            p = Process(
                target=inference_worker_qa,
                args=(task_queue, result_queue, gid, args.model_name,
                      args.questions_per_description, args.max_new_tokens)
            )
            p.start()
            processes.append(p)
    out = open(args.output_file, 'a', encoding='utf-8')
    total_qa = 0
    done = 0
    failed = 0
    n_tasks = len(todo)

    if args.inference_workers > 1:
        with tqdm.tqdm(total=n_tasks, desc="生成问答") as pbar:
            while done < n_tasks:
                res = result_queue.get()
                if res[0] == "ERROR":
                    failed += 1
                    print(f"\nWorker 错误: {res[3][:200]}")
                    done += 1
                    pbar.update(1)
                    continue
                clip_path, meta, pairs, err = res
                if err:
                    failed += 1
                    if err:
                        print(f"\n生成失败 {clip_path}: {err[:150]}")
                    done += 1
                    pbar.update(1)
                    continue
                for q, a in pairs:
                    record = {
                        "question": q,
                        "answer": a,
                        "source_clip_path": clip_path,
                        "video_id": meta.get("video_id"),
                        "clip_number": meta.get("clip_number"),
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_qa += 1
                out.flush()
                done += 1
                pbar.update(1)
        for p in processes:
            p.join()
    else:
        # 单进程：加载模型后逐条生成、立即写入并更新进度条（用户能看到进度，不会误以为卡死）
        device = f'cuda:{gpu_ids[0]}' if gpu_ids else args.device
        model, processor = load_model(args.model_name, device)
        print("开始逐条生成问答（每条约需 1～3 分钟，请耐心等待）…\n")
        with tqdm.tqdm(total=n_tasks, desc="生成问答") as pbar:
            for r in todo:
                clip_path = r.get('clip_path') or r.get('clip_filename') or ''
                meta = {'video_id': r.get('video_id'), 'clip_number': r.get('clip_number'),
                        'start_time': r.get('start_time'), 'end_time': r.get('end_time')}
                desc = (r.get('description') or r.get('desc') or '').strip()
                try:
                    raw = generate_qa_text_only(
                        model, processor, desc,
                        args.questions_per_description, args.max_new_tokens
                    )
                    pairs = parse_qa_from_response(raw)
                    err = None
                except Exception as e:
                    import traceback
                    err = f"{e}\n{traceback.format_exc()}"
                    pairs = []
                if err:
                    failed += 1
                    done += 1
                    if err:
                        tqdm.tqdm.write(f"生成失败 {clip_path}: {err[:150]}")
                    pbar.update(1)
                    continue
                for q, a in pairs:
                    record = {
                        "question": q,
                        "answer": a,
                        "source_clip_path": clip_path,
                        "video_id": meta.get("video_id"),
                        "clip_number": meta.get("clip_number"),
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_qa += 1
                out.flush()
                done += 1
                pbar.update(1)

    out.close()
    print("\n" + "=" * 60)
    print("处理完成")
    print(f"处理描述条数: {done}，失败: {failed}")
    print(f"生成问答对总数: {total_qa}")
    print(f"输出: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
