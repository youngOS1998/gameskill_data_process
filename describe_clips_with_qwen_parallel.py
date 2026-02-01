"""
使用Qwen3-VL-8B模型对已分割的视频片段进行描述（并行版本）
支持多进程并行提取帧，批处理模型推理，提高处理速度
"""

import json
import os
import argparse
# 设置环境变量优化显存分配（在导入torch之前）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from transformers import AutoProcessor
import torch
from PIL import Image
import cv2
import numpy as np
from qwen_vl_utils import process_vision_info
from multiprocessing import Pool, cpu_count, set_start_method, Process, Queue
import re


def get_args():
    parser = argparse.ArgumentParser(description='使用Qwen3-VL-8B对已分割的视频片段进行描述（并行版本）')
    parser.add_argument('--clips_dir', type=str, default='video_clips_6_15s',
                       help='视频片段目录')
    parser.add_argument('--output_file', type=str, default='clip_descriptions.jsonl',
                       help='输出描述文件路径')
    parser.add_argument('--model_name', type=str, 
                       default='/root/autodl-tmp/models/models--Qwen--Qwen3-VL-32B-Instruct',
                       # default='/root/autodl-tmp/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                       help='模型名称或路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备')
    parser.add_argument('--max_clips', type=int, default=None,
                       help='最多处理多少个片段（用于测试，None表示处理所有）')
    parser.add_argument('--fps', type=int, default=1,
                       help='从视频中提取帧的帧率（每秒多少帧）')
    parser.add_argument('--max_frames', type=int, default=8,
                       help='每个片段最多使用多少帧')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行提取帧的进程数（CPU密集型任务）')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批处理大小（同时处理多少个片段，根据GPU显存调整）')
    parser.add_argument('--extract_batch_size', type=int, default=20,
                       help='每批提取帧的视频数量（分批处理，减少内存占用，默认20）')
    parser.add_argument('--inference_workers', type=int, default=3,
                       help='模型推理的并行进程数（每个进程使用一个GPU，默认3个GPU）')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                       help='使用的GPU ID列表，用逗号分隔，如 "0,1,2"（默认使用前3个GPU）')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='生成描述的最大token数（减少可加快速度，默认256）')
    return parser.parse_args()


def load_model(model_name: str, device: str, verbose: bool = True, use_device_map: bool = True):
    """加载Qwen3-VL/Qwen2-VL模型"""
    if verbose:
        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {device}")
    
    try:
        # 如果指定了具体的GPU，设置当前进程使用的GPU
        if ':' in device:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
        
        # 尝试加载处理器
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # 确定数据类型
        if 'cuda' in device and torch.cuda.is_available():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # 对于多进程，不使用device_map，直接加载到CPU然后移动到指定GPU
        # 这样可以避免device_map自动分配导致的显存问题
        if ':' in device:
            # 多进程模式：先加载到CPU，再移动到指定GPU（更可控）
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,  # 不使用device_map
                trust_remote_code=True,
                torch_dtype=dtype
            )
            # 明确移动到指定GPU
            model = model.to(device)
        elif use_device_map and device == 'cuda':
            # 单GPU模式，使用auto
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype
            )
        else:
            # 不使用device_map，手动移动到设备
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            model = model.to(device)
        
        model.eval()
        
        # 清理未使用的缓存
        if 'cuda' in device:
            torch.cuda.empty_cache()
        
        if verbose:
            print("模型加载完成")
        return model, processor
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("提示:")
        print("1. 确保安装了transformers>=4.37.0: pip install transformers>=4.37.0")
        print("2. 如果使用Qwen3-VL，请检查模型名称是否正确")
        print("3. 如果显存不足，可以减少 --num_workers 参数")
        raise


def extract_frames_from_video(video_path: str, fps: int = 1, max_frames: int = 8) -> List[Image.Image]:
    """
    从视频片段中提取帧
    
    Args:
        video_path: 视频文件路径
        fps: 提取帧率（每秒多少帧）
        max_frames: 最多提取多少帧
    
    Returns:
        提取的帧列表（PIL Image）
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        
        # 获取视频的原始帧率和总帧数
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            video_fps = 30  # 默认30fps
        
        if total_frames <= 0:
            cap.release()
            return frames
        
        # 计算视频时长
        duration = total_frames / video_fps
        
        # 计算需要提取的帧数
        total_frames_needed = min(int(duration * fps), max_frames)
        
        if total_frames_needed <= 0:
            cap.release()
            return frames
        
        # 计算需要提取的帧的时间点
        time_points = []
        if total_frames_needed == 1:
            time_points = [duration / 2]
        else:
            for i in range(total_frames_needed):
                t = (i / (total_frames_needed - 1)) * duration
                time_points.append(t)
        
        for t in time_points:
            # 计算帧号
            frame_number = int(t * video_fps)
            frame_number = min(frame_number, total_frames - 1)  # 确保不超出范围
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为PIL Image
                img = Image.fromarray(frame_rgb)
                frames.append(img)
        
        cap.release()
    except Exception as e:
        print(f"提取帧时出错 {video_path}: {e}")
    
    return frames


def describe_video_segment(model, processor, frames: List[Image.Image], 
                          prompt: str = None, max_new_tokens: int = 256) -> str:
    """
    使用模型描述视频片段
    
    Args:
        model: Qwen3-VL/Qwen2-VL模型
        processor: 处理器
        frames: 视频帧列表
        prompt: 提示词
    
    Returns:
        描述文本
    """
    if not frames:
        return ""
    
    if prompt is None:
        prompt = """请仔细观察这个游戏视频片段，详细描述玩家的操作和战术行为。包括：
1. 玩家的移动和位置选择
2. 武器使用和瞄准技巧
3. 战术决策和团队配合
4. 关键时刻的操作细节
5. 可以学习的技巧和策略
6. 敌方的行动和反应

请用专业但易懂的语言描述，适合作为游戏AI助手的训练数据。字数要在256个字以内，语言要精炼。而且你要用一段话把这几点总结一下给我，不要分点"""
    
    try:
        # 准备输入 - 支持Qwen2-VL和Qwen3-VL的格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": frame
                    } for frame in frames
                ] + [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 对于 Qwen3-VL，使用独立的 process_vision_info 函数
        try:
            # 尝试使用 processor 的方法（Qwen2-VL）
            image_inputs, video_inputs = processor.process_vision_info(messages)
            video_kwargs = None
            use_video_kwargs = False
        except AttributeError:
            # 如果是 Qwen3-VL，使用独立的函数
            # 尝试不同的参数组合
            try:
                # 尝试只使用 return_video_kwargs
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True
                )
                use_video_kwargs = True
            except TypeError:
                # 如果还是不行，尝试不带任何额外参数
                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages)
                    use_video_kwargs = True
                except ValueError:
                    # 如果返回值的数量不对，尝试只获取前两个
                    result = process_vision_info(messages)
                    if len(result) == 2:
                        image_inputs, video_inputs = result
                        video_kwargs = None
                        use_video_kwargs = False
                    elif len(result) == 3:
                        image_inputs, video_inputs, video_kwargs = result
                        use_video_kwargs = True
                    else:
                        raise ValueError(f"process_vision_info 返回了 {len(result)} 个值，期望 2 或 3 个")
        
        # 准备输入
        if use_video_kwargs and video_kwargs is not None:
            # Qwen3-VL 需要额外的 video_kwargs
            # 尝试不同的参数组合
            try:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_kwargs.get('video_metadata', None),
                    padding=True,
                    return_tensors="pt"
                )
            except (TypeError, KeyError):
                # 如果上面的方式不行，尝试直接传递 video_kwargs
                try:
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        **video_kwargs,
                        padding=True,
                        return_tensors="pt"
                    )
                except TypeError:
                    # 如果还是不行，尝试不使用 video_kwargs
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
        else:
            # Qwen2-VL 不需要 video_kwargs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        inputs = inputs.to(model.device)
        
        # 生成描述
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        
        return output_text[0] if output_text else ""
    
    except Exception as e:
        print(f"生成描述时出错: {e}")
        import traceback
        traceback.print_exc()
        return ""


def parse_clip_filename(filename: str) -> Dict[str, Any]:
    """
    解析视频片段文件名，提取信息
    格式: {video_id}__clip{number}_{start}-{end}.mp4
    
    Args:
        filename: 文件名
    
    Returns:
        包含 video_id, clip_number, start_time, end_time 的字典
    """
    # 移除扩展名
    name_without_ext = Path(filename).stem
    
    # 匹配格式: {video_id}__clip{number}_{start}-{end}
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
        # 如果格式不匹配，尝试其他格式或返回默认值
        return {
            'video_id': name_without_ext,
            'clip_number': 0,
            'start_time': 0.0,
            'end_time': 0.0,
            'duration': 0.0
        }


def extract_frames_worker(args_tuple):
    """
    多进程工作函数：提取帧（CPU密集型任务）
    
    Args:
        args_tuple: (clip_path, fps, max_frames)
    
    Returns:
        (clip_path, frames) 或 (clip_path, None, error)
    """
    clip_path, fps, max_frames = args_tuple
    try:
        frames = extract_frames_from_video(clip_path, fps, max_frames)
        return (clip_path, frames, None)
    except Exception as e:
        return (clip_path, None, str(e))


# 进程级模型缓存（每个进程独立，在spawn模式下每个进程有自己的副本）
# 每个进程只绑定一个GPU，所以只需要存储一个模型实例
_worker_model = None
_worker_processor = None
_worker_gpu_id = None
_worker_model_name = None


def inference_worker_queue(task_queue, result_queue, gpu_id, model_name, max_new_tokens, prompt):
    """
    从队列获取任务并处理的worker函数（每个进程固定绑定一个GPU）
    
    Args:
        task_queue: 任务队列，每个任务为 (clip_path, frames)
        result_queue: 结果队列，每个结果为 (clip_path, description, error)
        gpu_id: 该进程绑定的GPU ID
        model_name: 模型名称或路径
        max_new_tokens: 最大生成token数
        prompt: 提示词
    """
    global _worker_model, _worker_processor, _worker_gpu_id
    
    try:
        # 初始化：绑定GPU并加载模型
        _worker_gpu_id = gpu_id
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        
        print(f"[Worker进程] 正在GPU {gpu_id}上加载模型...")
        _worker_model, _worker_processor = load_model(
            model_name, device, verbose=True, use_device_map=False
        )
        print(f"[Worker进程] GPU {gpu_id}上的模型加载完成")
        
        # 从队列获取任务并处理
        while True:
            task = task_queue.get()
            if task is None:  # 结束信号
                break
            
            clip_path, frames = task
            
            try:
                # 确保使用正确的GPU
                torch.cuda.set_device(gpu_id)
                
                # 生成描述
                description = describe_video_segment(
                    _worker_model, _worker_processor, frames, prompt, max_new_tokens
                )
                
                result_queue.put((clip_path, description, None))
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                result_queue.put((clip_path, "", error_msg))
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
        
    except Exception as e:
        import traceback
        error_msg = f"Worker进程初始化失败: {str(e)}\n{traceback.format_exc()}"
        print(f"[Worker进程] GPU {gpu_id}: {error_msg}")
        # 发送错误信号
        result_queue.put(("ERROR", "", error_msg))


def describe_batch_segments(model, processor, batch_data: List[Tuple[str, List[Image.Image]]], 
                            prompt: str = None, max_new_tokens: int = 256) -> List[str]:
    """
    批量描述多个视频片段（目前逐个处理，因为模型可能不支持真正的批处理）
    
    Args:
        model: Qwen3-VL/Qwen2-VL模型
        processor: 处理器
        batch_data: [(clip_path, frames), ...] 列表
        prompt: 提示词
        max_new_tokens: 最大生成token数
    
    Returns:
        描述文本列表
    """
    descriptions = []
    for clip_path, frames in batch_data:
        if frames:
            description = describe_video_segment(model, processor, frames, prompt, max_new_tokens)
            descriptions.append(description)
        else:
            descriptions.append("")
    return descriptions


def main():
    args = get_args()
    
    # 设置multiprocessing的start method为'spawn'（CUDA要求）
    # 注意：这必须在导入torch之后，创建Pool之前设置
    # 使用spawn方法可以避免CUDA在fork子进程中的问题
    if args.inference_workers > 1 and torch.cuda.is_available():
        try:
            set_start_method('spawn', force=True)
            print("已设置multiprocessing为'spawn'模式（CUDA多进程要求）")
        except RuntimeError:
            # 如果已经设置过，忽略错误
            pass
    
    print("=" * 60)
    print("使用Qwen3-VL-8B对已分割的视频片段进行描述（并行版本）")
    print("=" * 60)
    print(f"视频片段目录: {args.clips_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print(f"并行进程数: {args.num_workers}")
    print(f"提取批次大小: {args.extract_batch_size} 个视频/批")
    print(f"描述批处理大小: {args.batch_size} 个片段/批")
    print(f"推理并行进程数: {args.inference_workers}")
    print(f"使用的GPU: {args.gpu_ids}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"提取帧率: {args.fps} fps")
    print(f"最大帧数: {args.max_frames}")
    if args.max_clips:
        print(f"最多处理片段数: {args.max_clips}")
    print("=" * 60)
    
    # 查找所有视频片段
    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"错误: 目录不存在: {args.clips_dir}")
        return
    
    video_files = list(clips_dir.glob("*.mp4"))
    # 按文件名排序，确保顺序稳定
    video_files.sort(key=lambda x: x.name)

    # 检查输出文件，读取已处理的文件列表
    processed_clips = set()
    if Path(args.output_file).exists():
        print(f"\n检测到已存在的输出文件: {args.output_file}")
        print("正在读取已处理的文件列表...")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 根据输出格式，使用 clip_path 或 clip_filename
                    clip_path = data.get('clip_path') or data.get('clip_filename')
                    if clip_path:
                        # 转换为 Path 对象以便比较
                        processed_clips.add(Path(clip_path))
                except json.JSONDecodeError:
                    continue
        
        print(f"已处理 {len(processed_clips)} 个文件")
        
        # 过滤掉已处理的文件
        original_count = len(video_files)
        video_files = [f for f in video_files if f not in processed_clips]
        print(f"过滤后剩余 {len(video_files)} 个文件待处理（跳过了 {original_count - len(video_files)} 个已处理文件）")

    if not video_files:
        print("所有文件都已处理完成！")
        return
    
    if args.max_clips:
        video_files = video_files[:args.max_clips]
    
    print(f"\n找到 {len(video_files)} 个视频片段")
    
    # 准备多进程参数（用于提取帧）
    if args.num_workers <= 0:
        args.num_workers = max(1, cpu_count() - 1)
    
    # 解析GPU ID列表
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    if args.inference_workers > 1:
        # 检查GPU数量是否足够
        if len(gpu_ids) < args.inference_workers:
            print(f"警告: 指定的GPU数量 ({len(gpu_ids)}) 少于推理进程数 ({args.inference_workers})")
            print(f"将使用 {len(gpu_ids)} 个GPU，推理进程数调整为 {len(gpu_ids)}")
            args.inference_workers = len(gpu_ids)
        elif len(gpu_ids) > args.inference_workers:
            print(f"警告: 指定的GPU数量 ({len(gpu_ids)}) 多于推理进程数 ({args.inference_workers})")
            print(f"将只使用前 {args.inference_workers} 个GPU: {gpu_ids[:args.inference_workers]}")
            gpu_ids = gpu_ids[:args.inference_workers]
        
        print(f"使用 {args.inference_workers} 个进程并行推理（GPU: {gpu_ids}）")
        print(f"最大生成token数: {args.max_new_tokens}")
        # 多进程推理模式：不需要在主进程加载模型
        model = None
        processor = None
    else:
        print(f"单进程推理模式")
        # 单进程推理模式：在主进程加载模型
        print("\n正在加载模型...")
        model, processor = load_model(args.model_name, args.device)
    
    # 打开输出文件（追加模式，支持分批保存）
    output_file_handle = open(args.output_file, 'a', encoding='utf-8')
    
    all_results_count = 0
    failed_count = 0
    total_processed = 0
    
    # 分批处理视频文件
    print(f"\n开始分批处理，共 {len(video_files)} 个视频片段")
    
    for batch_idx in range(0, len(video_files), args.extract_batch_size):
        batch_video_files = video_files[batch_idx:batch_idx + args.extract_batch_size]
        batch_num = (batch_idx // args.extract_batch_size) + 1
        total_batches = (len(video_files) + args.extract_batch_size - 1) // args.extract_batch_size
        
        print(f"\n{'='*60}")
        print(f"处理批次 {batch_num}/{total_batches} ({len(batch_video_files)} 个视频片段)")
        print(f"{'='*60}")
        
        # 第一步：并行提取当前批次的帧
        print(f"\n步骤1: 提取帧...")
        extract_tasks = [
            (str(clip_path), args.fps, args.max_frames)
            for clip_path in batch_video_files
        ]
        
        clip_frames_map = {}  # {clip_path: frames}
        failed_extractions = []
        
        with Pool(processes=args.num_workers) as pool:
            with tqdm.tqdm(total=len(extract_tasks), desc=f"  提取帧 (批次 {batch_num})", leave=False) as pbar:
                for clip_path, frames, error in pool.imap(extract_frames_worker, extract_tasks):
                    if error is None and frames:
                        clip_frames_map[clip_path] = frames
                    else:
                        failed_extractions.append((clip_path, error or "无法提取帧"))
                    pbar.update(1)
        
        print(f"  成功提取 {len(clip_frames_map)} 个片段的帧，失败 {len(failed_extractions)} 个")
        
        # 第二步：批处理生成描述
        if clip_frames_map:
            print(f"\n步骤2: 生成描述...")
            clip_paths = list(clip_frames_map.keys())
            
            if args.inference_workers > 1:
                # 多进程推理模式
                # 准备提示词
                prompt = """请仔细观察这个游戏视频片段，详细描述玩家的操作和战术行为。包括：
1. 玩家的移动和位置选择
2. 武器使用和瞄准技巧
3. 战术决策和团队配合
4. 关键时刻的操作细节
5. 可以学习的技巧和策略
6. 敌方的行动和反应

请用专业但易懂的语言描述，适合作为游戏AI助手的训练数据。并且你要把这几点总结成一段话给我，不要分点"""
                
                # 准备所有任务
                all_tasks = [
                    (clip_path, clip_frames_map[clip_path])
                    for clip_path in clip_paths
                ]
                
                # 创建任务队列和结果队列
                task_queue = Queue()
                result_queue = Queue()
                
                # 将任务放入队列
                for task in all_tasks:
                    task_queue.put(task)
                
                # 为每个worker进程添加结束信号
                for _ in range(args.inference_workers):
                    task_queue.put(None)
                
                # 创建worker进程（每个进程绑定一个GPU）
                processes = []
                for worker_idx, gpu_id in enumerate(gpu_ids):
                    if worker_idx >= args.inference_workers:
                        break
                    p = Process(
                        target=inference_worker_queue,
                        args=(task_queue, result_queue, gpu_id, args.model_name, 
                              args.max_new_tokens, prompt)
                    )
                    p.start()
                    processes.append(p)
                
                # 收集结果
                completed = 0
                with tqdm.tqdm(total=len(all_tasks), desc=f"  生成描述 (批次 {batch_num})", leave=False) as pbar:
                    while completed < len(all_tasks):
                        clip_path, description, error = result_queue.get()
                        
                        if clip_path == "ERROR":
                            # 进程初始化错误
                            failed_count += 1
                            print(f"\n  进程初始化失败: {error[:200]}")
                            completed += 1
                            pbar.update(1)
                            continue
                        
                        clip_filename = Path(clip_path).name
                        clip_info = parse_clip_filename(clip_filename)
                        
                        if error is None and description:
                            output_result = {
                                'clip_path': clip_path,
                                'clip_filename': clip_filename,
                                'video_id': clip_info['video_id'],
                                'clip_number': clip_info['clip_number'],
                                'start_time': clip_info['start_time'],
                                'end_time': clip_info['end_time'],
                                'duration': clip_info['duration'],
                                'description': description
                            }
                            output_file_handle.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                            output_file_handle.flush()  # 立即刷新到磁盘
                            all_results_count += 1
                        else:
                            failed_count += 1
                            if error:
                                print(f"\n  生成描述失败: {clip_filename} - {error[:100]}")
                            else:
                                print(f"\n  生成描述失败: {clip_filename}")
                        
                        completed += 1
                        pbar.update(1)
                
                # 等待所有进程完成
                for p in processes:
                    p.join()
            else:
                # 单进程推理模式（原有逻辑）
                for i in tqdm.tqdm(range(0, len(clip_paths), args.batch_size), 
                                  desc=f"  生成描述 (批次 {batch_num})", leave=False):
                    batch_paths = clip_paths[i:i + args.batch_size]
                    batch_data = [(path, clip_frames_map[path]) for path in batch_paths]
                    
                    # 生成描述
                    descriptions = describe_batch_segments(model, processor, batch_data, None, args.max_new_tokens)
                    
                    # 立即保存结果到文件
                    for clip_path, description in zip(batch_paths, descriptions):
                        clip_filename = Path(clip_path).name
                        clip_info = parse_clip_filename(clip_filename)
                        
                        if description:
                            output_result = {
                                'clip_path': clip_path,
                                'clip_filename': clip_filename,
                                'video_id': clip_info['video_id'],
                                'clip_number': clip_info['clip_number'],
                                'start_time': clip_info['start_time'],
                                'end_time': clip_info['end_time'],
                                'duration': clip_info['duration'],
                                'description': description
                            }
                            output_file_handle.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                            output_file_handle.flush()  # 立即刷新到磁盘
                            all_results_count += 1
                        else:
                            failed_count += 1
                            print(f"\n  生成描述失败: {clip_filename}")
        
        # 记录提取帧失败的片段
        for clip_path, error in failed_extractions:
            failed_count += 1
            clip_filename = Path(clip_path).name
            print(f"\n  提取帧失败: {clip_filename} - {error}")
        
        total_processed += len(batch_video_files)
        
        # 清理当前批次的帧数据，释放内存
        del clip_frames_map
        if args.inference_workers > 1:
            # 多进程模式下，每个进程会自己清理GPU缓存
            pass
        elif args.device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n批次 {batch_num} 完成: 成功 {all_results_count - (failed_count - len(failed_extractions))} 个，失败 {len(failed_extractions)} 个")
    
    # 关闭输出文件
    output_file_handle.close()
    
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"共处理 {len(video_files)} 个视频片段")
    print(f"成功: {all_results_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"输出文件: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

