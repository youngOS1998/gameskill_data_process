"""
使用Qwen3-VL模型判断视频片段是否是CS2或CSGO游戏视频（并行版本）
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
from multiprocessing import Pool, cpu_count, set_start_method
import re


def get_args():
    parser = argparse.ArgumentParser(description='使用Qwen3-VL判断视频是否是CS2或CSGO游戏视频（并行版本）')
    parser.add_argument('--video_dir', type=str, default='video_clips_6_15s',
                       help='视频目录')
    parser.add_argument('--output_file', type=str, default='cs2_filter_results_2.jsonl',
                       help='输出判断结果文件路径')
    parser.add_argument('--model_name', type=str, 
                       default='/root/autodl-tmp/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                       help='模型名称或路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='最多处理多少个视频（用于测试，None表示处理所有）')
    parser.add_argument('--fps', type=int, default=1,
                       help='从视频中提取帧的帧率（每秒多少帧）')
    parser.add_argument('--max_frames', type=int, default=2,
                       help='每个视频最多使用多少帧')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行提取帧的进程数（CPU密集型任务）')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批处理大小（同时处理多少个视频，根据GPU显存调整）')
    parser.add_argument('--extract_batch_size', type=int, default=20,
                       help='每批提取帧的视频数量（分批处理，减少内存占用，默认20）')
    parser.add_argument('--inference_workers', type=int, default=3,
                       help='模型推理的并行进程数（每个进程使用一个GPU，默认3个GPU）')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                       help='使用的GPU ID列表，用逗号分隔，如 "0,1,2"（默认使用前3个GPU）')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='生成判断结果的最大token数（减少可加快速度，默认128）')
    parser.add_argument('--video_extensions', type=str, nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv'],
                       help='支持的视频文件扩展名')
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


def judge_cs2_csgo_video(model, processor, frames: List[Image.Image], 
                         max_new_tokens: int = 128) -> Dict[str, Any]:
    """
    使用模型判断视频是否是CS2或CSGO游戏视频
    
    Args:
        model: Qwen3-VL/Qwen2-VL模型
        processor: 处理器
        frames: 视频帧列表
        max_new_tokens: 最大生成token数
    
    Returns:
        包含判断结果的字典：{'is_cs2_csgo': bool, 'reason': str}
    """
    if not frames:
        return {'is_cs2_csgo': False, 'reason': '无法提取视频帧'}
    
    prompt = """请仔细观察这个视频片段，判断这是否是《反恐精英2》(Counter-Strike 2, CS2) 或《反恐精英：全球攻势》(Counter-Strike: Global Offensive, CSGO) 游戏的视频。

请从以下几个方面判断：
1. 游戏界面和UI元素（如HUD、计分板、武器图标等）
2. 游戏地图和场景（如Dust2、Mirage、Inferno等经典地图）
3. 武器和装备（如AK-47、M4A4、AWP等CS系列特有武器）
4. 游戏机制和玩法（如回合制、经济系统、炸弹拆除等）
5. 画面风格和渲染效果（CS2的Source 2引擎特征或CSGO的Source引擎特征）

请用以下格式回答：
- 是否是CS2或CSGO：是/否
- 判断理由：（简要说明判断依据）

如果无法确定，请说明原因。"""
    
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
        
        # 生成判断结果
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
        
        response = output_text[0] if output_text else ""
        
        # 解析判断结果
        result = parse_judgment_response(response)
        return result
    
    except Exception as e:
        print(f"生成判断结果时出错: {e}")
        import traceback
        traceback.print_exc()
        return {'is_cs2_csgo': False, 'reason': f'处理出错: {str(e)}'}


def parse_judgment_response(response: str) -> Dict[str, Any]:
    """
    解析模型的判断响应，提取是否是CS2或CSGO、理由
    
    Args:
        response: 模型的原始响应文本
    
    Returns:
        {'is_cs2_csgo': bool, 'reason': str}
    """
    response_lower = response.lower()
    
    # 判断是否是CS2或CSGO
    is_cs2_csgo = False
    
    # 检查肯定判断
    if any(keyword in response_lower for keyword in [
        '是cs2', '是csgo', '是反恐精英', '是counter-strike', 
        '是cs2或csgo', '是cs2或者csgo', '是csgo或cs2', '是csgo或者cs2',
        'cs2游戏', 'csgo游戏', 'counter-strike 2', 'counter-strike: global offensive'
    ]):
        # 检查是否有否定词
        if not any(neg in response_lower for neg in ['不是', '非', 'no', 'not', '否']):
            is_cs2_csgo = True
    
    # 检查否定判断
    if any(keyword in response_lower for keyword in [
        '不是cs2', '不是csgo', '不是反恐精英', '不是counter-strike',
        '不是cs2或csgo', '不是cs2或者csgo', '不是csgo或cs2', '不是csgo或者cs2'
    ]):
        is_cs2_csgo = False
    
    # 如果还没有明确判断，尝试从文本中提取更多信息
    if not is_cs2_csgo:
        # 检查是否包含CS2或CSGO相关关键词
        cs_keywords = ['cs2', 'csgo', 'counter-strike', '反恐精英', 'dust2', 'mirage', 'inferno']
        has_cs_keywords = any(keyword in response_lower for keyword in cs_keywords)
        
        if has_cs_keywords:
            # 如果包含CS关键词但没有否定词，可能是CS游戏
            if not any(neg in response_lower for neg in ['不是', '非', 'no', 'not', '否', '其他游戏', '别的游戏']):
                is_cs2_csgo = True
    
    # 提取理由（取前300个字符）
    reason = response.strip()
    if len(reason) > 300:
        reason = reason[:300] + '...'
    
    return {
        'is_cs2_csgo': is_cs2_csgo,
        'reason': reason
    }


def extract_frames_worker(args_tuple):
    """
    多进程工作函数：提取帧（CPU密集型任务）
    
    Args:
        args_tuple: (video_path, fps, max_frames)
    
    Returns:
        (video_path, frames) 或 (video_path, None, error)
    """
    video_path, fps, max_frames = args_tuple
    try:
        frames = extract_frames_from_video(video_path, fps, max_frames)
        return (video_path, frames, None)
    except Exception as e:
        return (video_path, None, str(e))


# 进程级模型缓存（每个进程独立，在spawn模式下每个进程有自己的副本）
# key: (gpu_id, model_name), value: (model, processor)
_worker_model_cache = {}


def inference_worker(args_tuple):
    """
    多进程模型推理工作函数（每个进程只加载一次模型，然后复用）
    
    Args:
        args_tuple: (video_path, frames, model_name, gpu_id, max_new_tokens)
    
    Returns:
        (video_path, judgment_result, error)
    """
    video_path, frames, model_name, gpu_id, max_new_tokens = args_tuple
    
    try:
        # 设置当前进程使用的GPU
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        
        # 检查模型是否已加载（使用gpu_id和model_name作为key）
        cache_key = (gpu_id, model_name)
        
        if cache_key not in _worker_model_cache:
            # 首次加载模型
            torch.cuda.empty_cache()
            model, processor = load_model(model_name, device, verbose=False, use_device_map=True)
            _worker_model_cache[cache_key] = (model, processor)
        else:
            # 复用已加载的模型
            model, processor = _worker_model_cache[cache_key]
            # 确保使用正确的GPU
            torch.cuda.set_device(gpu_id)
        
        # 生成判断结果
        judgment_result = judge_cs2_csgo_video(
            model, processor, frames, max_new_tokens
        )
        
        # 不删除模型，保留在缓存中供后续任务使用
        # 只在进程结束时自动清理
        
        return (video_path, judgment_result, None)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        # 即使出错也要清理缓存
        try:
            torch.cuda.empty_cache()
        except:
            pass
        return (video_path, {'is_cs2_csgo': False, 'reason': f'处理出错: {str(e)}'}, error_msg)


def judge_batch_videos(model, processor, batch_data: List[Tuple[str, List[Image.Image]]], 
                      max_new_tokens: int = 128) -> List[Dict[str, Any]]:
    """
    批量判断多个视频（目前逐个处理，因为模型可能不支持真正的批处理）
    
    Args:
        model: Qwen3-VL/Qwen2-VL模型
        processor: 处理器
        batch_data: [(video_path, frames), ...] 列表
        max_new_tokens: 最大生成token数
    
    Returns:
        判断结果列表
    """
    results = []
    for video_path, frames in batch_data:
        if frames:
            judgment = judge_cs2_csgo_video(model, processor, frames, max_new_tokens)
            results.append(judgment)
        else:
            results.append({'is_cs2_csgo': False, 'reason': '无法提取视频帧'})
    return results


def find_video_files(video_dir: str, extensions: list) -> list:
    """查找目录下的所有视频文件"""
    video_files = []
    video_dir_path = Path(video_dir)
    
    if not video_dir_path.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    
    for ext in extensions:
        # 支持大小写
        pattern_lower = f"*{ext.lower()}"
        pattern_upper = f"*{ext.upper()}"
        video_files.extend(video_dir_path.glob(pattern_lower))
        video_files.extend(video_dir_path.glob(pattern_upper))
    
    # 去重并排序
    video_files = sorted(list(set(video_files)))
    return [str(f) for f in video_files]


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
    print("使用Qwen3-VL判断视频是否是CS2或CSGO游戏视频（并行版本）")
    print("=" * 60)
    print(f"视频目录: {args.video_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print(f"并行进程数: {args.num_workers}")
    print(f"提取批次大小: {args.extract_batch_size} 个视频/批")
    print(f"判断批处理大小: {args.batch_size} 个视频/批")
    print(f"推理并行进程数: {args.inference_workers}")
    print(f"使用的GPU: {args.gpu_ids}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"提取帧率: {args.fps} fps")
    print(f"最大帧数: {args.max_frames}")
    if args.max_videos:
        print(f"最多处理视频数: {args.max_videos}")
    print("=" * 60)
    
    # 查找所有视频文件
    print(f"\n正在查找视频文件: {args.video_dir}...")
    try:
        video_files = find_video_files(args.video_dir, args.video_extensions)
    except Exception as e:
        print(f"错误: {e}")
        return
    
    if not video_files:
        print(f"错误: 在 {args.video_dir} 中未找到视频文件")
        return
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    print(f"找到 {len(video_files)} 个视频文件")
    
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
        # 多进程推理模式：不需要在主进程加载模型
        model = None
        processor = None
    else:
        print(f"单进程推理模式")
        # 单进程推理模式：在主进程加载模型
        print("\n正在加载模型...")
        model, processor = load_model(args.model_name, args.device)
    
    # 打开输出文件（追加模式，支持分批保存）
    output_file_handle = open(args.output_file, 'w', encoding='utf-8')
    
    all_results_count = 0
    cs2_count = 0
    non_cs2_count = 0
    failed_count = 0
    total_processed = 0
    
    # 分批处理视频文件
    print(f"\n开始分批处理，共 {len(video_files)} 个视频")
    
    for batch_idx in range(0, len(video_files), args.extract_batch_size):
        batch_video_files = video_files[batch_idx:batch_idx + args.extract_batch_size]
        batch_num = (batch_idx // args.extract_batch_size) + 1
        total_batches = (len(video_files) + args.extract_batch_size - 1) // args.extract_batch_size
        
        print(f"\n{'='*60}")
        print(f"处理批次 {batch_num}/{total_batches} ({len(batch_video_files)} 个视频)")
        print(f"{'='*60}")
        
        # 第一步：并行提取当前批次的帧
        print(f"\n步骤1: 提取帧...")
        extract_tasks = [
            (str(video_path), args.fps, args.max_frames)
            for video_path in batch_video_files
        ]
        
        video_frames_map = {}  # {video_path: frames}
        failed_extractions = []
        
        with Pool(processes=args.num_workers) as pool:
            with tqdm.tqdm(total=len(extract_tasks), desc=f"  提取帧 (批次 {batch_num})", leave=False) as pbar:
                for video_path, frames, error in pool.imap(extract_frames_worker, extract_tasks):
                    if error is None and frames:
                        video_frames_map[video_path] = frames
                    else:
                        failed_extractions.append((video_path, error or "无法提取帧"))
                    pbar.update(1)
        
        print(f"  成功提取 {len(video_frames_map)} 个视频的帧，失败 {len(failed_extractions)} 个")
        
        # 第二步：批处理生成判断结果
        if video_frames_map:
            print(f"\n步骤2: 判断是否是CS2或CSGO视频...")
            video_paths = list(video_frames_map.keys())
            
            if args.inference_workers > 1:
                # 多进程推理模式
                # 准备推理任务
                inference_tasks = []
                for i, video_path in enumerate(video_paths):
                    gpu_id = gpu_ids[i % len(gpu_ids)]  # 轮询分配GPU
                    inference_tasks.append((
                        video_path,
                        video_frames_map[video_path],
                        args.model_name,
                        gpu_id,
                        args.max_new_tokens
                    ))
                
                # 使用多进程并行推理
                # 每个worker进程会使用进程级缓存，第一次遇到某个GPU的任务时加载模型，
                # 后续遇到相同GPU的任务时会复用模型，避免重复加载
                with Pool(processes=args.inference_workers) as pool:
                    with tqdm.tqdm(total=len(inference_tasks), desc=f"  判断视频 (批次 {batch_num})", leave=False) as pbar:
                        for video_path, judgment_result, error in pool.imap(inference_worker, inference_tasks):
                            video_filename = Path(video_path).name
                            
                            if error is None and judgment_result:
                                is_cs2_csgo = judgment_result.get('is_cs2_csgo', False)
                                output_result = {
                                    'video_path': video_path,
                                    'video_filename': video_filename,
                                    'is_cs2_csgo': is_cs2_csgo,
                                    'reason': judgment_result.get('reason', '')
                                }
                                output_file_handle.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                                output_file_handle.flush()  # 立即刷新到磁盘
                                all_results_count += 1
                                
                                if is_cs2_csgo:
                                    cs2_count += 1
                                else:
                                    non_cs2_count += 1
                            else:
                                failed_count += 1
                                if error:
                                    print(f"\n  判断失败: {video_filename} - {error[:100]}")
                                else:
                                    print(f"\n  判断失败: {video_filename}")
                            pbar.update(1)
            else:
                # 单进程推理模式（原有逻辑）
                for i in tqdm.tqdm(range(0, len(video_paths), args.batch_size), 
                                  desc=f"  判断视频 (批次 {batch_num})", leave=False):
                    batch_paths = video_paths[i:i + args.batch_size]
                    batch_data = [(path, video_frames_map[path]) for path in batch_paths]
                    
                    # 生成判断结果
                    judgment_results = judge_batch_videos(model, processor, batch_data, args.max_new_tokens)
                    
                    # 立即保存结果到文件
                    for video_path, judgment_result in zip(batch_paths, judgment_results):
                        video_filename = Path(video_path).name
                        
                        if judgment_result:
                            is_cs2_csgo = judgment_result.get('is_cs2_csgo', False)
                            output_result = {
                                'video_path': video_path,
                                'video_filename': video_filename,
                                'is_cs2_csgo': is_cs2_csgo,
                                'reason': judgment_result.get('reason', '')
                            }
                            output_file_handle.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                            output_file_handle.flush()  # 立即刷新到磁盘
                            all_results_count += 1
                            
                            if is_cs2_csgo:
                                cs2_count += 1
                            else:
                                non_cs2_count += 1
                        else:
                            failed_count += 1
                            print(f"\n  判断失败: {video_filename}")
        
        # 记录提取帧失败的视频
        for video_path, error in failed_extractions:
            failed_count += 1
            video_filename = Path(video_path).name
            print(f"\n  提取帧失败: {video_filename} - {error}")
        
        total_processed += len(batch_video_files)
        
        # 清理当前批次的帧数据，释放内存
        del video_frames_map
        if args.inference_workers > 1:
            # 多进程模式下，每个进程会自己清理GPU缓存
            pass
        elif args.device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n批次 {batch_num} 完成: CS2/CSGO视频 {cs2_count} 个，非CS2/CSGO视频 {non_cs2_count} 个，失败 {len(failed_extractions)} 个")
    
    # 关闭输出文件
    output_file_handle.close()
    
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"共处理 {len(video_files)} 个视频")
    print(f"成功判断: {all_results_count} 个")
    print(f"  - CS2/CSGO视频: {cs2_count} 个")
    print(f"  - 非CS2/CSGO视频: {non_cs2_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"输出文件: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

