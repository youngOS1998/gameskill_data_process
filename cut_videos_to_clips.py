import argparse
import os
import random
import subprocess
import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

def get_args():
    parser = argparse.ArgumentParser(description='将视频切割成6~15秒的片段')
    parser.add_argument('--video_dir', type=str, default='processed_bilibili_cs2', 
                       help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default='video_clips_6_15s', 
                       help='输出视频片段目录')
    parser.add_argument('--min_clip_sec', type=float, default=6.0, 
                       help='最小片段时长（秒）')
    parser.add_argument('--max_clip_sec', type=float, default=15.0, 
                       help='最大片段时长（秒）')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='并行处理的工作进程数，0表示使用CPU核心数-1')
    parser.add_argument('--ffmpeg_preset', type=str, default='ultrafast', 
                       help='ffmpeg编码预设：ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow')
    parser.add_argument('--try_copy_first', action='store_true', default=True, 
                       help='先尝试copy模式（如果兼容会快很多）')
    parser.add_argument('--crf', type=int, default=28, 
                       help='CRF质量控制（18-28，数值越大质量越低但速度越快，默认28）')
    parser.add_argument('--video_extensions', type=str, nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv'], 
                       help='支持的视频文件扩展名')
    parser.add_argument('--max_videos', type=int, default=None, 
                       help='最多处理多少个视频（测试用，None表示处理所有）')
    return parser.parse_args()

def check_ffmpeg_available() -> bool:
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_ffprobe_available() -> bool:
    """检查ffprobe是否可用"""
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    if not check_ffprobe_available():
        raise RuntimeError("ffprobe 不可用，无法获取视频时长")
    
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            raise RuntimeError(f"无法获取视频时长: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("获取视频时长超时")
    except ValueError:
        raise RuntimeError("无法解析视频时长")

def verify_video_file(video_path: str) -> bool:
    """快速验证视频文件是否有效（检查是否可以读取）"""
    if not os.path.exists(video_path):
        return False
    if os.path.getsize(video_path) == 0:
        return False
    
    # 使用ffprobe快速检查（如果可用）
    if check_ffprobe_available():
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=codec_name,width,height', 
                   '-of', 'default=noprint_wrappers=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
            return result.returncode == 0 and 'codec_name' in result.stdout
        except:
            pass
    
    # 如果ffprobe不可用，至少检查文件大小是否合理（大于1KB）
    return os.path.getsize(video_path) > 1024

def cut_video(input_video: str, output_video: str, start_time: float, end_time: float, 
              preset: str = 'ultrafast', crf: int = 28, num_workers: int = 1, try_copy: bool = True) -> bool:
    """
    优化版本的视频切割函数
    1. 先尝试copy模式（如果兼容会快很多）
    2. 优化-ss参数位置（放在-i之前）
    3. 使用更快的编码参数
    """
    # 检查ffmpeg是否可用
    if not check_ffmpeg_available():
        return False
    
    try:
        duration = end_time - start_time
        
        # 优化1: 先尝试 copy 模式（如果源视频格式兼容，会快很多，几乎瞬间完成）
        if try_copy:
            cmd_copy = [
                'ffmpeg', '-ss', str(start_time),  # 放在 -i 之前，更快定位起始位置
                '-i', input_video,
                '-t', str(duration),
                '-c', 'copy',  # copy模式，不重新编码，速度极快
                '-avoid_negative_ts', 'make_zero',
                '-y',
                '-loglevel', 'error',
                output_video
            ]
            result = subprocess.run(cmd_copy, capture_output=True, text=True, check=False, timeout=300)
            if result.returncode == 0:
                # copy模式成功，验证文件是否有效
                if verify_video_file(output_video):
                    return True  # copy模式成功且文件有效
        
        # 优化2: copy模式失败或格式不兼容，使用重新编码
        # 计算每个进程应该使用的线程数
        total_cpus = cpu_count()
        if num_workers > 0:
            # 每个进程使用2-4个线程，根据总CPU数和进程数动态调整
            threads_per_process = max(2, min(4, max(1, total_cpus // num_workers)))
        else:
            threads_per_process = max(2, min(4, total_cpus // 4))
        
        cmd = [
            'ffmpeg', '-ss', str(start_time),  # 放在 -i 之前，更快定位
            '-i', input_video,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', preset,  # 使用最快的preset
            '-crf', str(crf),   # 使用更高的CRF值以换取速度（28比23快很多）
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac',
            '-b:a', '96k',      # 降低音频比特率以加速（96k足够）
            '-threads', str(threads_per_process),  # 每个进程使用多个线程
            '-avoid_negative_ts', 'make_zero',
            '-y',
            '-loglevel', 'error',
            output_video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
        if result.returncode != 0:
            return False
        
        # 验证输出文件
        return verify_video_file(output_video)
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def generate_clip_segments(video_duration: float, min_clip_sec: float, 
                          max_clip_sec: float) -> list:
    """
    生成视频切割片段的时间段列表（连续且不重叠）
    每个片段的长度在min_clip_sec到max_clip_sec之间随机选择
    返回: [(start_time, end_time), ...]
    """
    segments = []
    start_time = 0.0
    
    while start_time < video_duration:
        # 计算剩余时间
        remaining_time = video_duration - start_time
        
        # 如果剩余时间不足最小片段时长，停止
        if remaining_time < min_clip_sec:
            break
        
        # 随机选择片段长度（在min_clip_sec和max_clip_sec之间）
        # 但不能超过剩余时间
        max_possible_duration = min(max_clip_sec, remaining_time)
        clip_duration = random.uniform(min_clip_sec, max_possible_duration)
        
        # 计算当前片段的结束时间
        end_time = start_time + clip_duration
        duration = end_time - start_time
        
        # 确保片段时长满足最小要求
        if duration >= min_clip_sec:
            segments.append((start_time, end_time))
            # 下一个片段从当前片段的结束时间开始（连续且不重叠）
            start_time = end_time
        else:
            # 如果计算出的时长不满足要求，停止
            break
    
    return segments

def process_single_video(args_tuple):
    """处理单个视频的辅助函数，用于多进程"""
    video_path, args = args_tuple
    try:
        video_name = Path(video_path).stem
        video_ext = Path(video_path).suffix
        
        # 获取视频时长
        try:
            duration = get_video_duration(video_path)
        except Exception as e:
            return {
                'success': False,
                'video_path': video_path,
                'clips_created': 0,
                'clips_failed': 0,
                'error': f"获取视频时长失败: {str(e)}"
            }
        
        # 生成切割片段（连续且不重叠）
        segments = generate_clip_segments(
            duration, args.min_clip_sec, args.max_clip_sec
        )
        
        if not segments:
            return {
                'success': True,
                'video_path': video_path,
                'clips_created': 0,
                'clips_failed': 0,
                'error': None,
                'message': '视频时长不足，无法生成片段'
            }
        
        # 切割视频
        clips_created = 0
        clips_failed = 0
        
        for idx, (start_time, end_time) in enumerate(segments):
            # 生成输出文件名
            output_filename = f"{video_name}_clip{idx+1:04d}_{start_time:.2f}-{end_time:.2f}{video_ext}"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(output_path):
                clips_created += 1
                continue
            
            # 切割视频
            success = cut_video(
                video_path, output_path, start_time, end_time,
                args.ffmpeg_preset, args.crf, args.num_workers, args.try_copy_first
            )
            
            if success:
                clips_created += 1
            else:
                clips_failed += 1
        
        return {
            'success': True,
            'video_path': video_path,
            'clips_created': clips_created,
            'clips_failed': clips_failed,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'video_path': video_path,
            'clips_created': 0,
            'clips_failed': 0,
            'error': str(e)
        }

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

if __name__ == '__main__':
    args = get_args()
    
    print('=' * 60)
    print('视频切割工具：将视频切割成6~15秒的片段')
    print('=' * 60)
    
    # 检查ffmpeg和ffprobe
    ffmpeg_available = check_ffmpeg_available()
    ffprobe_available = check_ffprobe_available()
    
    if not ffmpeg_available:
        print("错误: ffmpeg 未安装，无法进行视频切割")
        print("提示: 可以运行 'apt install -y ffmpeg' 安装 ffmpeg")
        exit(1)
    
    if not ffprobe_available:
        print("错误: ffprobe 未安装，无法获取视频时长")
        print("提示: 可以运行 'apt install -y ffmpeg' 安装 ffprobe（通常与ffmpeg一起安装）")
        exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有视频文件
    print(f'正在查找视频文件: {args.video_dir}...')
    try:
        video_files = find_video_files(args.video_dir, args.video_extensions)
    except Exception as e:
        print(f"错误: {e}")
        exit(1)
    
    print(f'找到 {len(video_files)} 个视频文件')
    
    # 限制处理的视频数量（测试用）
    if args.max_videos and len(video_files) > args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f'测试模式：限制为前 {args.max_videos} 个视频')
    
    # 确定并行工作进程数
    if args.num_workers == 0:
        num_workers = max(1, cpu_count() - 1)  # 保留一个核心
    else:
        num_workers = args.num_workers
    
    print(f'使用 {num_workers} 个并行进程处理视频切割')
    print(f'片段时长范围: {args.min_clip_sec}~{args.max_clip_sec} 秒（随机）')
    print(f'片段模式: 连续且不重叠，每个片段长度随机')
    print(f'ffmpeg编码预设: {args.ffmpeg_preset}')
    print(f'CRF质量控制: {args.crf} (数值越大速度越快，质量稍低)')
    print(f'先尝试copy模式: {args.try_copy_first}')
    total_cpus = cpu_count()
    threads_per_process = max(2, min(4, max(1, total_cpus // num_workers))) if num_workers > 0 else 2
    print(f'每个ffmpeg进程使用 {threads_per_process} 个线程')
    print('=' * 60)
    
    # 准备处理任务
    video_tasks = [(video_file, args) for video_file in video_files]
    
    # 使用多进程处理
    total_clips_created = 0
    total_clips_failed = 0
    videos_processed = 0
    videos_failed = 0
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_single_video, video_tasks),
                total=len(video_tasks),
                desc='切割视频'
            ))
    else:
        # 单进程模式
        results = []
        for task in tqdm.tqdm(video_tasks, desc='切割视频'):
            results.append(process_single_video(task))
    
    # 统计结果
    for result in results:
        if result['success']:
            videos_processed += 1
            total_clips_created += result['clips_created']
            total_clips_failed += result['clips_failed']
            if result.get('error'):
                print(f"警告: {Path(result['video_path']).name} - {result['error']}")
        else:
            videos_failed += 1
            print(f"错误: {Path(result['video_path']).name} - {result.get('error', '未知错误')}")
    
    print('=' * 60)
    print(f'处理完成！')
    print(f'处理的视频数: {videos_processed} 个')
    print(f'失败的视频数: {videos_failed} 个')
    print(f'成功生成的片段: {total_clips_created} 个')
    print(f'失败的片段: {total_clips_failed} 个')
    print(f'输出目录: {args.output_dir}')
    print('=' * 60)

