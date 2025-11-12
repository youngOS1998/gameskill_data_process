import json
import argparse
import functools
import tqdm
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

def get_args():
    parser = argparse.ArgumentParser(description='优化版本：加速视频处理')
    parser.add_argument('--inputs', type=str, default='fp_processed.jsonl')
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--min_clip_sec', type=int, default=5)
    parser.add_argument('--max_clip_sec', type=int, default=15)
    parser.add_argument('--max_empty_sec', type=int, default=2)
    parser.add_argument('--min_wps', type=int, default=1)
    parser.add_argument('--max_wps', type=int, default=4)
    parser.add_argument('--output', type=str, default='gameskill_1_train.jsonl')
    parser.add_argument('--video_dir', type=str, default='processed_bilibili_cs2')
    parser.add_argument('--output_video_dir', type=str, default='videos_gameskill_1')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/Projects/qwen_gameskill/qwen-vl-finetune/dataset', help='视频文件的绝对路径，如果为空则使用当前工作目录')
    parser.add_argument('--skip_video_cut', action='store_true', help='跳过视频切割，只生成数据文件')
    parser.add_argument('--num_workers', type=int, default=0, help='并行处理的工作进程数，0表示使用CPU核心数')
    parser.add_argument('--ffmpeg_preset', type=str, default='ultrafast', help='ffmpeg编码预设：ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow')
    parser.add_argument('--try_copy_first', action='store_true', default=True, help='先尝试copy模式（如果兼容会快很多）')
    parser.add_argument('--crf', type=int, default=28, help='CRF质量控制（18-28，数值越大质量越低但速度越快，默认28）')
    # 测试参数
    parser.add_argument('--max_videos', type=int, default=10000, help='最多处理多少个视频（测试用）')
    parser.add_argument('--max_clips_per_video', type=int, default=10000, help='每个视频最多生成多少个片段（测试用）')
    return parser.parse_args()

def split2words(datum: dict):
    subtitles = datum.pop('subtitles')
    content = []
    for start, end, subtitle in subtitles:
        if '[' in subtitle or ']' in subtitle:
            continue
        words = []
        for word in subtitle.split(' '):
            if not words or words[-1] != word:
                words.append(word)
        if len(words) > 0:  # 确保有单词
            duration = end - start
            duration_per_word = duration / len(words)
            for i, word in enumerate(words):
                content.append([round(start + i * duration_per_word, 1), round(start + (i+1) * duration_per_word, 1), word])
    datum['content'] = content
    return datum

def clip4pretrain(datum: dict, args):
    words, title = datum['content'], datum['title']
    clips, contexts, i = [], [], 0
    while i < len(words):
        j = None
        for j in range(i+1, len(words)):
            if words[j][1] - words[i][1] > args.max_clip_sec:
                break
            if words[j][1] - words[j-1][1] > args.max_empty_sec:
                break
        if j is not None and j > i and words[j-1][1] - words[i][1] >= args.min_clip_sec:
            clips.append(words[i:j])
            contexts.append(' '.join(word[2] for word in words[:i]))
        if j is not None:
            i = j
        else:
            break
    return [{
        'video': datum['video'], 
        'content': clip, 
        'previous': context, 
        'title': title, 
        'category': datum['category'],
        'start_time': clip[0][0],
        'end_time': clip[-1][1]
    } for clip, context in zip(clips, contexts)]

def check(datum: dict, args):
    subtitles = datum['content']
    if len(subtitles) == 0:
        return False
    duration = subtitles[-1][1] - subtitles[0][1]
    if duration <= 0:
        return False
    wps = len(subtitles) / duration
    if wps < args.min_wps or wps > args.max_wps:
        return False
    return True

def process(datum: dict, args):
    datum = split2words(datum)
    clips_datum = clip4pretrain(datum, args)
    clips_datum = [clip_datum for clip_datum in clips_datum if check(clip_datum, args)]
    # 测试版本：限制每个视频的片段数量
    if len(clips_datum) > args.max_clips_per_video:
        clips_datum = clips_datum[:args.max_clips_per_video]
        print(f"  测试模式：视频 {datum['video']} 限制为 {args.max_clips_per_video} 个片段")
    return clips_datum

def simple_mt(items, func, desc='Processing'):
    """
    简单的多线程处理替代函数
    """
    results = []
    for item in tqdm.tqdm(items, desc=desc):
        try:
            result = func(item)
            results.append(result)
        except Exception as e:
            print(f"处理项目时出错: {e}")
            results.append(None)
    return results

def generate_video_filename(video_id: str, start_time: float, end_time: float) -> str:
    """根据视频ID和时间戳生成视频文件名"""
    # 格式: video_id_start-end_2.0fps.mp4
    # 例如: av1000237410__0.50-15.30_2.0fps.mp4
    return f"{video_id}_{start_time:.2f}-{end_time:.2f}_2.0fps.mp4"

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

def process_single_clip(args_tuple):
    """处理单个clip的辅助函数，用于多进程"""
    clip_datum, args = args_tuple
    try:
        # 转换为live_cc格式
        live_cc_data, video_filename, start_time, end_time = convert_to_live_cc_format(clip_datum, args)
        
        # 切割视频（如果未跳过）
        video_cut_success = False
        if not args.skip_video_cut:
            video_id = clip_datum['video']
            input_video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
            output_video_path = os.path.join(args.output_video_dir, video_filename)
            
            if os.path.exists(input_video_path):
                video_cut_success = cut_video(
                    input_video_path, output_video_path, start_time, end_time, 
                    args.ffmpeg_preset, args.crf, args.num_workers, args.try_copy_first
                )
        
        return {
            'success': True,
            'live_cc_data': live_cc_data,
            'video_cut_success': video_cut_success,
            'video_id': clip_datum.get('video', ''),
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'live_cc_data': None,
            'video_cut_success': False,
            'video_id': clip_datum.get('video', ''),
            'error': str(e)
        }

def convert_to_live_cc_format(clip_datum: dict, args) -> tuple:
    """将clip数据转换为live_cc_train.jsonl格式"""
    video_id = clip_datum['video']
    content = clip_datum['content']
    previous = clip_datum['previous']
    title = clip_datum['title']
    category = clip_datum['category']
    start_time = clip_datum['start_time']
    end_time = clip_datum['end_time']
    
    # 生成解说文本（从content中提取所有单词）
    commentary = ' '.join([word[2] for word in content])
    
    # 生成视频文件名
    video_filename = generate_video_filename(video_id, start_time, end_time)
    # 使用 args.output_video_dir 确保路径一致
    video_path = f"{args.output_video_dir}/{video_filename}"
    
    # 构建之前的解说内容（用于human提示）
    previous_text = previous if previous else "..."
    if len(previous_text) > 500:  # 限制长度
        previous_text = "..." + previous_text[-500:]
    
    # 根据类别生成不同的提示词
    if 'cs2' in category.lower() or 'game' in category.lower():
        instruction = "请观看这个游戏视频片段，并给出专业的游戏解说和分析，包括战术策略、操作技巧和关键决策。"
    else:
        instruction = "请观看视频并给出解说: Provide a commentary on a mechanical repair process, highlighting specific measurements and steps involved."
    
    # 构建human提示
    human_value = f"<video>\n视频标题: {title}\n类别: {category}"
    if previous_text and previous_text != "...":
        human_value += f"\n之前的解说内容: {previous_text}"
    human_value += f"\n请观看视频并给出解说: {instruction}"
    
    # 构建metadata
    metadata = {
        "title": title,
        "category": category,
        "video_start": start_time,
        "video_end": end_time,
        "duration": end_time - start_time,
        "source_file": f"{video_id}.json"
    }
    
    # 构建live_cc格式数据
    live_cc_data = {
        "video": video_path,
        "data_path": args.data_path if args.data_path else os.path.abspath(args.output_video_dir),
        "conversations": [
            {
                "from": "human",
                "value": human_value
            },
            {
                "from": "gpt",
                "value": commentary
            }
        ],
        "metadata": metadata
    }
    
    return live_cc_data, video_filename, start_time, end_time

if __name__ == '__main__':
    args = get_args()
    index, total = args.part.split('/')
    index, total = int(index), int(total)
    
    # 设置data_path默认值
    if not args.data_path:
        args.data_path = os.path.abspath(args.output_video_dir)

    print('=' * 60)
    print('优化版本：加速视频处理')
    print(f'最多处理视频数: {args.max_videos}')
    print(f'每个视频最多片段数: {args.max_clips_per_video}')
    print(f'先尝试copy模式: {args.try_copy_first}')
    print('=' * 60)

    print(f'正在读取 {args.inputs}...')
    with open(args.inputs, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'读取完成，共 {len(lines)} 行')
    
    print('正在解析JSON...')
    datums = []
    for line in lines:
        try:
            datum = json.loads(line.strip())
            datums.append(datum)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            continue
    
    print(f'成功解析 {len(datums)} 条记录')
    
    # 分片处理
    datums = datums[index-1::total]
    print(f'当前分片处理 {len(datums)} 条记录 (分片 {index}/{total})')
    
    # 测试版本：限制处理的视频数量
    if len(datums) > args.max_videos:
        datums = datums[:args.max_videos]
        print(f'测试模式：限制为前 {args.max_videos} 个视频')
    
    print('正在处理数据...')
    clips_datums = simple_mt(datums, functools.partial(process, args=args), '处理视频数据')
    
    # 创建输出视频目录
    if not args.skip_video_cut:
        os.makedirs(args.output_video_dir, exist_ok=True)
    
    # 检查ffmpeg是否可用
    ffmpeg_available = check_ffmpeg_available()
    if not args.skip_video_cut and not ffmpeg_available:
        print("警告: ffmpeg 未安装，将跳过视频切割，只生成数据文件")
        print("提示: 可以运行 'apt install -y ffmpeg' 安装 ffmpeg，或使用 --skip_video_cut 参数明确跳过视频切割")
        args.skip_video_cut = True
    
    # 准备所有需要处理的clip
    all_clips = []
    for clips_datum in clips_datums:
        if clips_datum:
            for clip_datum in clips_datum:
                all_clips.append((clip_datum, args))
    
    print(f'共需要处理 {len(all_clips)} 个视频片段')
    print(f'正在转换格式{"并切割视频" if not args.skip_video_cut else "（跳过视频切割）"}...')
    
    # 确定并行工作进程数
    if args.num_workers == 0:
        num_workers = max(1, cpu_count() - 1)  # 保留一个核心
    else:
        num_workers = args.num_workers
    
    if not args.skip_video_cut:
        print(f'使用 {num_workers} 个并行进程处理视频切割')
        print(f'ffmpeg编码预设: {args.ffmpeg_preset}')
        print(f'CRF质量控制: {args.crf} (数值越大速度越快，质量稍低)')
        print(f'先尝试copy模式: {args.try_copy_first}')
        total_cpus = cpu_count()
        threads_per_process = max(2, min(4, max(1, total_cpus // num_workers))) if num_workers > 0 else 2
        print(f'每个ffmpeg进程使用 {threads_per_process} 个线程')
    
    total_clips = 0
    video_cut_success = 0
    video_cut_failed = 0
    processed_videos = set()  # 记录已处理的视频
    
    # 使用多进程处理
    if not args.skip_video_cut and num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_single_clip, all_clips),
                total=len(all_clips),
                desc='转换格式和切割视频'
            ))
    else:
        # 单进程模式
        results = []
        for clip_tuple in tqdm.tqdm(all_clips, desc='转换格式和切割视频'):
            results.append(process_single_clip(clip_tuple))
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            if result['success']:
                if result['live_cc_data']:
                    f.write(json.dumps(result['live_cc_data'], ensure_ascii=False) + '\n')
                    total_clips += 1
                    if result['video_cut_success']:
                        video_cut_success += 1
                        if result['video_id']:
                            processed_videos.add(result['video_id'])
                    else:
                        video_cut_failed += 1
            else:
                if result['error']:
                    print(f"处理clip时出错: {result['error']}")
                video_cut_failed += 1
    
    print('=' * 60)
    print(f'处理完成！')
    print(f'共生成 {total_clips} 个视频片段')
    if not args.skip_video_cut:
        print(f'视频切割成功: {video_cut_success} 个')
        print(f'视频切割失败: {video_cut_failed} 个')
        print(f'处理的视频数: {len(processed_videos)} 个')
        print(f'输出视频目录: {args.output_video_dir}')
        
        # 验证文件是否存在
        print('\n验证生成的文件...')
        missing_files = 0
        with open(args.output, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    video_path = data.get('video', '')
                    if video_path:
                        # 提取文件名
                        video_filename = os.path.basename(video_path)
                        full_path = os.path.join(args.output_video_dir, video_filename)
                        if not os.path.exists(full_path):
                            missing_files += 1
                            if missing_files <= 5:  # 只显示前5个缺失的文件
                                print(f"  警告: 文件不存在 - {video_filename}")
                except:
                    continue
        
        if missing_files == 0:
            print(f'✓ 所有 {total_clips} 个视频文件都已成功生成！')
        else:
            print(f'✗ 有 {missing_files} 个视频文件缺失')
    else:
        print(f'已跳过视频切割（使用 --skip_video_cut 或 ffmpeg 未安装）')
    print(f'输出文件: {args.output}')
    print('=' * 60)

