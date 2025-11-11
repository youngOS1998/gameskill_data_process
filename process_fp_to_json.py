import os
import json
import re
from pathlib import Path

def extract_video_id_from_filename(filename):
    """
    从文件名中提取视频ID（去掉扩展名）
    """
    return os.path.splitext(filename)[0]

def extract_title_from_filename(filename):
    """
    从文件名中提取标题信息
    由于文件名已经被重命名，我们使用视频ID作为标题
    """
    video_id = extract_video_id_from_filename(filename)
    return f"{video_id}"

def process_subtitle_data(subtitle_data):
    """
    处理字幕数据，转换为pretrain_to_clips.py需要的格式
    输入格式：[{"text": "文本", "start": 开始时间(毫秒), "end": 结束时间(毫秒), "timestamp": [...]}, ...]
    输出格式：[[开始时间(秒), 结束时间(秒), "字幕文本"], ...]
    """
    subtitles = []
    
    for item in subtitle_data:
        text = item.get('text', '').strip()
        if not text:  # 跳过空文本
            continue
            
        # 将毫秒转换为秒，保留1位小数
        start_time = round(item.get('start', 0) / 1000.0, 1)
        end_time = round(item.get('end', 0) / 1000.0, 1)
        
        # 过滤掉包含方括号的字幕（通常是背景音或非对话内容）
        if '[' in text or ']' in text:
            continue
            
        subtitles.append([start_time, end_time, text])
    
    return subtitles

def process_single_video(video_id, json_file_path):
    """
    处理单个视频的JSON文件 
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)
        
        # 处理字幕数据
        subtitles = process_subtitle_data(subtitle_data)
        
        if not subtitles:  # 如果没有有效字幕，跳过
            return None
        
        # 构建输出数据
        result = {
            "video": video_id,  # 使用视频ID作为视频标识
            "title": extract_title_from_filename(video_id),
            "category": "cs2_web_data",  # 固定分类
            "subtitles": subtitles
        }
        
        return result
        
    except Exception as e:
        print(f"处理文件 {json_file_path} 时出错: {str(e)}")
        return None

def process_fp_directory(fp_dir="fp", output_file="fp_processed.json"):
    """
    处理fp目录下的所有JSON文件，生成统一的JSON文件
    """
    if not os.path.exists(fp_dir):
        print(f"错误：目录 {fp_dir} 不存在")
        return
    
    # 获取所有JSON文件
    json_files = []
    for file in os.listdir(fp_dir):
        if file.endswith('.json'):
            json_files.append(file)
    
    if not json_files:
        print(f"在目录 {fp_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理所有JSON文件
    processed_data = []
    success_count = 0
    
    for json_file in json_files:
        video_id = extract_video_id_from_filename(json_file)
        json_path = os.path.join(fp_dir, json_file)
        
        print(f"处理: {json_file}")
        result = process_single_video(video_id, json_path)
        
        if result:
            processed_data.append(result)
            success_count += 1
            print(f"  ✓ 成功处理，包含 {len(result['subtitles'])} 条字幕")
        else:
            print(f"  ✗ 处理失败或跳过")
    
    # 保存结果
    if processed_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！")
        print(f"成功处理: {success_count}/{len(json_files)} 个文件")
        print(f"输出文件: {output_file}")
        print(f"总字幕条数: {sum(len(item['subtitles']) for item in processed_data)}")
    else:
        print("没有成功处理任何文件")

def preview_processing(fp_dir="fp", max_files=3):
    """
    预览处理结果，只处理前几个文件
    """
    if not os.path.exists(fp_dir):
        print(f"错误：目录 {fp_dir} 不存在")
        return
    
    json_files = [f for f in os.listdir(fp_dir) if f.endswith('.json')][:max_files]
    
    if not json_files:
        print(f"在目录 {fp_dir} 中没有找到JSON文件")
        return
    
    print(f"预览处理前 {len(json_files)} 个文件:")
    print("=" * 60)
    
    for json_file in json_files:
        video_id = extract_video_id_from_filename(json_file)
        json_path = os.path.join(fp_dir, json_file)
        
        print(f"\n文件: {json_file}")
        result = process_single_video(video_id, json_path)
        
        if result:
            print(f"视频ID: {result['video']}")
            print(f"标题: {result['title']}")
            print(f"分类: {result['category']}")
            print(f"字幕条数: {len(result['subtitles'])}")
            
            # 显示前3条字幕作为示例
            print("前3条字幕示例:")
            for i, subtitle in enumerate(result['subtitles'][:3]):
                print(f"  {i+1}. [{subtitle[0]}s - {subtitle[1]}s] {subtitle[2]}")
            
            if len(result['subtitles']) > 3:
                print(f"  ... 还有 {len(result['subtitles']) - 3} 条字幕")
        else:
            print("处理失败")

if __name__ == "__main__":
    print("FP目录文件处理工具")
    print("=" * 50)
    print("功能：将fp目录下的JSON文件转换为pretrain_to_clips.py需要的格式")
    
    # 设置目录和输出文件
    fp_directory = "processed_bilibili_cs2"
    output_filename = "fp_processed.json"
    
    print(f"输入目录: {fp_directory}")
    print(f"输出文件: {output_filename}")
    
    # 选择操作
    print("\n选择操作：")
    print("1) 预览处理结果（只处理前3个文件）")
    print("2) 执行完整处理")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        print("\n预览模式：")
        preview_processing(fp_directory)
    elif choice == "2":
        print("\n执行完整处理：")
        process_fp_directory(fp_directory, output_filename)
    else:
        print("无效选择，退出程序")
