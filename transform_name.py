import os
import re
import shutil

def transform_and_collect_files(source_dir, target_dir):
    """
    递归处理指定目录下的所有视频和JSON文件，去除文件名中的中文部分和方括号，
    只保留方括号内的字母标识部分，并将处理后的文件汇总到目标目录
    
    Args:
        source_dir (str): 源目录路径（要处理的目录）
        target_dir (str): 目标目录路径（汇总存储的位置）
    """
    if not os.path.exists(source_dir):
        print(f"错误：源目录 {source_dir} 不存在")
        return
    
    # 创建目标目录（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标目录: {target_dir}")
    
    # 用于存储处理信息的列表
    process_info = []
    skipped_files = []
    
    # 支持的文件扩展名
    supported_extensions = ['.mp4', '.json']
    
    # 正则表达式模式：匹配文件名格式：任意内容[字母数字下划线标识].扩展名
    pattern = r'^.*?\[([A-Za-z0-9_]+)\]\.([^.]+)$'
    
    # 递归遍历所有文件
    print(f"\n开始扫描目录: {source_dir}")
    print("-" * 80)
    
    for root, dirs, files in os.walk(source_dir):
        # 获取相对路径用于显示
        rel_path = os.path.relpath(root, source_dir)
        if rel_path == '.':
            display_path = source_dir
        else:
            display_path = os.path.join(source_dir, rel_path)
        
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # 只处理支持的文件类型
            _, ext = os.path.splitext(filename)
            if ext.lower() not in supported_extensions:
                continue
            
            # 使用正则表达式匹配文件名格式
            match = re.match(pattern, filename)
            
            if match:
                # 提取方括号中的标识和文件扩展名
                identifier = match.group(1)  # 方括号内的标识部分（不含方括号）
                extension = match.group(2)   # 文件扩展名
                
                # 生成新的文件名：标识 + 扩展名
                new_filename = f"{identifier}.{extension}"
                target_file_path = os.path.join(target_dir, new_filename)
                
                # 检查目标文件是否已存在（如果存在，添加源文件夹前缀以避免冲突）
                if os.path.exists(target_file_path):
                    # 如果文件已存在，添加源文件夹名称作为前缀
                    folder_name = os.path.basename(root) if rel_path != '.' else 'root'
                    new_filename = f"{folder_name}_{identifier}.{extension}"
                    target_file_path = os.path.join(target_dir, new_filename)
                    print(f"警告：文件 {identifier}.{extension} 已存在，使用新名称: {new_filename}")
                
                process_info.append({
                    'old_path': file_path,
                    'old_name': filename,
                    'new_name': new_filename,
                    'target_path': target_file_path,
                    'source_folder': display_path
                })
            else:
                skipped_files.append({
                    'path': file_path,
                    'name': filename,
                    'reason': '不符合命名格式（未找到方括号标识）'
                })
    
    # 显示处理信息
    if process_info:
        print(f"\n找到 {len(process_info)} 个文件需要处理：")
        print("-" * 80)
        
        # 显示前10个示例
        preview_count = min(10, len(process_info))
        for i, info in enumerate(process_info[:preview_count], 1):
            print(f"{i}. 源文件: {info['source_folder']}")
            print(f"   原文件名: {info['old_name']}")
            print(f"   新文件名: {info['new_name']}")
            print(f"   目标位置: {info['target_path']}")
            print("-" * 40)
        
        if len(process_info) > preview_count:
            print(f"... 还有 {len(process_info) - preview_count} 个文件")
        
        # 显示跳过文件信息
        if skipped_files:
            print(f"\n跳过 {len(skipped_files)} 个不符合格式的文件")
            if len(skipped_files) <= 5:
                for skipped in skipped_files:
                    print(f"  - {skipped['name']}: {skipped['reason']}")
            else:
                for skipped in skipped_files[:5]:
                    print(f"  - {skipped['name']}: {skipped['reason']}")
                print(f"  ... 还有 {len(skipped_files) - 5} 个文件被跳过")
        
        # 询问用户确认
        print(f"\n准备处理 {len(process_info)} 个文件，并复制到: {target_dir}")
        confirm = input(f"\n确认要开始处理吗？(y/n): ").lower().strip()
        
        if confirm in ['y', 'yes', '是', '确认']:
            success_count = 0
            error_count = 0
            
            print("\n开始处理文件...")
            print("-" * 80)
            
            for i, info in enumerate(process_info, 1):
                try:
                    # 复制文件到目标目录并重命名
                    shutil.copy2(info['old_path'], info['target_path'])
                    if i % 100 == 0 or i == len(process_info):
                        print(f"进度: {i}/{len(process_info)} - 已处理: {info['new_name']}")
                    success_count += 1
                except Exception as e:
                    print(f"✗ 处理失败: {info['old_name']} - 错误: {str(e)}")
                    error_count += 1
            
            print("-" * 80)
            print(f"\n处理完成！")
            print(f"成功处理: {success_count}/{len(process_info)} 个文件")
            if error_count > 0:
                print(f"处理失败: {error_count} 个文件")
            print(f"文件已汇总到: {target_dir}")
        else:
            print("操作已取消")
    else:
        print("没有找到需要处理的文件")
        if skipped_files:
            print(f"发现 {len(skipped_files)} 个文件，但都不符合命名格式")

def preview_transformation(source_dir):
    """
    预览处理操作，不实际执行文件复制
    
    Args:
        source_dir (str): 要预览的源目录路径
    """
    if not os.path.exists(source_dir):
        print(f"错误：目录 {source_dir} 不存在")
        return
    
    # 支持的文件扩展名
    supported_extensions = ['.mp4', '.json']
    
    # 正则表达式模式
    pattern = r'^.*?\[([A-Za-z0-9_]+)\]\.([^.]+)$'
    
    preview_info = []
    skipped_count = 0
    
    # 递归遍历所有文件
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # 只处理支持的文件类型
            _, ext = os.path.splitext(filename)
            if ext.lower() not in supported_extensions:
                continue
            
            match = re.match(pattern, filename)
            
            if match:
                identifier = match.group(1)
                extension = match.group(2)
                new_filename = f"{identifier}.{extension}"
                
                preview_info.append({
                    'old_name': filename,
                    'new_name': new_filename,
                    'source': root
                })
            else:
                skipped_count += 1
    
    if preview_info:
        print(f"\n预览：将处理 {len(preview_info)} 个文件")
        print("-" * 80)
        
        # 显示前20个示例
        preview_count = min(20, len(preview_info))
        for i, info in enumerate(preview_info[:preview_count], 1):
            rel_source = os.path.relpath(info['source'], source_dir)
            print(f"{i}. {rel_source}/{info['old_name']} -> {info['new_name']}")
        
        if len(preview_info) > preview_count:
            print(f"... 还有 {len(preview_info) - preview_count} 个文件")
        
        if skipped_count > 0:
            print(f"\n跳过 {skipped_count} 个不符合格式的文件")
    else:
        print("没有找到需要处理的文件")
        if skipped_count > 0:
            print(f"发现 {skipped_count} 个文件，但都不符合命名格式")

if __name__ == "__main__":
    # 设置源目录和目标目录
    source_directory = "bilibili_cs2"
    target_directory = "processed_bilibili_cs2"
    
    print("B站视频文件处理和汇总工具")
    print("=" * 80)
    print("功能：")
    print("  1. 递归处理指定目录下所有子文件夹中的视频和JSON文件")
    print("  2. 去除文件名中的中文部分和方括号，只保留方括号内的标识部分")
    print("  3. 将处理后的文件汇总到一个目标文件夹")
    print("=" * 80)
    print(f"源目录: {source_directory}")
    print(f"目标目录: {target_directory}")
    
    # 首先预览处理操作
    print("\n1. 预览处理操作：")
    preview_transformation(source_directory)
    
    # 询问是否继续执行
    print("\n2. 选择操作：")
    print("a) 执行处理并汇总文件")
    print("b) 仅预览，不执行")
    
    choice = input("\n请选择 (a/b): ").lower().strip()
    
    if choice in ['a', '执行']:
        print("\n3. 执行处理操作：")
        transform_and_collect_files(source_directory, target_directory)
    else:
        print("操作已取消，仅预览完成")
