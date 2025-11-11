import json
import os

def convert_json_to_jsonl(json_file, jsonl_file):
    """
    将JSON数组文件转换为JSONL格式（每行一个JSON对象）
    """
    print(f"正在转换 {json_file} 到 {jsonl_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成！共处理 {len(data)} 条记录")

if __name__ == "__main__":
    input_file = "fp_processed.json"
    output_file = "fp_processed.jsonl"
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        exit(1)
    
    convert_json_to_jsonl(input_file, output_file)
    print(f"输出文件：{output_file}")

