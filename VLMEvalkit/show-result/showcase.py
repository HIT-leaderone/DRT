import argparse
import json
import os

def inspect_json_file(file_path):
    print(f"[*] 正在读取文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[!] 错误: 找不到文件 {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] 读取 JSON 失败: {e}")
        return

    # --- 1. 检查结构 ---
    print("\n=== 文件结构分析 ===")
    print(f"数据类型: {type(data)}")
    
    sample_item = None
    
    if isinstance(data, list):
        print(f"数据总条数: {len(data)}")
        if len(data) > 0:
            sample_item = data[500]
            print(f"单条数据字段 (Keys): {list(sample_item.keys())}")
            for key in sample_item.keys():
                print(f"字段 '{key}' 类型: {type(sample_item[key])}")
    elif isinstance(data, dict):
        print(f"顶层字段 (Keys): {list(data.keys())}")
        # 尝试寻找包含数据的列表键
        for key in data:
            if isinstance(data[key], list) and len(data[key]) > 0:
                print(f"发现主要数据列表在 key: '{key}', 长度: {len(data[key])}")
                sample_item = data[key][0]
                break
        if sample_item is None:
            sample_item = data # 如果没有列表，就展示整个字典
    
    # --- 2. Showcase (带截断) ---
    if sample_item:
        print("\n=== Showcase (第一条数据预览) ===")
        print("{")
        for key, value in sample_item.items():
            val_str = str(value)
            # 截断逻辑：如果字符串长度超过 150 字符，则截断
            limit = 150
            if len(val_str) > limit:
                preview = val_str[:limit].replace('\n', ' ') + f" ... [已截断, 总长 {len(val_str)}]"
            else:
                preview = val_str.replace('\n', ' ')
            
            print(f"  '{key}': {preview}")
        print("}")
    else:
        print("[!] 文件为空或结构无法解析")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the structure and a sample row of a JSON result file.")
    parser.add_argument("filename", help="Path to the JSON file to inspect.")
    args = parser.parse_args()
    inspect_json_file(args.filename)
