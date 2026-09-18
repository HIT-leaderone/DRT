import json
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULT_DIR = THIS_DIR / "result_analysis" / "VCR-bench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_model import GPT4o

def discover_subcategories(gpt, error_type, analysis_samples):
    """
    Phase 1: 给 GPT 一堆分析文本，让它总结出子类
    """
    print(f"[*] 正在为 [{error_type}] 定义子类体系 (基于 {len(analysis_samples)} 个样本)...")
    
    samples_text = "\n".join([f"- {txt}" for txt in analysis_samples])
    
    prompt = f"""You are a taxonomy expert in Multimodal LLM evaluation.
I will provide a list of error analyses belonging to the major category: "{error_type}".
Please analyze these examples and summarize 3 to 6 distinct, mutually exclusive "Sub-Error Types".

--- Examples ---
{samples_text}
--- End Examples ---

Task:
1. Identify common patterns in these errors.
2. Define 3-6 sub-categories.
3. Return a JSON object where keys are sub-category names and values are brief descriptions.

Output JSON format example:
{{
    "Object Hallucination": "The model detects objects that are not present.",
    "Action Hallucination": "The model misinterprets the action being performed.",
    "OCR Error": "The model fails to read text in the image correctly."
}}
Output JSON only.
"""
    response = gpt.send_stable_request([{"role": "user", "content": prompt}], temperature=0.2)
    
    try:
        # 清洗 markdown
        clean_json = response.replace("```json", "").replace("```", "").strip()
        taxonomy = json.loads(clean_json)
        return taxonomy
    except:
        print(f"[!] 解析子类定义失败，使用默认分类")
        return {"General Error": "Unable to categorize further."}

def annotate_subtype(gpt, error_type, analysis_text, taxonomy):
    """
    Phase 2: 根据定义好的 taxonomy，给单个样本分类
    """
    # 构造分类选项字符串
    options_str = "\n".join([f"- {k}: {v}" for k, v in taxonomy.items()])
    
    prompt = f"""You are an error annotator.
Major Error Type: "{error_type}"
Specific Analysis: "{analysis_text}"

Based on the following sub-categories, classify this error:
{options_str}

Output JSON only: {{"sub_error_type": "Your Choice"}}
"""
    response = gpt.send_stable_request([{"role": "user", "content": prompt}], temperature=0.0)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        res = json.loads(clean_json)
        return res.get("sub_error_type", "Other")
    except:
        return "Uncategorized"

# ==========================================
# 3. 主流程
# ==========================================
def refine_error_analysis(input_file, output_file, model_name):
    gpt = GPT4o(deployment_name=model_name)
    
    # 1. 读取文件
    if not os.path.exists(input_file):
        print(f"找不到文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f) # 假设是 Dict[index, item]
    
    print(f"[*] 加载数据 {len(data)} 条")

    # 2. 数据分组 (按 Error Type)
    grouped_data = {} # { "Hallucination": [item_id1, item_id2...], ... }
    
    for idx, item in data.items():
        # 提取 error_type
        e_type = item.get('error_analysis', {}).get('error_type', 'Unknown')
        
        # 跳过 Correct 和 Unknown
        if e_type in ['Correct', 'Unknown']:
            continue
            
        if e_type not in grouped_data:
            grouped_data[e_type] = []
        grouped_data[e_type].append(idx)

    # 3. 逐个大类处理
    final_taxonomy = {} # 存储所有定义好的子类体系
    
    for error_type, indices in grouped_data.items():
        print(f"\n{'='*10} 处理大类: {error_type} (共 {len(indices)} 条) {'='*10}")
        
        # --- Phase 1: 发现子类 ---
        # 随机抽取最多 20 条 analysis 文本用于定义
        sample_indices = random.sample(indices, min(len(indices), 300))
        sample_analyses = [data[i]['error_analysis']['analysis'] for i in sample_indices]
        
        taxonomy = discover_subcategories(gpt, error_type, sample_analyses)
        final_taxonomy[error_type] = taxonomy
        
        print(f"[*] 定义子类成功:")
        for k, v in taxonomy.items():
            print(f"   - {k}")

        # --- Phase 2: 标注子类 ---
        print(f"[*] 开始标注 {len(indices)} 条数据...")
        for idx in tqdm(indices, desc=f"Annotating {error_type}"):
            analysis_text = data[idx]['error_analysis']['analysis']
            
            sub_type = annotate_subtype(gpt, error_type, analysis_text, taxonomy)
            
            # 将结果写回 data 对象
            data[idx]['error_analysis']['sub_error_type'] = sub_type

    # 4. 保存结果
    # 将 taxonomy 也保存进去，方便后续查看定义
    output_data = {
        "taxonomy_definitions": final_taxonomy,
        "detailed_reports": data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n[Done] 细分报告已保存至: {output_file}")

# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    # 你的输入文件路径
    input_path = str(RESULT_DIR / "Qwen3VL-8B_gpt-5.1-2025-11-13_Error_Analysis_Report.json")
    
    # 输出文件路径
    output_path = str(RESULT_DIR / "Qwen3VL-8B_Refined_Subtypes_Report.json")
    
    # 请替换为你的模型名称
    model_name = "gpt-5.1-2025-11-13" 
    
    refine_error_analysis(input_path, output_path, model_name)
