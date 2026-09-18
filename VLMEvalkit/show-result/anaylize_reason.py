import json
import os
import random
import math
import re
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULT_DIR = THIS_DIR / "result_analysis" / "VCR-bench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_model import GPT4o

# ==========================================
# Phase 1: 自动发现两级废话体系 (现象 -> 根因)
# ==========================================
def discover_deep_waste_taxonomy(waste_reasons, gpt_model):
    """
    构建两级分类体系：
    Level 1: Waste Category (e.g., Verbose, Hallucination)
    Level 2: Specific Root Cause (e.g., Repeats Question, Invents Background)
    """
    print(f"[*] Phase 1: 正在构建两级废话根因体系 (基于 {len(waste_reasons)} 条记录)...")

    # 采样更多样本以覆盖更多情况
    sample_size = min(len(waste_reasons), 80)
    samples = random.sample(waste_reasons, sample_size)
    samples_text = "\n".join([f"- {s}" for s in samples])

    prompt = f"""You are a Lead AI Researcher analyzing model failure modes.
I will provide a list of "waste reasons" (why parts of a VQA response were useless).
Your task is to build a **Hierarchical Taxonomy** of these failures to identify the root causes.

--- Examples ---
{samples_text}

--- Task ---
1. Group these errors into 4-6 High-Level Categories (e.g., Repetition, Hallucination, Irrelevant).
2. For EACH category, identify 2-4 Specific Root Causes (Sub-types) based on the examples.

Output a JSON object strictly in this format:
{{
    "Repetition": {{
        "Repeats Question": "The model repeats the user prompt verbatim.",
        "Self-Repetition": "The model repeats its own answer or phrases multiple times."
    }},
    "Hallucination": {{
        "Object Fabrication": "Inventing objects not present in the image.",
        "OCR Hallucination": "Reading text that doesn't exist."
    }},
    "Verbose Style": {{
        "Unnecessary Background": "Describing background details irrelevant to the question.",
        "Over-Explanation": "Defining common terms or explaining obvious concepts."
    }}
}}
Output JSON only.
"""
    messages = [{"role": "user", "content": prompt}]
    response = gpt_model.send_stable_request(messages, temperature=0.3)

    try:
        taxonomy = json.loads(response.replace("```json", "").replace("```", "").strip())
        print("[*] 根因体系构建成功：")
        print(json.dumps(taxonomy, indent=2, ensure_ascii=False))
        return taxonomy
    except Exception as e:
        print(f"[!] 聚类失败，使用默认体系: {e}")
        return {
            "Repetition": {"General Repetition": "Repeats content."},
            "Hallucination": {"General Hallucination": "Invents details."},
            "Verbose": {"General Verbose": "Too wordy."}
        }

# ==========================================
# Phase 2: 深度批量标注 (Label + Root Cause)
# ==========================================
def batch_tag_deep_waste(data_items, taxonomy, gpt_model):
    """
    批量标注函数：同时返回 Category 和 Root Cause
    """
    # 将 Taxonomy 扁平化为说明文本，方便 GPT 理解
    taxonomy_desc = json.dumps(taxonomy, indent=2)

    batch_text = ""
    for idx, reason in data_items:
        batch_text += f"ID {idx}: {reason}\n"

    prompt = f"""You are a data labeler.
Map each "waste reason" to the most specific **Category** and **Root Cause** from the Taxonomy below.

--- Taxonomy ---
{taxonomy_desc}

--- Input List ---
{batch_text}

--- Task ---
Return a JSON object mapping ID to a dictionary containing "category" and "root_cause".
If the reason is empty/NA, use "None".

Example Output:
{{
  "12": {{"category": "Repetition", "root_cause": "Repeats Question"}},
  "45": {{"category": "Hallucination", "root_cause": "Object Fabrication"}}
}}
Output JSON only.
"""
    messages = [{"role": "user", "content": prompt}]
    response = gpt_model.send_stable_request(messages, temperature=0.0)

    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"[!] Batch parsing error: {e}")
        return {}

# ==========================================
# Phase 3: 关键词提取辅助函数
# ==========================================
def extract_keywords(text_list, top_n=5):
    """提取高频关键词，帮助定位问题"""
    if not text_list: return []
    text = " ".join(str(t) for t in text_list).lower()
    # 简单的正则分词，保留3个字母以上的词
    words = re.findall(r'\b[a-z]{3,}\b', text)
    stop_words = {'the', 'and', 'that', 'for', 'with', 'this', 'not', 'but', 'are', 'was', 'model', 'response', 'answer', 'question', 'image', 'provides', 'includes', 'mentioning', 'about', 'because', 'which', 'from'}
    filtered = [w for w in words if w not in stop_words]
    return Counter(filtered).most_common(top_n)

# ==========================================
# 主流程
# ==========================================
def tag_waste_reasons_deep(input_file, output_file, model_name="gpt-5.1-2025-11-13"):
    gpt = GPT4o(deployment_name=model_name)

    if not os.path.exists(input_file):
        print("文件不存在")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    if "detailed_reports" in full_data:
        reports = full_data["detailed_reports"]
        existing_taxonomy = full_data.get("taxonomy_used", {})
    else:
        reports = full_data
        existing_taxonomy = {}

    # 1. 提取待处理数据
    to_process = []
    valid_reasons = []

    for idx, item in reports.items():
        analysis = item.get('error_analysis', {})
        reason = analysis.get('waste_reason', 'N/A')
        ratio = analysis.get('useful_ratio', 1.0)

        if ratio < 1.0 and reason and reason != "N/A":
            to_process.append((idx, reason))
            valid_reasons.append(reason)
        else:
            item['error_analysis']['waste_category'] = "None"
            item['error_analysis']['waste_root_cause'] = "None"

    print(f"[*] 共 {len(reports)} 条数据，其中 {len(to_process)} 条包含废话需要深挖。")

    # 2. 构建两级体系
    deep_taxonomy = discover_deep_waste_taxonomy(valid_reasons, gpt)

    # 3. 批量标注
    batch_size = 20
    total_batches = math.ceil(len(to_process) / batch_size)

    print(f"[*] 开始批量标注 (共 {total_batches} 个批次)...")

    for i in tqdm(range(0, len(to_process), batch_size), desc="Deep Tagging"):
        batch = to_process[i : i + batch_size]
        tags_map = batch_tag_deep_waste(batch, deep_taxonomy, gpt)

        for idx, _ in batch:
            tag_info = tags_map.get(str(idx), {"category": "Uncategorized", "root_cause": "Uncategorized"})
            
            # 写入两个字段：大类和根因
            reports[idx]['error_analysis']['waste_category'] = tag_info.get("category", "Uncategorized")
            reports[idx]['error_analysis']['waste_root_cause'] = tag_info.get("root_cause", "Uncategorized")

    # 4. 生成简单的终端报告
    print("\n" + "="*50)
    print("🚀 ROOT CAUSE ANALYSIS REPORT")
    print("="*50)

    # 统计数据
    stats = {} # {Category: {RootCause: [reasons...]}}
    
    for idx, _ in to_process:
        item = reports[idx]['error_analysis']
        cat = item.get('waste_category', 'Uncategorized')
        root = item.get('waste_root_cause', 'Uncategorized')
        reason = item.get('waste_reason', '')
        
        if cat not in stats: stats[cat] = {}
        if root not in stats[cat]: stats[cat][root] = []
        stats[cat][root].append(reason)

    # 打印层级报告
    for cat, sub_stats in sorted(stats.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True):
        total_cat = sum(len(v) for v in sub_stats.values())
        print(f"\n📦 Category: {cat} (Total: {total_cat})")
        
        for root, reasons_list in sorted(sub_stats.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(reasons_list)
            pct = (count / total_cat) * 100
            
            # 提取关键词
            keywords = extract_keywords(reasons_list)
            kw_str = ", ".join([k for k, v in keywords])
            
            print(f"   ├── 🔍 Root Cause: {root:<30} Count: {count:>3} ({pct:>4.1f}%)")
            print(f"   │    Keywords: [{kw_str}]")

    # 5. 保存
    final_output = {
        "taxonomy_used": existing_taxonomy,
        "waste_taxonomy_deep": deep_taxonomy,
        "detailed_reports": reports
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] 深度分析结果已保存至: {output_file}")

if __name__ == "__main__":
    input_path = str(RESULT_DIR / "Qwen3VL-8B_Hierarchical_Analysis_Report.json")
    output_path = str(RESULT_DIR / "Qwen3VL-8B_Deep_Root_Cause_Report.json")
    
    tag_waste_reasons_deep(input_path, output_path)
