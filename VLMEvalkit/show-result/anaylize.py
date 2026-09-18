import json
import os
import random
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import openai
from openai import OpenAI

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULT_DIR = THIS_DIR / "result_analysis" / "VCR-bench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_model import GPT4o  # 假设你的 GPT4o 类在这个文件中

# ==========================================
# 辅助函数
# ==========================================

def truncate_text(text, limit=300):
    """辅助函数：截断文本"""
    if text is None:
        return "N/A"
    s = str(text).replace('\n', ' ')
    if len(s) > limit:
        return s[:limit] + "... [已截断]"
    return s

# ==========================================
# Phase 1: 自动发现二级分类体系
# ==========================================

def discover_taxonomy(data, gpt_model, sample_size=4000):
    """
    从数据中采样，构建二级分类体系
    一级分类固定为：
    1. Visual Perception Failure (视觉信息获取失败)
    2. Reasoning & Hallucination Failure (视觉正确但推理/幻觉失败)
    """
    print(f"[*] Phase 1: 正在构建二级分类体系 (采样数: {sample_size})...")
    
    # 1. 筛选出分数较低的样本作为“错题集”进行分析
    error_samples = [item for item in data if float(item.get('answer_scoring', 0)) < 1.0]
    
    # 如果错题不够，就用全部
    if len(error_samples) < sample_size:
        selected_samples = error_samples
    else:
        selected_samples = random.sample(error_samples, sample_size)
    
    print(f"[*] 已采样 {len(selected_samples)} 条错题样本")
    
    # 构造样本字符串
    examples_text = ""
    for i, item in enumerate(selected_samples):
        examples_text += f"Case {i+1}:\n"
        examples_text += f"[Q]: {item.get('question')}\n"
        examples_text += f"[GT]: {item.get('answer')}\n"
        examples_text += f"[Pred]: {item.get('prediction')}\n"
        examples_text += f"[Reasoning]: {item.get('reasoning')}\n\n"

    # 构造 Prompt
    prompt = f"""You are a taxonomy expert for VQA (Visual Question Answering) evaluation.
I have a dataset of VQA errors. I have already defined the Level-1 Categories.
Your task is to analyze the provided cases and define specific **Level-2 Sub-categories** for each Level-1 category.

--- Level-1 Categories (Fixed) ---
Type A: **Visual Perception Failure**
Definition: The model failed to recognize objects, text (OCR), attributes (color/count), or spatial relations correctly. The input visual information was not captured accurately.

Type B: **Reasoning or Hallucination Failure**
Definition: The model likely saw the image elements correctly (or the visual part wasn't the hard part), but failed in logical reasoning, hallucinated non-existent details/events, failed to follow instructions, or used wrong external knowledge.

--- Task ---
Analyze the following {len(selected_samples)} error cases.
Summarize 3-5 distinct Level-2 sub-categories for Type A.
Summarize 3-5 distinct Level-2 sub-categories for Type B.

Output a JSON object strictly in this format:
{{
    "Visual Perception Failure": {{
        "Subtype Name 1": "Description...",
        "Subtype Name 2": "Description..."
    }},
    "Reasoning or Hallucination Failure": {{
        "Subtype Name 1": "Description...",
        "Subtype Name 2": "Description..."
    }}
}}

--- Cases ---
{examples_text[:224000]} 
(Truncated if too long)
"""
    # 注意：这里截断了 examples_text 防止超过 token 限制，根据你的模型 context window 调整
    
    messages = [{"role": "user", "content": prompt}]
    response = gpt_model.send_stable_request(messages, temperature=0.4) # 稍微高一点温度以获得更好的总结
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        taxonomy = json.loads(clean_json)
        print("[*] 分类体系构建成功！")
        print(json.dumps(taxonomy, indent=2, ensure_ascii=False))
        return taxonomy
    except Exception as e:
        print(f"[!] 构建分类体系失败: {e}")
        # 返回一个默认的保底分类
        return {
            "Visual Perception Failure": {
                "Object Detection Error": "Failed to find the object.",
                "Attribute Error": "Wrong color, shape, or count.",
                "OCR Error": "Failed to read text."
            },
            "Reasoning or Hallucination Failure": {
                "Logic Hallucination": "Made up facts not in image.",
                "Causal Error": "Wrong cause-effect reasoning.",
                "Instruction Error": "Failed to follow output format."
            }
        }

# ==========================================
# Phase 2: 具体标注
# ==========================================

def generate_annotation_prompt(item, taxonomy):
    """
    构造标注 Prompt，注入生成的 Taxonomy
    """
    taxonomy_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)
    
    prompt = f"""You are an expert in evaluating Multimodal LLMs.
Please analyze the following VQA case based on the provided Taxonomy.

--- Taxonomy Definitions ---
{taxonomy_str}

--- Case Data ---
[Question]: {item.get('question', 'N/A')}
[Ground Truth Answer]: {item.get('answer', 'N/A')}
[Model Prediction]: {item.get('prediction', 'N/A')}
[Original Scoring Reasoning]: {item.get('reasoning', 'N/A')}

--- Task ---
1. Determine if the result is "Correct" or an "Error".
2. If it is an error, classify it into **Level 1** and **Level 2** categories based on the Taxonomy above.
3. **Analyze Information Density**: Determine how much of the [Model Prediction] is actually helpful.

Output a JSON object with the following keys:
1. "is_correct": Boolean (true/false).
2. "level_1_category": Choose strictly from ["Visual Perception Failure", "Reasoning or Hallucination Failure", "Correct"].
3. "level_2_category": Choose the most fitting key from the taxonomy (or "N/A" if correct).
4. "analysis": A concise 1-sentence explanation.
5. "useful_ratio": Float 0.0-1.0.
6. "waste_reason": Brief explanation of useless content.

Output JSON only:
"""
    return prompt

def analyze_errors_with_taxonomy(input_file, output_file, model_name="gpt-4o"):
    # 1. 初始化模型
    gpt = GPT4o(deployment_name=model_name)

    # 2. 读取数据
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[*] 加载数据成功，共 {len(data)} 条")

    # 3. 执行 Phase 1: 构建分类体系
    # 检查是否已经有生成的 taxonomy 文件，避免重复生成
    taxonomy_file = output_file.replace(".json", "_taxonomy_config.json")
    if os.path.exists(taxonomy_file):
        print(f"[*] 发现已有分类配置文件: {taxonomy_file}")
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
    else:
        taxonomy = discover_taxonomy(data, gpt)
        # 保存分类体系
        with open(taxonomy_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    # 4. 准备结果容器 (支持断点续传)
    analysis_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                # 兼容处理：如果保存的是列表，转为字典；如果是字典，直接用
                if isinstance(saved_data, dict) and "detailed_reports" in saved_data:
                    analysis_results = saved_data["detailed_reports"]
                elif isinstance(saved_data, dict):
                    analysis_results = saved_data
            print(f"[*] 已加载 {len(analysis_results)} 条已有分析")
        except:
            print("[!] 结果文件格式不兼容或损坏，重新开始")

    # 5. 执行 Phase 2: 遍历标注
    for item in tqdm(data, desc="Annotating"):
        # 获取索引
        idx = str(item.get('index', item.get('id', str(data.index(item)))))

        if idx in analysis_results:
            continue

        prompt = generate_annotation_prompt(item, taxonomy)
        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = gpt.send_stable_request(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                analysis_json = json.loads(clean_text)
            except:
                analysis_json = {"raw_analysis": clean_text, "level_1_category": "Parse Error"}

            # 存入结果
            analysis_results[idx] = {
                "original_data": item,
                "error_analysis": analysis_json
            }

            # 实时打印预览
            if len(analysis_results) % 5 == 0:
                l1 = analysis_json.get('level_1_category', 'N/A')
                l2 = analysis_json.get('level_2_category', 'N/A')
                print(f" [Idx {idx}] L1: {l1} | L2: {l2}")

            # 定期保存
            if len(analysis_results) % 10 == 0:
                final_output = {
                    "taxonomy_used": taxonomy,
                    "detailed_reports": analysis_results
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\n[!] Error processing {idx}: {e}")

    # 6. 最终保存
    final_output = {
        "taxonomy_used": taxonomy,
        "detailed_reports": analysis_results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[*] 完成！分类体系保存在: {taxonomy_file}")
    print(f"[*] 详细标注保存在: {output_file}")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VCR-Bench error categories with an OpenAI-compatible judge.")
    parser.add_argument("--input", default=os.environ.get("VCR_ANALYSIS_INPUT", "./Qwen3-VL-8B-Instruct_VCR-Bench_answer_score.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=os.environ.get("VCR_ANALYSIS_MODEL", "gpt-4o"))
    args = parser.parse_args()

    input_filename = args.input
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = args.output or str(RESULT_DIR / "Qwen3VL-8B_Hierarchical_Analysis_Report.json")

    # 请确保 model_name 正确
    analyze_errors_with_taxonomy(input_filename, output_filename, model_name=args.model)
