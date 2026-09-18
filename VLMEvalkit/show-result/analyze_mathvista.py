import json
import os
import random
import time
import base64
import mimetypes
import argparse
import pandas as pd
from tqdm import tqdm
import openai
from openai import OpenAI
from vlmeval.dataset import MathVista
from gpt_model import GPT4o  # Assuming gpt_model.py is in the same directory

# ==========================================
# 辅助函数
# ==========================================

def encode_image(image_path):
    """将本地图片编码为 base64 格式"""
    if not os.path.exists(image_path):
        return None
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg'
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def truncate_text(text, limit=300):
    """辅助函数：截断文本"""
    if text is None:
        return "N/A"
    s = str(text).replace('\n', ' ')
    if len(s) > limit:
        return s[:limit] + "... [已截断]"
    return s

def is_correct(item, eps=1e-4):
    """
    判断样本是否正确
    逻辑：尝试转 float 比较差值 < eps，否则比较字符串相等
    """
    res = item.get('res')
    answer = item.get('answer')
    
    # 处理 None 的情况
    if res is None or answer is None:
        # 如果 res 或 answer 是 None，通常视为错误，或者根据数据情况调整
        # 这里假设没有结果或者没有答案都算错
        return False

    # 尝试作为数字比较
    try:
        res_float = float(str(res).strip())
        ans_float = float(str(answer).strip())
        return abs(res_float - ans_float) < eps
    except (ValueError, TypeError):
        # 无法转数字，进行字符串比较
        # 归一化：转字符串，去除首尾空白，转小写（可选，视具体要求而定，这里先做 strip）
        str_res = str(res).strip()
        str_ans = str(answer).strip()
        return str_res == str_ans

# ==========================================
# Phase 1: 自动发现二级分类体系
# ==========================================

def discover_taxonomy(data, dataset, gpt_model, sample_size=50):
    """
    从数据中采样，构建废话/冗余分类体系 (Waste Taxonomy)
    """
    print(f"[*] Phase 1: 正在构建冗余分类体系 (采样数: {sample_size})...")
    
    # 策略：不再仅筛选错题，而是优先采样预测结果较长的样本，或者随机采样
    # 这里我们简单按预测长度排序，取较长的，因为长的更容易有废话
    try:
        sorted_data = sorted(data, key=lambda x: len(str(x.get('prediction', ''))), reverse=True)
        # 取前 3*sample_size 个中随机采样，增加多样性
        candidate_pool = sorted_data[:min(len(data), sample_size * 3)]
        selected_samples = random.sample(candidate_pool, min(len(candidate_pool), sample_size))
    except Exception as e:
        print(f"[Warn] Sampling failed: {e}, utilizing random sampling.")
        selected_samples = random.sample(data, min(len(data), sample_size))
    
    print(f"[*] 已采样 {len(selected_samples)} 条样本用于构建体系")
    
    # 构造样本字符串
    examples_text = ""
    for i, item in enumerate(selected_samples):
        examples_text += f"Case {i+1}:\n"
        examples_text += f"[Q]: {item.get('question')}\n"
        examples_text += f"[GT]: {item.get('answer')}\n"
        examples_text += f"[Pred]: {item.get('prediction')}\n\n"

    # 构造 Prompt
    prompt = f"""You are a Lead AI Researcher analyzing model verbosity and redundancy (Waste Analysis).
I have a dataset of VQA model outputs.
Your task is to analyze the provided cases and build a **Hierarchical Taxonomy** of "Waste Categories" (why parts of a response were useless/redundant).

--- Task ---
1. Group the redundancy/waste patterns into 4-6 High-Level Categories (e.g., Repetition, Hallucination, Irrelevant, Verbose Style).
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

--- Cases ---
{examples_text[:20000]} 
(Truncated if too long)
"""
    
    messages = [{"role": "user", "content": prompt}]
    response = gpt_model.send_stable_request(messages, temperature=0.4) 
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        taxonomy = json.loads(clean_json)
        print("[*] 冗余分类体系构建成功！")
        print(json.dumps(taxonomy, indent=2, ensure_ascii=False))
        return taxonomy
    except Exception as e:
        print(f"[!] 构建分类体系失败: {e}")
        return {
            "Repetition": {"General Repetition": "Repeats content."},
            "Hallucination": {"General Hallucination": "Invents details."},
            "Verbose": {"General Verbose": "Too wordy."},
            "Irrelevant": {"General Irrelevant": "Off-topic content."}
        }


# ==========================================
# Phase 2: 具体标注
# ==========================================

def generate_annotation_msgs(item, taxonomy, dataset):
    """
    构造标注 Prompt，注入生成的 Taxonomy，并包含图片
    """
    taxonomy_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)
    
    # 尝试获取图片
    img_path = None
    try:
        # 使用 item['index'] 在 dataset 中查找
        # 注意：dataset.data 是原始数据，item 是结果 excel 的一行
        idx = item['index']
        # 确保 idx 类型匹配
        # MathVista 的 index 可能是 int 或 str
        if idx in dataset.data['index'].values:
            line = dataset.data[dataset.data['index'] == idx].iloc[0]
        else:
            # 尝试转 int (如果是 str) 或 str (如果是 int)
            try:
                line = dataset.data[dataset.data['index'] == int(idx)].iloc[0]
            except:
                line = dataset.data[dataset.data['index'] == str(idx)].iloc[0]
            
        img_path_raw = dataset.dump_image(line)
        # img_path 可能是列表或单个字符串
        if isinstance(img_path_raw, list):
            img_path = img_path_raw[0] # MathVista 通常是一张图
        else:
            img_path = img_path_raw
            
    except Exception as e:
        # print(f"[Warn] Cannot find image for index {item.get('index')}: {e}")
        img_path = None

    prompt_text = f"""You are an expert in evaluating Multimodal LLMs.
Please analyze the following VQA case for **Redundancy and Waste** based on the provided Taxonomy.

--- Taxonomy Definitions ---
{taxonomy_str}

--- Case Data ---
[Question]: {item.get('question', 'N/A')}
[Ground Truth Answer]: {item.get('answer', 'N/A')}
[Model Prediction]: {item.get('prediction', 'N/A')}

--- Task ---
1. **Analyze Information Density**: Identify the *useful information* in the [Model Prediction] that contributes to answering the question. Everything else is "Waste".
2. **Categorize Waste**: If there is waste, classify it into the most fitting **Category(s)** and **Root Cause(s)** from the Taxonomy. If there are multiple distinct types of waste, LIST ALL OF THEM. If there is NO waste (perfectly concise), use ["None"].
3. **Compress**: Generate a compressed version of the prediction containing ONLY the information strictly necessary to derive the answer. Do NOT use natural language if possible. Use formal language, mathematical notation, or arrow-based derivations (e.g., A -> B -> C) to represent the reasoning process concisely.

Output a JSON object with the following keys:
1. "waste_categories": List of strings. Choose strictly from the Taxonomy keys. Example: ["Repetition", "Verbose Style"]. If none, ["None"].
2. "waste_root_causes": List of strings. Choose strictly from the Taxonomy sub-keys. Example: ["Repeats Question", "Over-Explanation"]. If none, ["None"].
3. "waste_reason": A concise 1-sentence explanation of why this is considered waste.
4. "compressed_response": String. The compressed version of the prediction.

Output JSON only:
"""
    
    content = []
    
    # 如果有图片，添加图片到消息
    if img_path:
        base64_image = encode_image(img_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
    
    content.append({"type": "text", "text": prompt_text})
    
    # 最终 message
    final_messages = [{"role": "user", "content": content}]
    
    return final_messages

def analyze_errors_with_taxonomy(input_file, output_file, model_name="gpt-4o"):
    # 1. 初始化模型
    gpt = GPT4o(deployment_name=model_name)
    
    # 2. 初始化 MathVista 数据集以获取图片
    try:
        # 尝试指定 dataset 名称，vlmeval 会自动处理
        # 假设 MathVista_MINI 是支持的 dataset name
        dataset = MathVista(dataset='MathVista_MINI')
        print("[*] MathVista dataset loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load MathVista dataset: {e}")
        print("[!] Will proceed without images.")
        dataset = None

    # 3. 读取 Excel 数据
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        # Fallback for json
        with open(input_file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        df = pd.DataFrame(data_json)
        
    data = df.to_dict('records')
    print(f"[*] 加载数据成功，共 {len(data)} 条")

    # 4. 执行 Phase 1: 构建分类体系
    taxonomy_file = output_file.replace(".json", "_taxonomy_config.json")
    if os.path.exists(taxonomy_file):
        print(f"[*] 发现已有分类配置文件: {taxonomy_file}")
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
    else:
        # Phase 1 只需要文本
        taxonomy = discover_taxonomy(data, dataset, gpt)
        # 保存分类体系
        with open(taxonomy_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    # 5. 准备结果容器 (支持断点续传)
    analysis_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if isinstance(saved_data, dict) and "detailed_reports" in saved_data:
                    analysis_results = saved_data["detailed_reports"]
                elif isinstance(saved_data, dict):
                    analysis_results = saved_data
            print(f"[*] 已加载 {len(analysis_results)} 条已有分析")
        except:
            print("[!] 结果文件格式不兼容或损坏，重新开始")

    # 6. 执行 Phase 2: 遍历标注
    for item in tqdm(data, desc="Annotating"):
        # 获取索引
        idx = str(item.get('index', item.get('id', str(data.index(item)))))

        if idx in analysis_results:
            continue
        
        if dataset:
            messages = generate_annotation_msgs(item, taxonomy, dataset)
        else:
            # Fallback if dataset failed to load (no images)
            messages = [{"role": "user", "content": f"Analyze this (no image available):\n{item}"}] # Simplified

        try:
            # 这里需要确认 send_stable_request 是否支持多模态输入
            # 如果 gpt_model.GPT4o 的实现不支持本地路径，可能需要转 base64
            # 我们假设它支持 openai client 的标准输入
            response_text = gpt.send_stable_request(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                analysis_json = json.loads(clean_text)
                
                # 计算 useful_ratio
                prediction = str(item.get('prediction', ''))
                compressed = analysis_json.get('compressed_response', '')
                
                if len(prediction) > 0:
                    ratio = len(compressed) / len(prediction)
                else:
                    ratio = 0.0
                
                analysis_json['useful_ratio'] = round(ratio, 4)
                
                # 兼容性处理：如果返回的是旧格式（字符串），强制转列表
                if "waste_category" in analysis_json and "waste_categories" not in analysis_json:
                    val = analysis_json.pop("waste_category")
                    analysis_json["waste_categories"] = [val] if val != "None" else ["None"]
                
                if "waste_root_cause" in analysis_json and "waste_root_causes" not in analysis_json:
                    val = analysis_json.pop("waste_root_cause")
                    analysis_json["waste_root_causes"] = [val] if val != "None" else ["None"]
                
            except:
                analysis_json = {"raw_analysis": clean_text, "waste_categories": ["Parse Error"], "useful_ratio": 0.0}


            # 存入结果
            # 为了序列化，将 item 中的 NaN 处理掉
            cleaned_item = {k: (v if pd.notna(v) else None) for k, v in item.items()}
            
            # 判断是否正确
            is_item_correct = is_correct(item)

            analysis_results[idx] = {
                "original_data": cleaned_item,
                "error_analysis": analysis_json,
                "is_correct": is_item_correct
            }

            # 实时打印预览
            if len(analysis_results) % 5 == 0:
                cats = analysis_json.get('waste_categories', ['N/A'])
                cat_str = ",".join(cats)
                ratio = analysis_json.get('useful_ratio', 0.0)
                print(f" [Idx {idx}] Cat: {cat_str} | Ratio: {ratio:.2f}")
                
                # Print sample compressed response
                comp_resp = analysis_json.get('compressed_response', 'N/A')
                print(f"   [Compressed]: {truncate_text(comp_resp, 100)}")


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

    # 7. 最终保存
    final_output = {
        "taxonomy_used": taxonomy,
        "detailed_reports": analysis_results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[*] 完成！分类体系保存在: {taxonomy_file}")
    print(f"[*] 详细标注保存在: {output_file}")
    
    # 8. 生成统计报告
    print("\n" + "="*50)
    print("🚀 REDUNDANCY ANALYSIS REPORT (Correct vs Incorrect vs Overall)")
    print("="*50)

    stats = {
        "Correct": {"count": 0, "total_ratio": 0.0, "waste_cats": {}},
        "Incorrect": {"count": 0, "total_ratio": 0.0, "waste_cats": {}},
        "Overall": {"count": 0, "total_ratio": 0.0, "waste_cats": {}}
    }

    for idx, res in analysis_results.items():
        is_corr = res.get("is_correct", False)
        group = "Correct" if is_corr else "Incorrect"
        
        # Individual groups
        stats[group]["count"] += 1
        ratio = res["error_analysis"].get("useful_ratio", 0.0)
        stats[group]["total_ratio"] += ratio
        
        cats = res["error_analysis"].get("waste_categories", ["None"])
        # 如果是字符串，转为列表
        if isinstance(cats, str): cats = [cats]
        
        for cat in cats:
            if cat not in stats[group]["waste_cats"]:
                stats[group]["waste_cats"][cat] = 0
            stats[group]["waste_cats"][cat] += 1
        
        # Overall
        stats["Overall"]["count"] += 1
        stats["Overall"]["total_ratio"] += ratio
        
        for cat in cats:
            if cat not in stats["Overall"]["waste_cats"]:
                stats["Overall"]["waste_cats"][cat] = 0
            stats["Overall"]["waste_cats"][cat] += 1

    for group in ["Correct", "Incorrect", "Overall"]:
        count = stats[group]["count"]
        if count == 0:
            print(f"\n📦 Group: {group} (Total: 0)")
            continue
            
        avg_ratio = stats[group]["total_ratio"] / count
        print(f"\n📦 Group: {group} (Total: {count})")
        print(f"   📊 Average Useful Ratio: {avg_ratio:.4f}")
        print(f"   🗑️  Waste Category Distribution:")
        
        sorted_cats = sorted(stats[group]["waste_cats"].items(), key=lambda x: x[1], reverse=True)
        for cat, c in sorted_cats:
            pct = (c / count) * 100
            print(f"      - {cat:<30}: {c:>3} ({pct:>5.1f}%)")

    # 更新最终输出包含统计
    final_output["statistics"] = stats
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MathVista errors with an OpenAI-compatible judge.")
    parser.add_argument("--input", default=os.environ.get("MATHVISTA_ANALYSIS_INPUT", "./Qwen3-VL-8B-Instruct_MathVista_MINI_score.xlsx"))
    parser.add_argument("--output", default=os.environ.get("MATHVISTA_ANALYSIS_OUTPUT", "./Qwen3VL-8B_MathVista_Analysis_Report.json"))
    parser.add_argument("--model", default=os.environ.get("MATHVISTA_ANALYSIS_MODEL", "gpt-4o"))
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    analyze_errors_with_taxonomy(input_filename, output_filename, model_name=args.model)
