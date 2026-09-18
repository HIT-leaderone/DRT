import json
import os
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from gpt_model import GPT4o 

# ==========================================
# 核心逻辑：思维密度分析
# ==========================================

def generate_density_analysis_prompt(item):
    conversations = item.get('conversations', [])
    
    question_content = ""
    original_response = ""
    
    # 1. 提取 User 问题
    for conv in conversations:
        if conv['from'] == 'user':
            question_content = conv['value'].replace("<image>", "").strip()
            break
            
    # 2. 提取 Assistant 完整回复 (包含 think 和 answer)
    for conv in conversations:
        if conv['from'] == 'assistant':
            original_response = conv['value']
            break

    if not original_response:
        return None
    
    # 3. 构建 Prompt
    # [修改点] 在 Task Requirements 中增加了对 Prior Knowledge/Formulas 的要求
    prompt = f"""You are an expert in optimizing Chain-of-Thought (CoT) efficiency for VLM.
Your task is to compress the Assistant's response into a structured, high-density format.

--- Input Data ---
[Question]: 
{question_content}

[Original Response]:
{original_response}

--- Task Requirements ---
1. **Deconstruct**: Analyze the [Original Response] to separate visual observations, logical reasoning, and the final answer.
2. **Compress**: Rewrite the content into a "Dense Cognitive Trace".
   - **Style**: Telegraphic (arrows ->, symbols, short phrases). NO fillers.
   - **Visual**: Extract visual evidence mentioned in the thought process.
   - **Think**: 
     - **Step 1 (Priors)**: At the very beginning, briefly list necessary commonsense or formulas (e.g., "Area=πr²", "Boiling point=100°C") if applicable.
     - **Step 2 (Logic)**: Compress the subsequent logical steps.
3. **Format**: The "dense_version" string MUST strictly follow this XML structure:
   <visual>...concise visual evidence...</visual><think>...[Priors] -> ...compressed reasoning...</think><answer>...final answer...</answer>

--- Output Format ---
Return a strictly valid JSON object:
{{
    "original_length": <int, char count of original text>,
    "dense_version": "<string, the compressed text containing <visual>, <think>, and <answer> tags>",
    "dense_length": <int, char count of dense version>,
    "efficiency_ratio": <float, dense_length / original_length>,
    "waste_analysis": "<string, brief comment on what was removed>"
}}
"""
    return prompt

# ==========================================
# 单条处理函数
# ==========================================
def process_single_item(item, gpt_instance):
    # 使用 images 路径作为 ID
    p_id = str(item.get('id', item.get('images', 'unknown_id')))
    
    prompt = generate_density_analysis_prompt(item)

    if not prompt:
        return None

    messages = [{"role": "user", "content": prompt}]

    try:
        # 调用模型
        response_text = gpt_instance.send_stable_request(messages, temperature=0.1)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            analysis_json = json.loads(clean_text)
            
            # 简单校验标签是否存在
            dense_content = analysis_json.get("dense_version", "")
            if "<visual>" not in dense_content or "<think>" not in dense_content:
                analysis_json["format_warning"] = "Tags missing in output"

        except json.JSONDecodeError:
            analysis_json = {
                "error": "JSON Parse Failed",
                "raw_output": clean_text,
                "efficiency_ratio": 1.0
            }

        return {
            "problem_id": p_id,
            "original_item": item,
            "dense_analysis": analysis_json
        }

    except Exception as e:
        print(f"\n[!] Error processing {p_id}: {e}")
        return None

# ==========================================
# 并行主逻辑
# ==========================================
def analyze_think_efficiency_parallel(input_file, output_file, gpt_model, num_data_to_process=1000, max_workers=10):
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:num_data_to_process]
    print(f"[*] 加载数据成功，共 {len(data)} 条")

    results = []
    processed_ids = set()

    # 断点续传
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                results = saved_data
                for item in results:
                    pid = item.get('problem_id')
                    if not pid and 'original_item' in item:
                        pid = item['original_item'].get('images')
                    if pid:
                        processed_ids.add(str(pid))
            print(f"[*] 已加载 {len(results)} 条已有分析")
        except Exception as e:
            print(f"[!] 结果文件读取失败，重新开始: {e}")

    tasks_to_run = []
    for item in data:
        p_id = str(item.get('id', item.get('images', 'unknown_id')))
        if p_id not in processed_ids:
            tasks_to_run.append(item)

    print(f"[*] 剩余待处理任务: {len(tasks_to_run)} 条")
    print(f"[*] 开启并行处理，线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, gpt_model): item 
            for item in tasks_to_run
        }

        completed_count = 0
        save_interval = 20  

        for future in tqdm(as_completed(future_to_item), total=len(tasks_to_run), desc="Parallel Analysis"):
            result = future.result()
            if result:
                results.append(result)
            completed_count += 1

            if completed_count % save_interval == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[*] 分析完成！结果保存在: {output_file}")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    input_filename = os.environ.get("VISION_R1_SFT_INPUT_JSON", "./vision_r1_mulberry_sft_full.json")
    output_filename = os.environ.get("VISION_R1_SFT_OUTPUT_JSON", "./Vision-R1-short-cot-formatted.json")

    gpt = GPT4o(deployment_name=os.environ.get("VISION_R1_SFT_MODEL", "gpt-4o"))
    
    analyze_think_efficiency_parallel(
        input_filename,
        output_filename,
        gpt_model=gpt,
        num_data_to_process=int(os.environ.get("VISION_R1_SFT_LIMIT", "200000")),
        max_workers=int(os.environ.get("VISION_R1_SFT_WORKERS", "8")),
    )
