import json
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from gpt_model import GPT4o  # 假设你的 GPT4o 类在这个文件中

# ==========================================
# 核心逻辑：思维密度分析
# ==========================================

def generate_density_analysis_prompt(item):
    # ... (保持原样，无需修改) ...
    question = item.get('problem', '')
    options = str(item.get('options', []))
    original_think = item.get('process', '')

    if not original_think:
        return None

    prompt = f"""You are an expert in optimizing Chain-of-Thought (CoT) efficiency.
Your task is to analyze the "Information Density" of a VLM's thought process.

--- Input Data ---
[Question]: {question}
[Options]: {options}
[Original Think]:
{original_think}

--- Task Requirements ---
1. **Analyze**: Identify the valid visual perceptions and logical inferences in the [Original Think].
2. **Compress**: Rewrite the thought into a "Dense Cognitive Trace".
   - **Constraint**: EXTREME utilization. Every token must contribute to the answer.
   - **Format**: Use telegraphic style, arrows (->), symbols, or short phrases. NO conversational fillers (e.g., "Let me think", "I see", "Therefore").
   - **Content**: Only include: Visual Evidence -> Inference -> Decision/Elimination.
3. **Calculate**: Compare the character count (ignoring whitespace) of the Dense version vs. Original version.

--- Output Format ---
Return a strictly valid JSON object:
{{
    "original_length": <int, char count of original text>,
    "dense_version": "<string, the compressed telegraphic thought>",
    "dense_length": <int, char count of dense version>,
    "efficiency_ratio": <float, dense_length / original_length, e.g., 0.25>,
    "waste_analysis": "<string, brief comment on what was removed (e.g., 'Removed self-correction and redundant summary')>"
}}
"""
    return prompt

# ==========================================
# 新增：单条处理函数 (线程内运行)
# ==========================================
def process_single_item(item, model_name):
    """
    处理单条数据的函数，用于提交给线程池
    注意：这里每次实例化 GPT4o 可能会有开销，如果 GPT4o 类是线程安全的，
    可以在外部实例化后传入。为了保险起见，这里假设需要独立实例或它内部处理了连接池。
    """
    p_id = str(item.get('problem_id', 'unknown'))
    prompt = generate_density_analysis_prompt(item)
    
    if not prompt:
        return None

    # 建议：如果 GPT4o 类内部使用了 requests.Session，最好在外部实例化传入
    # 这里为了简单，假设每次新建实例（如果初始化很慢，请改为传入 gpt 实例）
    gpt = GPT4o(deployment_name=model_name)
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # 调用模型
        response_text = gpt.send_stable_request(messages, temperature=0.1)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            analysis_json = json.loads(clean_text)
        except:
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
def analyze_think_efficiency_parallel(input_file, output_file, model_name="gpt-4o", num_data_to_process=1000, max_workers=10):
    # 1. 读取数据
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 切片
    data = data[:num_data_to_process]
    print(f"[*] 加载数据成功，共 {len(data)} 条")

    # 2. 准备结果容器 (断点续传)
    results = []
    processed_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                results = saved_data
                processed_ids = {str(item['original_data'].get('problem_id')) if 'original_data' in item else str(item['original_item'].get('problem_id')) for item in results}
            print(f"[*] 已加载 {len(results)} 条已有分析")
        except Exception as e:
            print(f"[!] 结果文件读取失败，重新开始: {e}")

    # 3. 过滤待处理任务
    tasks_to_run = []
    for item in data:
        p_id = str(item.get('problem_id', 'unknown'))
        if p_id not in processed_ids:
            tasks_to_run.append(item)
    
    print(f"[*] 剩余待处理任务: {len(tasks_to_run)} 条")

    # 4. 并行处理
    # max_workers 建议设置为 5-20，取决于你的 API 速率限制 (RPM/TPM)
    print(f"[*] 开启并行处理，线程数: {max_workers}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_single_item, item, model_name): item for item in tasks_to_run}
        
        # 使用 tqdm 监控完成进度
        completed_count = 0
        save_interval = 20  # 每处理完 20 条保存一次，避免频繁 IO
        
        for future in tqdm(as_completed(future_to_item), total=len(tasks_to_run), desc="Parallel Analysis"):
            result = future.result()
            
            if result:
                results.append(result)
                
                # 简单的日志打印
                if len(results) % 5 == 0:
                    ratio = result['dense_analysis'].get('efficiency_ratio', 0)
                    # print(f" [Ratio: {ratio:.2f}]", end="", flush=True) # 可选：减少刷屏

            completed_count += 1
            
            # 定期保存 (在主线程进行，线程安全)
            if completed_count % save_interval == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[*] 分析完成！结果保存在: {output_file}")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    input_filename = os.environ.get("VISION_R1_INPUT_JSON", "./Video-R1-COT-165k.json")
    output_filename = os.environ.get("DENSE_REASON_OUTPUT_JSON", "./Video-R1_Think_Density_Report.json")

    # 建议根据你的 API 限流情况调整 max_workers
    # 如果是 GPT-4o 这种高并发模型，可以开到 10-20
    # 如果经常遇到 429 错误，请调低这个数字
    analyze_think_efficiency_parallel(
        input_filename, 
        output_filename, 
        model_name=os.environ.get("DENSE_REASON_MODEL", "gpt-4o"),
        num_data_to_process=int(os.environ.get("DENSE_REASON_LIMIT", "165000")),
        max_workers=int(os.environ.get("DENSE_REASON_WORKERS", "4")),
    )
