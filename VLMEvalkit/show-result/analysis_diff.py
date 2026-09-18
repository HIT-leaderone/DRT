import json
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RESULT_DIR = THIS_DIR / "result_analysis" / "VCR-bench"

def load_analysis_files(file_pattern):
    """读取所有匹配的 JSON 分析文件"""
    files = glob.glob(file_pattern)
    all_models_data = {}
    
    print(f"[*] 找到 {len(files)} 个文件匹配模式: {file_pattern}")
    
    for file_path in files:
        # 提取简短的模型名作为标识
        model_name = os.path.basename(file_path).replace("_Error_Analysis_Report.json", "")
        # 如果名字太长，截断一下方便显示
        if len(model_name) > 30:
            model_name = "..." + model_name[-25:]
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_models_data[model_name] = data
                print(f"    [+] 已加载: {model_name} ({len(data)} 条样本)")
        except Exception as e:
            print(f"    [!] 加载失败 {file_path}: {e}")
            
    return all_models_data

def generate_statistical_report(all_models_data):
    """生成宏观统计报告"""
    stats_list = []
    
    for model_name, data in all_models_data.items():
        # 提取关键指标
        useful_ratios = []
        error_types = []
        scores = [] # 原始分数
        
        for idx, item in data.items():
            analysis = item.get('error_analysis', {})
            original = item.get('original_data', {})
            
            # 1. 有效信息比例
            ratio = analysis.get('useful_ratio')
            if isinstance(ratio, (int, float)):
                useful_ratios.append(ratio)
            
            # 2. 错误类型
            e_type = analysis.get('error_type', 'Unknown')
            error_types.append(e_type)
            
            # 3. 原始分数 (如果有)
            score = original.get('score')
            if isinstance(score, (int, float)):
                scores.append(score)

        # 构建统计字典
        stats = {
            "Model": model_name,
            "Sample Count": len(data),
            "Avg Useful Ratio": np.mean(useful_ratios) if useful_ratios else 0,
            "Avg Original Score": np.mean(scores) if scores else 0,
            "Correct Rate": error_types.count("Correct") / len(error_types) if error_types else 0,
            "Hallucination Rate": error_types.count("Hallucination") / len(error_types) if error_types else 0,
        }
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)

def analyze_error_distribution(all_models_data):
    """分析错误类型分布对比"""
    dist_data = []
    
    for model_name, data in all_models_data.items():
        error_counts = {}
        total = 0
        for item in data.values():
            e_type = item.get('error_analysis', {}).get('error_type', 'Unknown')
            error_counts[e_type] = error_counts.get(e_type, 0) + 1
            total += 1
            
        # 转为百分比
        for k, v in error_counts.items():
            dist_data.append({
                "Model": model_name,
                "Error Type": k,
                "Percentage": (v / total) * 100
            })
            
    return pd.DataFrame(dist_data)

def find_significant_diffs(all_models_data, model_a_name, model_b_name, diff_threshold=0.3):
    """
    寻找两个模型差异巨大的样本
    条件：Useful Ratio 差异超过 threshold，或者 Error Type 一个是 Correct 一个不是
    """
    if model_a_name not in all_models_data or model_b_name not in all_models_data:
        print("[!] 指定的模型名称不存在，无法对比")
        return []

    data_a = all_models_data[model_a_name]
    data_b = all_models_data[model_b_name]
    
    # 找交集 index
    common_indices = set(data_a.keys()) & set(data_b.keys())
    
    diff_cases = []
    
    for idx in common_indices:
        info_a = data_a[idx].get('error_analysis', {})
        info_b = data_b[idx].get('error_analysis', {})
        
        ratio_a = info_a.get('useful_ratio', 0)
        ratio_b = info_b.get('useful_ratio', 0)
        type_a = info_a.get('error_type', 'Unknown')
        type_b = info_b.get('error_type', 'Unknown')
        
        # 判断差异逻辑
        ratio_diff = abs(ratio_a - ratio_b)
        type_diff = (type_a == 'Correct' and type_b != 'Correct') or (type_b == 'Correct' and type_a != 'Correct')
        
        if ratio_diff > diff_threshold or type_diff:
            diff_cases.append({
                "Index": idx,
                f"{model_a_name}_Type": type_a,
                f"{model_a_name}_Ratio": ratio_a,
                f"{model_b_name}_Type": type_b,
                f"{model_b_name}_Ratio": ratio_b,
                "Diff_Reason": "Type Mismatch" if type_diff else "Ratio Gap"
            })
            
    return pd.DataFrame(diff_cases)

# ==========================================
# 主执行逻辑
# ==========================================
if __name__ == "__main__":
    # 1. 设置文件匹配模式
    # 假设你的文件都在结果目录，且以 Qwen3VL-8B 开头
    file_pattern = str(RESULT_DIR / "Qwen3VL-8B*.json")
    
    # 2. 加载数据
    all_data = load_analysis_files(file_pattern)
    
    if not all_data:
        print("未找到数据，请检查路径")
        exit()

    # 3. 宏观统计表格
    print("\n" + "="*20 + " 宏观表现对比 (Overall Stats) " + "="*20)
    df_stats = generate_statistical_report(all_data)
    # 格式化输出
    print(df_stats.to_markdown(index=False, floatfmt=".3f"))
    
    # 4. 错误分布透视表
    print("\n" + "="*20 + " 错误类型分布 (Error Distribution %) " + "="*20)
    df_dist = analyze_error_distribution(all_data)
    if not df_dist.empty:
        pivot_dist = df_dist.pivot(index="Model", columns="Error Type", values="Percentage").fillna(0)
        print(pivot_dist.to_markdown(floatfmt=".1f"))

    # 5. (可选) 两个特定模型的深度 Diff
    # 如果你有两个模型想重点看，可以在这里指定它们的名字（上面打印出来的名字）
    models = list(all_data.keys())
    if len(models) >= 2:
        m1, m2 = models[0], models[1]
        print(f"\n" + "="*20 + f" 深度 Diff: {m1} vs {m2} " + "="*20)
        df_diff = find_significant_diffs(all_data, m1, m2)
        
        if not df_diff.empty:
            # 按 Ratio 差异排序，看最大的 Gap
            df_diff['Ratio_Gap'] = abs(df_diff[f"{m1}_Ratio"] - df_diff[f"{m2}_Ratio"])
            top_diffs = df_diff.sort_values('Ratio_Gap', ascending=False).head(10)
            print(top_diffs.to_markdown(index=False, floatfmt=".2f"))
            
            # 保存 Diff 结果到 CSV 方便人工查看
            df_diff.to_csv("model_diff_analysis.csv", index=False)
            print(f"\n[*] 详细 Diff 结果已保存至 model_diff_analysis.csv")
        else:
            print("未发现显著差异样本。")
