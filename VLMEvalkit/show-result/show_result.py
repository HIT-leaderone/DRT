import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RESULT_DIR = THIS_DIR / "result_analysis" / "VCR-bench"

def load_data_to_df(file_path):
    """加载 JSON 并转换为 Pandas DataFrame (适配深度分析字段)"""
    if not os.path.exists(file_path):
        print(f"[!] 文件不存在: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "detailed_reports" in data:
        reports = data["detailed_reports"]
    else:
        reports = data

    rows = []
    for idx, item in reports.items():
        analysis = item.get('error_analysis', {})
        original = item.get('original_data', {})
        # 获取预测文本长度
        prediction_text = str(original.get('prediction', ''))
        pred_len = len(prediction_text)
        # 优先读取深度分析生成的 category/root_cause，如果没有则回退到 label
        cat = analysis.get('waste_category', analysis.get('waste_label', 'None'))
        root = analysis.get('waste_root_cause', 'Unspecified')

        # 如果是 None 或 Uncategorized，统一处理
        if cat in ['Uncategorized', 'None']: cat = 'None'
        if root in ['Uncategorized', 'None']: root = 'Unspecified'

        row = {
            "id": idx,
            "l1": analysis.get('level_1_category', 'Unknown'),
            "l2": analysis.get('level_2_category', 'N/A'),
            "useful_ratio": analysis.get('useful_ratio', 0.0),
            "waste_cat": cat,
            "waste_root": root,
            "pred_len": pred_len 
        }
        rows.append(row)

    return pd.DataFrame(rows)

def print_ascii_bar(label, count, total, width=30, indent=0):
    """打印进度条"""
    if total == 0:
        pct = 0
        bar_len = 0
    else:
        pct = (count / total) * 100
        bar_len = int((count / total) * width)

    bar = '█' * bar_len
    space = " " * indent
    # 格式化输出：Label (Count, Pct) | Bar
    print(f"{space}{label:<40} : {count:>3} ({pct:>5.1f}%) | {bar}")

def print_ascii_histogram(series, bins=10, title="Histogram"):
    """打印直方图"""
    counts, bin_edges = np.histogram(series, bins=bins, range=(0, 1))
    total = len(series)

    print(f"\n--- {title} ---")
    for i in range(len(counts)):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        count = counts[i]
        if total > 0:
            pct = (count / total) * 100
            bar_len = int((count / total) * 50)
        else:
            pct = 0
            bar_len = 0
        bar = '#' * bar_len

        print(f"{lower:.1f}-{upper:.1f} : {count:>4} ({pct:>5.1f}%) | {bar}")

def analyze_comprehensive_report(file_path):
    df = load_data_to_df(file_path)
    if df is None or df.empty:
        return

    print(f"{'='*50}\n 📊 VQA Full Analysis Report (Deep Dive) \n{'='*50}")
    print(f"Total Samples: {len(df)}")

    # ==========================================
    # PART 1: Error Taxonomy (L1 & L2 分布) [保留]
    # ==========================================
    print(f"\n{'='*20} 1. Error Taxonomy Distribution {'='*20}")

    l1_counts = df['l1'].value_counts()

    for l1_name, l1_count in l1_counts.items():
        l1_pct = (l1_count / len(df)) * 100
        print(f"\n[L1] {l1_name}: {l1_count} ({l1_pct:.1f}%)")

        if l1_name == "Correct":
            continue

        sub_df = df[df['l1'] == l1_name]
        l2_counts = sub_df['l2'].value_counts()

        for l2_name, l2_count in l2_counts.items():
            l2_pct_local = (l2_count / l1_count) * 100
            print(f"    ├── [L2] {l2_name:<40} : {l2_count:>3} (Local: {l2_pct_local:>4.1f}%)")

    # ==========================================
    # PART 2: Information Density & Length Analysis
    # ==========================================
    print(f"\n{'='*20} 2. Information Density & Prediction Length {'='*20}")

    # --- 1. 整体统计 ---
    mean_ratio = df['useful_ratio'].mean()
    median_ratio = df['useful_ratio'].median()
    mean_len = df['pred_len'].mean() # 新增
    median_len = df['pred_len'].median() # 新增
    print(f"--- Overall (All Samples) ---")
    print(f"Average Useful Ratio : {mean_ratio:.4f}")
    print(f"Median Useful Ratio  : {median_ratio:.4f}")
    print(f"Average Length (char): {mean_len:.1f}") # 新增
    print(f"Median Length (char) : {median_len:.1f}") # 新增
    print_ascii_histogram(df['useful_ratio'], title="Overall Useful Ratio Distribution")
    # --- 2. Correct 样本统计 ---
    correct_df = df[df['l1'] == 'Correct']
    if not correct_df.empty:
        mean_c = correct_df['useful_ratio'].mean()
        median_c = correct_df['useful_ratio'].median()
        mean_len_c = correct_df['pred_len'].mean() # 新增
        median_len_c = correct_df['pred_len'].median() # 新增
        print(f"\n--- Correct Samples Only (N={len(correct_df)}) ---")
        print(f"Average Useful Ratio : {mean_c:.4f}")
        print(f"Median Useful Ratio  : {median_c:.4f}")
        print(f"Average Length (char): {mean_len_c:.1f}") # 新增
        print(f"Median Length (char) : {median_len_c:.1f}") # 新增
        print_ascii_histogram(correct_df['useful_ratio'], title="Correct Samples Distribution")
    else:
        print("\n[!] No Correct samples found.")
    # --- 3. Wrong 样本统计 ---
    wrong_df = df[df['l1'] != 'Correct']
    if not wrong_df.empty:
        mean_w = wrong_df['useful_ratio'].mean()
        median_w = wrong_df['useful_ratio'].median()
        mean_len_w = wrong_df['pred_len'].mean() # 新增
        median_len_w = wrong_df['pred_len'].median() # 新增
        print(f"\n--- Wrong Samples Only (N={len(wrong_df)}) ---")
        print(f"Average Useful Ratio : {mean_w:.4f}")
        print(f"Median Useful Ratio  : {median_w:.4f}")
        print(f"Average Length (char): {mean_len_w:.1f}") # 新增
        print(f"Median Length (char) : {median_len_w:.1f}") # 新增
        print_ascii_histogram(wrong_df['useful_ratio'], title="Wrong Samples Distribution")
    else:
        print("\n[!] No Wrong samples found.")

    # ==========================================
    # PART 3: General Waste Analysis [保留]
    # ==========================================
    print(f"\n{'='*20} 3. General Waste Analysis (Overview) {'='*20}")

    waste_df = df[df['waste_cat'] != 'None']
    total_waste = len(waste_df)

    print(f"Samples containing waste: {total_waste} ({total_waste/len(df)*100:.1f}% of total)")

    if total_waste > 0:
        print("\n--- Top Waste Categories ---")
        waste_counts = waste_df['waste_cat'].value_counts()
        for label, count in waste_counts.items():
            print_ascii_bar(label, count, total_waste)

    # ==========================================
    # PART 4: Waste in CORRECT Samples [修改：层级展示]
    # ==========================================
    print(f"\n{'='*20} 4. FOCUS: Waste in CORRECT Samples (Deep Dive) {'='*20}")

    correct_df = df[df['l1'] == 'Correct']
    total_correct = len(correct_df)

    if total_correct > 0:
        # 筛选出有废话的 Correct 样本
        wasteful_correct_df = correct_df[correct_df['waste_cat'] != 'None']
        waste_count_c = len(wasteful_correct_df)
        perfect_count = total_correct - waste_count_c

        print(f"Total Correct Samples: {total_correct}")
        print(f"  ✅ Perfectly Concise (No Waste) : {perfect_count} ({perfect_count/total_correct*100:.1f}%)")
        print(f"  ⚠️ Correct but Wasteful         : {waste_count_c} ({waste_count_c/total_correct*100:.1f}%)")

        if waste_count_c > 0:
            print("\n--- Hierarchy of Waste in Correct Answers ---")
            
            # 1. 先按大类 (Category) 统计并排序
            cat_counts = wasteful_correct_df['waste_cat'].value_counts()
            
            for cat, c_count in cat_counts.items():
                # 打印大类条形图
                print(f"\n📦 Category: {cat}")
                print_ascii_bar("Total", c_count, waste_count_c, width=20, indent=3)
                
                # 2. 在该大类下，统计根因 (Root Cause)
                sub_df = wasteful_correct_df[wasteful_correct_df['waste_cat'] == cat]
                root_counts = sub_df['waste_root'].value_counts()
                
                for root, r_count in root_counts.items():
                    # 打印根因 (缩进显示)
                    # 计算根因占该大类的比例
                    r_pct_local = (r_count / c_count) * 100
                    print(f"      └── {root:<35} : {r_count:>2} ({r_pct_local:>4.1f}%)")
                    
    else:
        print("No Correct samples found.")

    # 保存
    csv_path = file_path.replace(".json", "_full_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] 完整统计数据已保存至: {csv_path}")

if __name__ == "__main__":
    # 请确保这里指向的是包含 deep root cause 的 json 文件
    input_file = str(RESULT_DIR / "Qwen3VL-8B_Deep_Root_Cause_Report.json")
    analyze_comprehensive_report(input_file)
