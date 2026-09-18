import pandas as pd
import numpy as np
import os.path as osp
from vlmeval.dataset import ImageMCQDataset
from vlmeval.smp import load, dump, get_intermediate_file_path

class ScienceQADataset(ImageMCQDataset):
    
    # 使用 VLMEvalKit 官方托管的 ScienceQA TSV 数据链接
    # 对应 HuggingFace: derek-thomas/ScienceQA
    DATASET_URL = {
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv',
    }
    
    DATASET_MD5 = {
        'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
        'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        """
        重写 evaluate 方法以生成 ScienceQA 特定的细粒度报表
        """
        # 1. 调用父类方法进行推理和基础评估，获取总体准确率
        # 这会生成预测结果并保存到文件中
        acc = super().evaluate(eval_file, **judge_kwargs)
        
        # 2. 获取结果文件路径 (通常是 {eval_file}_{model}_result.pkl)
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4', "gpt-4.1-2025-04-14": "gpt4.1"}
        name_str = name_str_map[model] if model in name_str_map else model
        
        suffix = eval_file.split('.')[-1]
        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')
        
        # 如果 pkl 不存在，尝试读取 xlsx 或原始 eval_file (fallback)
        if not osp.exists(result_file):
            result_file_xlsx = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'xlsx')
            if osp.exists(result_file_xlsx):
                data = load(result_file_xlsx)
            else:
                data = load(eval_file)
        else:
            data = load(result_file)

        # 3. 确保数据包含必要的元数据 (subject, grade, image, hint)
        # 如果结果文件是中间生成的，可能缺少部分元数据，需要从 self.data 补全
        if 'subject' not in data.columns or 'grade' not in data.columns:
            meta = self.data
            # 确保 index 列类型一致以便合并
            data['index'] = data['index'].astype(str)
            meta['index'] = meta['index'].astype(str)
            
            # 提取需要的元数据列
            meta_cols = ['index', 'subject', 'grade', 'image', 'hint']
            # 仅选择存在的列
            meta_cols = [c for c in meta_cols if c in meta.columns]
            meta_subset = meta[meta_cols]
            
            # 合并
            data = pd.merge(data, meta_subset, on='index', how='left', suffixes=('', '_meta'))
            
            # 优先使用 meta 中的数据填充
            for col in ['subject', 'grade', 'image', 'hint']:
                if f'{col}_meta' in data.columns:
                    data[col] = data[f'{col}_meta'].fillna(data[col])
                    data.drop(columns=[f'{col}_meta'], inplace=True)

        # 4. 定义分类逻辑
        def get_context_type(row):
            # 判断是否有图像 (IMG)
            has_image = False
            if 'image' in row and pd.notna(row['image']):
                img_val = str(row['image'])
                if img_val.strip() and img_val.lower() != 'nan':
                    has_image = True
            
            # 判断是否有 Hint (TXT)
            has_hint = False
            if 'hint' in row and pd.notna(row['hint']):
                hint_val = str(row['hint'])
                if hint_val.strip() and hint_val.lower() != 'nan':
                    has_hint = True
            
            # 优先级: IMG > TXT > NO (ScienceQA 常用分类逻辑)
            if has_image:
                return 'IMG'
            elif has_hint:
                return 'TXT'
            else:
                return 'NO'

        def get_grade_group(val):
            # 将 grade1-grade12 映射到 G1-6 和 G7-12
            s = str(val).lower()
            import re
            nums = re.findall(r'\d+', s)
            if not nums:
                return 'Other'
            g = int(nums[0])
            if 1 <= g <= 6:
                return 'G1-6'
            elif 7 <= g <= 12:
                return 'G7-12'
            return 'Other'

        # 应用分类
        data['context_type'] = data.apply(get_context_type, axis=1)
        data['grade_group'] = data['grade'].apply(get_grade_group)

        # 5. 计算各维度准确率
        metrics = {}
        
        # Avg (Overall)
        metrics['Avg'] = data['hit'].mean()
        
        # Subject: NAT (Natural Science), SOC (Social Science), LAN (Language Science)
        subject_map = {
            'NAT': 'natural science',
            'SOC': 'social science',
            'LAN': 'language science'
        }
        for code, name in subject_map.items():
            sub_df = data[data['subject'] == name]
            metrics[code] = sub_df['hit'].mean() if len(sub_df) > 0 else 0.0
            
        # Context: IMG, TXT, NO
        for ctx in ['IMG', 'TXT', 'NO']:
            sub_df = data[data['context_type'] == ctx]
            metrics[ctx] = sub_df['hit'].mean() if len(sub_df) > 0 else 0.0
            
        # Grade: G1-6, G7-12
        for grp in ['G1-6', 'G7-12']:
            sub_df = data[data['grade_group'] == grp]
            metrics[grp] = sub_df['hit'].mean() if len(sub_df) > 0 else 0.0

        # 6. 生成并打印报表
        # 转换为百分比
        results = {k: v * 100 for k, v in metrics.items()}
        
        # 按照要求的顺序排列
        headers = ['NAT', 'SOC', 'LAN', 'TXT', 'IMG', 'NO', 'G1-6', 'G7-12', 'Avg']
        row = [results.get(h, 0.0) for h in headers]
        
        df_report = pd.DataFrame([row], columns=headers)
        
        print(f"\nScienceQA Detailed Performance ({name_str}):")
        print(df_report.to_markdown(index=False, floatfmt=".2f"))
        
        # 保存详细报表到 CSV
        report_path = eval_file.replace(f'.{suffix}', f'_{name_str}_scienceqa_report.csv')
        df_report.to_csv(report_path, index=False)
        
        return acc
