import os
import os.path as osp
import pandas as pd
from datasets import load_dataset
from vlmeval.dataset import ImageBaseDataset
from vlmeval.smp import *

class RealWorldQADataset(ImageBaseDataset):
    
    # 定义数据集名称，用于生成文件名
    DATASET_NAME = 'RealWorldQA'

    def __init__(self, dataset='RealWorldQA', skip_noimg=True):
        # 1. 准备数据路径
        data_root = LMUDataRoot()
        if not osp.exists(data_root):
            os.makedirs(data_root, exist_ok=True)
            
        # 定义本地 TSV 和图片存储路径
        self.dataset_dir = osp.join(data_root, 'RealWorldQA')
        self.tsv_file = osp.join(self.dataset_dir, 'RealWorldQA.tsv')
        self.image_dir = osp.join(self.dataset_dir, 'images')
        
        # 2. 如果本地 TSV 不存在，则从 HuggingFace 下载并转换
        if not osp.exists(self.tsv_file):
            print(f"Dataset not found at {self.tsv_file}. Downloading from HuggingFace (xai-org/RealworldQA)...")
            self.prepare_dataset()
            
        # 3. 初始化父类
        # 注意：这里我们传入 dataset_name 而不是 URL，因为我们已经本地准备好了文件
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        
        # 强制覆盖父类的 data_root 和 data_file，确保指向我们生成的本地文件
        self.data_root = data_root
        self.data_file = self.tsv_file
        # 重新加载数据以确保使用正确的本地文件
        self.data = load(self.data_file)

    def prepare_dataset(self):
        """
        从 HuggingFace 下载数据，保存图片，并生成 TSV 文件
        """
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 加载 HF 数据集 (仅 test 集)
        try:
            ds = load_dataset("xai-org/RealworldQA", split="test")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e

        data_list = []
        print(f"Processing {len(ds)} samples...")
        
        for i, item in enumerate(ds):
            # 1. 保存图片
            img = item['image']
            img_name = f"{i:06d}.jpg"
            img_path = osp.join(self.image_dir, img_name)
            
            # 如果图片是 RGB 模式，直接保存；否则转换
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img_path)
            
            # 2. 构建数据行
            # RealWorldQA 的 HF 格式: image, question, answer
            # VLMEvalKit 需要: index, image, question, answer
            row = {
                'index': i,
                'image': img_name, # 存相对路径或文件名
                'question': item['question'],
                'answer': item['answer']
            }
            data_list.append(row)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} images")

        # 3. 保存为 TSV
        df = pd.DataFrame(data_list)
        dump(df, self.tsv_file)
        print(f"Dataset prepared and saved to {self.tsv_file}")

    def evaluate(self, eval_file, **judge_kwargs):
        """
        RealWorldQA 通常使用 Exact Match (EM) 进行评估。
        """
        # 1. 运行推理
        super().evaluate(eval_file, **judge_kwargs)
        
        # 2. 获取结果文件
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
        
        if not osp.exists(result_file):
            # Fallback
            result_file_xlsx = eval_file.replace(f'.{suffix}', f'_{name_str}_result.xlsx')
            if osp.exists(result_file_xlsx):
                data = load(result_file_xlsx)
            else:
                data = load(eval_file)
        else:
            data = load(result_file)

        # 3. 计算 Exact Match Accuracy
        # RealWorldQA 的 answer 列通常是单个词、数字或选项字母 (A/B/C)
        # 我们使用简单的字符串匹配，忽略大小写和标点
        
        if 'hit' not in data.columns:
            # 如果没有 hit 列，手动计算
            # 定义一个简单的清洗函数
            def clean_text(s):
                return str(s).strip().lower().replace('.', '')

            def is_correct(row):
                pred = clean_text(row['prediction'])
                ans = clean_text(row['answer'])
                # 简单的包含匹配或精确匹配
                return 1 if ans in pred or pred in ans else 0
            
            data['hit'] = data.apply(is_correct, axis=1)

        acc = data['hit'].mean()
        
        print(f"\nRealWorldQA Results ({name_str}):")
        print(f"Accuracy: {acc:.2%}")
        
        return acc
