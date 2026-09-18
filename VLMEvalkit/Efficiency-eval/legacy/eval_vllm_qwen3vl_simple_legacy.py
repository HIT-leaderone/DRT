#!/usr/bin/env python3
"""
Qwen3-VL-7B VLLM 评测脚本 - 简化版
支持数据集: MathVista_MINI, MathVision, MathVerse_MINI, LogicVista

使用方法:
    python Efficiency-eval/legacy/eval_vllm_qwen3vl_simple_legacy.py \
        --vllm-url http://localhost:8000 \
        --output-dir ./results
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.smp import dump


class VLLMClient:
    """VLLM API 客户端"""
    
    def __init__(self, base_url: str, model_name: str = "Qwen3-VL-7B-Instruct"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def chat(self, messages: List[Dict[str, Any]], 
             temperature: float = 0.7,
             max_tokens: int = 16384,
             top_p: float = 0.8,
             top_k: int = 20) -> str:
        """发送聊天请求到 VLLM"""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        if top_k > 0:
            payload["top_k"] = top_k
            
        try:
            response = requests.post(
                self.chat_url, 
                headers=headers, 
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"VLLM API 请求失败: {e}")
            raise


def load_tsv_dataset(dataset_name: str) -> pd.DataFrame:
    """加载 TSV 数据集"""
    from vlmeval.smp import LMUDataRoot
    
    data_root = LMUDataRoot()
    tsv_path = os.path.join(data_root, f"{dataset_name}.tsv")
    
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"数据集文件不存在: {tsv_path}")
    
    df = pd.read_csv(tsv_path, sep='\t')
    return df


def decode_base64_image(base64_str: str, output_dir: str, index: str) -> str:
    """解码 base64 图片并保存"""
    img_path = os.path.join(output_dir, f"{index}.jpg")
    
    if os.path.exists(img_path):
        return img_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 解码 base64
    img_data = base64.b64decode(base64_str)
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    return img_path


def build_messages(image_path: str, question: str) -> List[Dict[str, Any]]:
    """构建 VLLM 消息格式"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]
    return messages


def evaluate_dataset(
    dataset_name: str,
    vllm_client: VLLMClient,
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    top_p: float = 0.8,
    top_k: int = 20
) -> Dict[str, Any]:
    """评测单个数据集"""
    
    print(f"\n{'='*60}")
    print(f"评测 {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据集
    df = load_tsv_dataset(dataset_name)
    num_samples = len(df)
    print(f"样本数量: {num_samples}")
    
    # 创建图片缓存目录
    img_cache_dir = os.path.join(output_dir, "images", dataset_name)
    os.makedirs(img_cache_dir, exist_ok=True)
    
    # 存储预测结果
    predictions = []
    start_time = time.time()
    
    for idx in tqdm(range(num_samples), desc=f"评测 {dataset_name}"):
        row = df.iloc[idx]
        
        try:
            # 获取图片和问题
            image_base64 = row.get('image', '')
            if pd.isna(image_base64) or not image_base64:
                # 尝试从 image_path 加载
                image_path = row.get('image_path', '')
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            if not image_base64:
                print(f"警告: 第 {idx} 个样本没有图片")
                predictions.append({
                    'index': row.get('index', idx),
                    'prediction': '',
                    'answer': row.get('answer', ''),
                    'question': row.get('question', '')
                })
                continue
            
            # 解码图片
            image_path = decode_base64_image(
                image_base64, 
                img_cache_dir, 
                str(row.get('index', idx))
            )
            
            # 获取问题
            question = row.get('question', '')
            if pd.isna(question):
                question = ''
            
            # 构建消息并调用 VLLM
            messages = build_messages(image_path, question)
            response = vllm_client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            
            predictions.append({
                'index': row.get('index', idx),
                'prediction': response,
                'answer': row.get('answer', ''),
                'question': question
            })
            
        except Exception as e:
            print(f"处理第 {idx} 个样本时出错: {e}")
            predictions.append({
                'index': row.get('index', idx),
                'prediction': '',
                'answer': row.get('answer', ''),
                'question': row.get('question', '')
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 保存预测结果
    pred_df = pd.DataFrame(predictions)
    pred_path = os.path.join(output_dir, f"{dataset_name}_predictions.xlsx")
    dump(pred_df, pred_path)
    
    result = {
        'dataset_name': dataset_name,
        'num_samples': num_samples,
        'total_time': total_time,
        'avg_time_per_sample': total_time / num_samples if num_samples > 0 else 0,
        'predictions_file': pred_path
    }
    
    return result


def print_summary(results: List[Dict[str, Any]]):
    """打印评测结果汇总"""
    print("\n" + "="*80)
    print("评测结果汇总")
    print("="*80)
    
    # 单个数据集结果
    for result in results:
        print(f"\n【{result['dataset_name']}】")
        print(f"  样本数量: {result['num_samples']}")
        print(f"  总评测时间: {result['total_time']:.2f} 秒")
        print(f"  平均每个样本时间: {result['avg_time_per_sample']:.2f} 秒")
    
    # 合并统计
    print("\n" + "-"*80)
    print("【合并统计】")
    total_samples = sum(r['num_samples'] for r in results)
    total_time = sum(r['total_time'] for r in results)
    avg_time = total_time / total_samples if total_samples > 0 else 0
    
    print(f"  四个数据集总样本数: {total_samples}")
    print(f"  四个数据集总评测时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  四个数据集平均每个样本时间: {avg_time:.2f} 秒")
    
    # 各数据集样本数量
    print("\n  各数据集样本数量:")
    for result in results:
        print(f"    {result['dataset_name']}: {result['num_samples']}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-7B VLLM 评测脚本')
    parser.add_argument('--model', type=str, default='Qwen3-VL-7B-Instruct',
                        help='模型名称')
    parser.add_argument('--datasets', nargs='+',
                        default=['MathVista_MINI', 'MathVision', 'MathVerse_MINI', 'LogicVista'],
                        help='要评测的数据集列表')
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8000',
                        help='VLLM API 地址')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='输出目录')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='最大生成 token 数')
    parser.add_argument('--top-p', type=float, default=0.8,
                        help='Top-p 采样')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top-k 采样')
    
    args = parser.parse_args()
    
    # 创建 VLLM 客户端
    print(f"连接 VLLM Server: {args.vllm_url}")
    vllm_client = VLLMClient(args.vllm_url, args.model)
    
    # 测试连接
    try:
        response = requests.get(f"{args.vllm_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ VLLM Server 连接成功")
        else:
            print(f"⚠ VLLM Server 返回状态码: {response.status_code}")
    except Exception as e:
        print(f"✗ 无法连接到 VLLM Server: {e}")
        print("请确保 VLLM Server 已启动")
        return
    
    # 执行评测
    results = []
    
    for dataset_name in args.datasets:
        try:
            result = evaluate_dataset(
                dataset_name=dataset_name,
                vllm_client=vllm_client,
                output_dir=args.output_dir,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k
            )
            results.append(result)
        except Exception as e:
            print(f"评测 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    print_summary(results)
    
    # 保存汇总结果到文件
    summary_data = []
    for result in results:
        summary_data.append({
            'dataset': result['dataset_name'],
            'num_samples': result['num_samples'],
            'total_time_sec': result['total_time'],
            'avg_time_per_sample_sec': result['avg_time_per_sample'],
        })
    
    # 添加汇总行
    total_samples = sum(r['num_samples'] for r in results)
    total_time = sum(r['total_time'] for r in results)
    summary_data.append({
        'dataset': 'TOTAL',
        'num_samples': total_samples,
        'total_time_sec': total_time,
        'avg_time_per_sample_sec': total_time / total_samples if total_samples > 0 else 0,
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(args.output_dir) / "summary.xlsx"
    dump(summary_df, str(summary_path))
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == '__main__':
    main()
