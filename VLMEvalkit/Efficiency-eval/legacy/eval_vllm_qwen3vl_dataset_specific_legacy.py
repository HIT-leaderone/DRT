#!/usr/bin/env python3
"""
Qwen3-VL-7B VLLM 评测脚本
支持数据集: MathVista_MINI, MathVision, MathVerse_MINI, LogicVista

使用方法:
    python Efficiency-eval/legacy/eval_vllm_qwen3vl_dataset_specific_legacy.py \
        --model Qwen3-VL-7B-Instruct \
        --model-path /path/to/Qwen3-VL-7B-Instruct \
        --datasets MathVista_MINI MathVision MathVerse_MINI LogicVista \
        --vllm-url http://localhost:8000 \
        --output-dir ./results
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset.image_vqa import MathVista, MathVision, MathVerse, LogicVista
from vlmeval.smp import load, dump


@dataclass
class EvalResult:
    """评测结果数据结构"""
    dataset_name: str
    num_samples: int
    total_time: float  # 秒
    avg_time_per_sample: float  # 秒
    accuracy: float  # 百分比
    avg_output_length: float  # 平均输出长度（token 数或字符数）
    details: Dict[str, Any] = field(default_factory=dict)


class VLLMClient:
    """VLLM API 客户端"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def encode_pil_image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 编码"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def chat(self, messages: List[Dict[str, Any]], 
             temperature: float = 0.7,
             max_tokens: int = 16384,
             top_p: float = 0.8,
             top_k: int = 20,
             repetition_penalty: float = 1.0,
             presence_penalty: float = 1.5) -> str:
        """发送聊天请求到 VLLM"""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty
        }
        
        # 添加 top_k (某些 VLLM 版本可能不支持)
        if top_k > 0:
            payload["top_k"] = top_k
            
        try:
            response = requests.post(
                self.chat_url, 
                headers=headers, 
                json=payload,
                timeout=300  # 5分钟超时
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"VLLM API 请求失败: {e}")
            raise


class DatasetEvaluator:
    """数据集评测器"""
    
    def __init__(self, vllm_client: VLLMClient, output_dir: str):
        self.vllm_client = vllm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_messages(self, image_path: str, question: str) -> List[Dict[str, Any]]:
        """构建 VLLM 消息格式"""
        # 将图片转换为 base64
        if os.path.exists(image_path):
            image_base64 = self.vllm_client.encode_image_to_base64(image_path)
        else:
            # 尝试从 PIL Image 加载
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
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
    
    def evaluate_mathvista_mini(self) -> EvalResult:
        """评测 MathVista_MINI 数据集"""
        print("\n" + "="*60)
        print("评测 MathVista_MINI")
        print("="*60)
        
        # 加载数据集
        dataset = MathVista(dataset='MathVista_MINI')
        data = dataset.data
        num_samples = len(data)
        
        print(f"样本数量: {num_samples}")
        
        # 存储预测结果和输出长度
        predictions = []
        output_lengths = []  # 记录每个输出的长度
        start_time = time.time()
        
        for idx in tqdm(range(num_samples), desc="评测进度"):
            row = data.iloc[idx]
            
            # 构建 prompt
            msgs = dataset.build_prompt(row)
            
            # 提取问题和图片路径
            question = None
            image_path = None
            
            for msg in msgs:
                if msg['type'] == 'text':
                    question = msg['value']
                elif msg['type'] == 'image':
                    image_path = msg['value']
            
            if question is None or image_path is None:
                print(f"警告: 第 {idx} 个样本缺少问题或图片")
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
                output_lengths.append(0)
                continue
            
            try:
                # 构建消息并调用 VLLM
                messages = self.build_messages(image_path, question)
                response = self.vllm_client.chat(messages)
                
                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', '')
                })
                output_lengths.append(len(response))  # 记录输出长度（字符数）
                
            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
                output_lengths.append(0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算平均输出长度
        avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        
        # 保存预测结果
        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / "MathVista_MINI_predictions.xlsx"
        dump(pred_df, str(pred_path))
        
        # 使用数据集的 evaluate 方法计算准确率
        # 创建一个包含 prediction 列的完整数据框
        eval_df = data.copy()
        eval_df['prediction'] = pred_df.set_index('index').loc[eval_df['index'], 'prediction'].values
        
        # 保存为临时文件用于 evaluate
        eval_path = self.output_dir / "MathVista_MINI_eval.xlsx"
        dump(eval_df, str(eval_path))
        
        # 调用 evaluate (需要配置 judge model)
        try:
            score = dataset.evaluate(str(eval_path))
            accuracy = score.get('Overall', 0.0) if isinstance(score, dict) else 0.0
        except Exception as e:
            print(f"评估时出错: {e}")
            accuracy = 0.0
        
        result = EvalResult(
            dataset_name="MathVista_MINI",
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            accuracy=accuracy,
            avg_output_length=avg_output_length,
            details={
                "predictions_file": str(pred_path),
                "eval_file": str(eval_path)
            }
        )
        
        return result
    
    def evaluate_mathvision(self) -> EvalResult:
        """评测 MathVision 数据集"""
        print("\n" + "="*60)
        print("评测 MathVision")
        print("="*60)
        
        # 加载数据集
        dataset = MathVision(dataset='MathVision')
        data = dataset.data
        num_samples = len(data)
        
        print(f"样本数量: {num_samples}")
        
        predictions = []
        output_lengths = []  # 记录每个输出的长度
        start_time = time.time()
        
        for idx in tqdm(range(num_samples), desc="评测进度"):
            row = data.iloc[idx]
            
            try:
                msgs = dataset.build_prompt(row)
                
                question = None
                image_path = None
                
                for msg in msgs:
                    if msg['type'] == 'text':
                        question = msg['value']
                    elif msg['type'] == 'image':
                        image_path = msg['value']
                
                if question is None or image_path is None:
                    predictions.append({
                        'index': row['index'],
                        'prediction': '',
                        'answer': row.get('answer', '')
                    })
                    output_lengths.append(0)
                    continue
                
                messages = self.build_messages(image_path, question)
                response = self.vllm_client.chat(messages)
                
                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', '')
                })
                output_lengths.append(len(response))  # 记录输出长度
                
            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
                output_lengths.append(0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算平均输出长度
        avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        
        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / "MathVision_predictions.xlsx"
        dump(pred_df, str(pred_path))
        
        result = EvalResult(
            dataset_name="MathVision",
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            accuracy=0.0,  # MathVision 需要特殊的评估方法
            avg_output_length=avg_output_length,
            details={"predictions_file": str(pred_path)}
        )
        
        return result
    
    def evaluate_mathverse_mini(self) -> EvalResult:
        """评测 MathVerse_MINI 数据集"""
        print("\n" + "="*60)
        print("评测 MathVerse_MINI")
        print("="*60)
        
        # 加载数据集
        dataset = MathVerse(dataset='MathVerse_MINI')
        data = dataset.data
        num_samples = len(data)
        
        print(f"样本数量: {num_samples}")
        
        predictions = []
        start_time = time.time()
        
        for idx in tqdm(range(num_samples), desc="评测进度"):
            row = data.iloc[idx]
            
            try:
                msgs = dataset.build_prompt(row)
                
                question = None
                image_path = None
                
                for msg in msgs:
                    if msg['type'] == 'text':
                        question = msg['value']
                    elif msg['type'] == 'image':
                        image_path = msg['value']
                
                if question is None or image_path is None:
                    predictions.append({
                        'index': row['index'],
                        'prediction': '',
                        'answer': row.get('answer', '')
                    })
                    continue
                
                messages = self.build_messages(image_path, question)
                response = self.vllm_client.chat(messages)
                
                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', '')
                })
                
            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / "MathVerse_MINI_predictions.xlsx"
        dump(pred_df, str(pred_path))
        
        result = EvalResult(
            dataset_name="MathVerse_MINI",
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            accuracy=0.0,  # MathVerse 需要特殊的评估方法
            details={"predictions_file": str(pred_path)}
        )
        
        return result
    
    def evaluate_logicvista(self) -> EvalResult:
        """评测 LogicVista 数据集"""
        print("\n" + "="*60)
        print("评测 LogicVista")
        print("="*60)
        
        # 加载数据集
        dataset = LogicVista(dataset='LogicVista')
        data = dataset.data
        num_samples = len(data)
        
        print(f"样本数量: {num_samples}")
        
        predictions = []
        start_time = time.time()
        
        for idx in tqdm(range(num_samples), desc="评测进度"):
            row = data.iloc[idx]
            
            try:
                msgs = dataset.build_prompt(row)
                
                question = None
                image_path = None
                
                for msg in msgs:
                    if msg['type'] == 'text':
                        question = msg['value']
                    elif msg['type'] == 'image':
                        image_path = msg['value']
                
                if question is None or image_path is None:
                    predictions.append({
                        'index': row['index'],
                        'prediction': '',
                        'answer': row.get('answer', '')
                    })
                    continue
                
                messages = self.build_messages(image_path, question)
                response = self.vllm_client.chat(messages)
                
                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', '')
                })
                
            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / "LogicVista_predictions.xlsx"
        dump(pred_df, str(pred_path))
        
        result = EvalResult(
            dataset_name="LogicVista",
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            accuracy=0.0,  # LogicVista 需要特殊的评估方法
            details={"predictions_file": str(pred_path)}
        )
        
        return result


def print_summary(results: List[EvalResult]):
    """打印评测结果汇总"""
    print("\n" + "="*80)
    print("评测结果汇总")
    print("="*80)
    
    # 单个数据集结果
    for result in results:
        print(f"\n【{result.dataset_name}】")
        print(f"  样本数量: {result.num_samples}")
        print(f"  总评测时间: {result.total_time:.2f} 秒")
        print(f"  平均每个样本时间: {result.avg_time_per_sample:.2f} 秒")
        if result.accuracy > 0:
            print(f"  准确率: {result.accuracy:.2f}%")
    
    # 合并统计
    print("\n" + "-"*80)
    print("【合并统计】")
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / total_samples if total_samples > 0 else 0
    
    print(f"  四个数据集总样本数: {total_samples}")
    print(f"  四个数据集总评测时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  四个数据集平均每个样本时间: {avg_time:.2f} 秒")
    
    # 各数据集样本数量
    print("\n  各数据集样本数量:")
    for result in results:
        print(f"    {result.dataset_name}: {result.num_samples}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-7B VLLM 评测脚本')
    parser.add_argument('--model', type=str, default='Qwen3-VL-7B-Instruct',
                        help='模型名称')
    parser.add_argument('--model-path', type=str, default='',
                        help='本地模型路径 (可选)')
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
    
    # 创建评测器
    evaluator = DatasetEvaluator(vllm_client, args.output_dir)
    
    # 执行评测
    results = []
    
    dataset_map = {
        'MathVista_MINI': evaluator.evaluate_mathvista_mini,
        'MathVision': evaluator.evaluate_mathvision,
        'MathVerse_MINI': evaluator.evaluate_mathverse_mini,
        'LogicVista': evaluator.evaluate_logicvista,
    }
    
    for dataset_name in args.datasets:
        if dataset_name in dataset_map:
            try:
                result = dataset_map[dataset_name]()
                results.append(result)
            except Exception as e:
                print(f"评测 {dataset_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"未知的数据集: {dataset_name}")
    
    # 打印汇总结果
    print_summary(results)
    
    # 保存汇总结果到文件
    summary_data = []
    for result in results:
        summary_data.append({
            'dataset': result.dataset_name,
            'num_samples': result.num_samples,
            'total_time_sec': result.total_time,
            'avg_time_per_sample_sec': result.avg_time_per_sample,
            'accuracy': result.accuracy
        })
    
    # 添加汇总行
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    summary_data.append({
        'dataset': 'TOTAL',
        'num_samples': total_samples,
        'total_time_sec': total_time,
        'avg_time_per_sample_sec': total_time / total_samples if total_samples > 0 else 0,
        'accuracy': None
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(args.output_dir) / "summary.xlsx"
    dump(summary_df, str(summary_path))
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == '__main__':
    main()
