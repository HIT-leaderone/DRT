#!/usr/bin/env python3
"""
Qwen3-VL-7B VLLM 评测脚本 - 支持多种 Prompt 类型和评测 Setting
支持数据集: MathVista_MINI, MathVision, MathVerse_MINI, LogicVista

Prompt 类型:
- short-cot-image: Short-COT-Image (短 CoT 图片 prompt)
- short-cot-video: Short-COT-Video (短 CoT 视频 prompt)
- directly-answer: Directly-Answer (直接回答)
- custom-prompt: Custom-Prompt (使用 dataset 自带的 prompt)

评测 Setting:
- sequential: 串型评测，上一个问题的输出得到后再发下一个问题
- parallel: 并行评测，并行发送32个问题，将模型利用率打满

使用方法:
    python Efficiency-eval/legacy/eval_vllm_qwen3vl_full_legacy.py \
        --datasets MathVista_MINI MathVision MathVerse_MINI LogicVista \
        --prompt-type short-cot-image \
        --setting sequential \
        --vllm-url http://localhost:8000 \
        --output-dir ./results
"""

import argparse
import base64
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset.image_vqa import MathVista, MathVision, MathVerse, LogicVista
from vlmeval.smp import dump


@dataclass
class EvalResult:
    """评测结果数据结构"""
    dataset_name: str
    num_samples: int
    total_time: float  # 秒
    avg_time_per_sample: float  # 秒
    accuracy: float  # 百分比
    avg_output_length: float  # 平均输出长度（字符数）
    details: Dict[str, Any] = field(default_factory=dict)


class VLLMClient:
    """VLLM API 客户端"""
    
    def __init__(self, base_url: str, model_name: str = "Qwen3-VL-7B-Instruct"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self._lock = threading.Lock()
        
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
            "top_p": top_p,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5
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


class PromptBuilder:
    """Prompt 构建器 - 支持四种 prompt 类型"""
    
    # Short-COT-Image Prompt 模板
    SHORT_COT_IMAGE_TEMPLATE = (
        "{question}\n"
        "Analyze this question to provide a **Dense Cognitive Trace**.\n"
        "--- Requirements ---\n"
        "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols). NO conversational fillers.\n"
        "2. **Structure**: Your response MUST strictly follow this XML structure:\n"
        "   <visual>...concise visual evidence...</visual>  [思考过程]...[Priors] -> ...compressed reasoning... [思考结束] <answer>...final answer...</answer>\n"
        "3. **Content**: Deconstruct into visual observations, logical reasoning, and the final answer."
    )
    
    # Short-COT-Video Prompt 模板
    SHORT_COT_VIDEO_TEMPLATE = (
        "{question}\n\n"
        "Analyze this question with **maximum information density** and strict logical precision.\n"
        "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols).\n"
        "2. **Constraint**: STRICTLY FORBIDDEN to use conversational fillers (e.g., 'Let me think', 'Hmm', 'I see', 'Wait').\n"
        "3. **Content**: Focus only on: [Key Visual Evidence] -> [Logical Inference] -> [Conclusion].\n"
        "Provide your dense cognitive trace between the [思考过程] and [思考结束] tags, and then give your final answer."
    )
    
    # Directly-Answer Prompt 模板
    DIRECTLY_ANSWER_TEMPLATE = (
        "{question}\n"
        "Please answer concisely with short words or phrases when possible."
    )
    
    def __init__(self, prompt_type: str = "custom-prompt"):
        """
        初始化 PromptBuilder
        
        Args:
            prompt_type: prompt 类型，可选值:
                - "short-cot-image": Short-COT-Image
                - "short-cot-video": Short-COT-Video
                - "directly-answer": Directly-Answer
                - "custom-prompt": 使用 dataset 自带的 prompt
        """
        self.prompt_type = prompt_type.lower()
        
    def build_prompt(self, question: str, options: Dict[str, str] = None) -> str:
        """
        构建 prompt
        
        Args:
            question: 问题文本
            options: 选项字典（用于选择题）
            
        Returns:
            构建好的 prompt 文本
        """
        # 如果有选项，添加到问题中
        question_with_options = question
        if options:
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            question_with_options = f"{question}\nOptions:\n{options_text}"
        
        if self.prompt_type == "short-cot-image":
            return self.SHORT_COT_IMAGE_TEMPLATE.format(question=question_with_options)
        
        elif self.prompt_type == "short-cot-video":
            return self.SHORT_COT_VIDEO_TEMPLATE.format(question=question_with_options)
        
        elif self.prompt_type == "directly-answer":
            return self.DIRECTLY_ANSWER_TEMPLATE.format(question=question_with_options)
        
        elif self.prompt_type == "custom-prompt":
            # 返回原始问题（让 dataset 自带的 prompt 处理）
            return question_with_options
        
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")


class DatasetEvaluator:
    """数据集评测器 - 支持串型和并行两种评测模式"""
    
    def __init__(self, vllm_client: VLLMClient, output_dir: str, prompt_type: str = "custom-prompt", 
                 setting: str = "sequential", parallel_workers: int = 32):
        self.vllm_client = vllm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_builder = PromptBuilder(prompt_type)
        self.prompt_type = prompt_type
        self.setting = setting  # "sequential" 或 "parallel"
        self.parallel_workers = parallel_workers  # 并行模式的工作线程数
        
    def build_messages(self, image_path: str, question: str, options: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """构建 VLLM 消息格式"""
        image_base64 = self.vllm_client.encode_image_to_base64(image_path)
        
        # 使用 PromptBuilder 构建 prompt
        if self.prompt_type == "custom-prompt":
            # 使用原始问题，让 VLMEvalKit 的 dataset 处理 prompt
            prompt_text = question
        else:
            prompt_text = self.prompt_builder.build_prompt(question, options)
        
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
                        "text": prompt_text
                    }
                ]
            }
        ]
        return messages
    
    def _process_single_sample(self, args: Tuple[int, Any, Any, str]) -> Dict[str, Any]:
        """
        处理单个样本（用于并行模式）
        
        Args:
            args: (idx, row, dataset, dataset_name) 元组
            
        Returns:
            包含预测结果的字典
        """
        idx, row, dataset, dataset_name = args
        
        try:
            # 获取问题和图片
            question = row.get('question', '')
            if pd.isna(question):
                question = ''
            
            # 获取选项（如果有）
            options = None
            import string
            for cand in string.ascii_uppercase:
                if cand in row and not pd.isna(row[cand]):
                    if options is None:
                        options = {}
                    options[cand] = row[cand]
            
            # 处理图片
            image_path = self._get_image_path(dataset, row, dataset_name, idx)
            if image_path is None:
                return {
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', ''),
                    'output_length': 0,
                    'success': True
                }
            
            # 构建消息并调用 VLLM
            if self.prompt_type == "custom-prompt":
                msgs = dataset.build_prompt(row)
                question_text = None
                image_path_from_msg = None
                for msg in msgs:
                    if msg['type'] == 'text':
                        question_text = msg['value']
                    elif msg['type'] == 'image':
                        image_path_from_msg = msg['value']
                
                if question_text and image_path_from_msg:
                    messages = self.build_messages(image_path_from_msg, question_text, options)
                else:
                    messages = self._msgs_to_vllm_format(msgs)
            else:
                messages = self.build_messages(image_path, question, options)
            
            response = self.vllm_client.chat(messages)
            
            return {
                'index': row['index'],
                'prediction': response,
                'answer': row.get('answer', ''),
                'output_length': len(response),
                'success': True
            }
            
        except Exception as e:
            print(f"处理第 {idx} 个样本时出错: {e}")
            return {
                'index': row['index'],
                'prediction': '',
                'answer': row.get('answer', ''),
                'output_length': 0,
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_sequential(self, dataset_name: str, dataset, data: pd.DataFrame) -> Tuple[List[Dict], List[int], int]:
        """
        串型评测模式
        
        Returns:
            (predictions, output_lengths, num_samples)
        """
        num_samples = len(data)
        predictions = []
        output_lengths = []
        
        for idx in tqdm(range(num_samples), desc="评测进度(串型)"):
            row = data.iloc[idx]
            
            try:
                # 获取问题和图片
                question = row.get('question', '')
                if pd.isna(question):
                    question = ''
                
                # 获取选项（如果有）
                options = None
                import string
                for cand in string.ascii_uppercase:
                    if cand in row and not pd.isna(row[cand]):
                        if options is None:
                            options = {}
                        options[cand] = row[cand]
                
                # 处理图片
                image_path = self._get_image_path(dataset, row, dataset_name, idx)
                if image_path is None:
                    predictions.append({
                        'index': row['index'],
                        'prediction': '',
                        'answer': row.get('answer', '')
                    })
                    output_lengths.append(0)
                    continue
                
                # 构建消息并调用 VLLM
                if self.prompt_type == "custom-prompt":
                    msgs = dataset.build_prompt(row)
                    question_text = None
                    image_path_from_msg = None
                    for msg in msgs:
                        if msg['type'] == 'text':
                            question_text = msg['value']
                        elif msg['type'] == 'image':
                            image_path_from_msg = msg['value']
                    
                    if question_text and image_path_from_msg:
                        messages = self.build_messages(image_path_from_msg, question_text, options)
                    else:
                        messages = self._msgs_to_vllm_format(msgs)
                else:
                    messages = self.build_messages(image_path, question, options)
                
                response = self.vllm_client.chat(messages)
                
                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', '')
                })
                output_lengths.append(len(response))
                
            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
                output_lengths.append(0)
        
        return predictions, output_lengths, num_samples
    
    def _evaluate_parallel(self, dataset_name: str, dataset, data: pd.DataFrame) -> Tuple[List[Dict], List[int], int]:
        """
        并行评测模式 - 并行发送多个请求以打满模型利用率
        
        Returns:
            (predictions, output_lengths, num_samples)
        """
        num_samples = len(data)
        
        # 准备任务参数
        task_args = []
        for idx in range(num_samples):
            row = data.iloc[idx]
            task_args.append((idx, row, dataset, dataset_name))
        
        print(f"并行评测模式: 使用 {self.parallel_workers} 个并行工作线程")
        print(f"总样本数: {num_samples}, 批次数: {(num_samples + self.parallel_workers - 1) // self.parallel_workers}")
        
        # 存储结果
        results = [None] * num_samples
        output_lengths = [0] * num_samples
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(self._process_single_sample, args): args[0] 
                            for args in task_args}
            
            # 使用 tqdm 显示进度
            with tqdm(total=num_samples, desc="评测进度(并行)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                        if result.get('success', False):
                            output_lengths[idx] = result.get('output_length', 0)
                        else:
                            output_lengths[idx] = 0
                    except Exception as e:
                        print(f"处理第 {idx} 个样本时发生异常: {e}")
                        row = data.iloc[idx]
                        results[idx] = {
                            'index': row['index'],
                            'prediction': '',
                            'answer': row.get('answer', ''),
                            'output_length': 0,
                            'success': False,
                            'error': str(e)
                        }
                        output_lengths[idx] = 0
                    pbar.update(1)
        
        # 转换为 predictions 格式
        predictions = []
        for idx, result in enumerate(results):
            if result is not None:
                predictions.append({
                    'index': result['index'],
                    'prediction': result['prediction'],
                    'answer': result['answer']
                })
            else:
                # 如果结果丢失，添加空记录
                row = data.iloc[idx]
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', '')
                })
        
        return predictions, output_lengths, num_samples
    
    def evaluate_dataset(self, dataset_name: str, dataset_class) -> EvalResult:
        """通用数据集评测函数 - 支持串型和并行两种模式"""
        print("\n" + "="*60)
        print(f"评测 {dataset_name}")
        print(f"Prompt 类型: {self.prompt_type}")
        print(f"评测模式: {self.setting}")
        if self.setting == "parallel":
            print(f"并行工作线程数: {self.parallel_workers}")
        print("="*60)
        
        # 加载数据集
        dataset = dataset_class(dataset=dataset_name)
        data = dataset.data
        num_samples = len(data)
        
        print(f"样本数量: {num_samples}")
        
        # 根据 setting 选择评测模式
        start_time = time.time()
        
        if self.setting == "sequential":
            predictions, output_lengths, _ = self._evaluate_sequential(dataset_name, dataset, data)
        elif self.setting == "parallel":
            predictions, output_lengths, _ = self._evaluate_parallel(dataset_name, dataset, data)
        else:
            raise ValueError(f"Unknown setting: {self.setting}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算平均输出长度
        avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        
        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / f"{dataset_name}_{self.prompt_type}_{self.setting}_predictions.xlsx"
        dump(pred_df, str(pred_path))
        
        result = EvalResult(
            dataset_name=dataset_name,
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0,
            accuracy=0.0,  # 需要特殊评估方法
            avg_output_length=avg_output_length,
            details={"predictions_file": str(pred_path)}
        )
        
        return result
    
    def _get_image_path(self, dataset, row, dataset_name, idx):
        """获取图片路径"""
        try:
            # 尝试使用 dataset 的 dump_image 方法
            image_path = dataset.dump_image(row)
            if image_path:
                return image_path[0] if isinstance(image_path, list) else image_path
        except Exception as e:
            pass
        
        # 尝试从 image_path 字段获取
        if 'image_path' in row and not pd.isna(row['image_path']):
            return row['image_path']
        
        # 尝试从缓存目录构建路径
        import base64
        img_base64 = row.get('image', '')
        if pd.notna(img_base64) and img_base64:
            try:
                cache_dir = Path(self.output_dir) / "images" / dataset_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                img_path = cache_dir / f"{row['index']}.jpg"
                if not img_path.exists():
                    img_data = base64.b64decode(img_base64)
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                return str(img_path)
            except Exception as e:
                pass
        
        return None
    
    def _msgs_to_vllm_format(self, msgs):
        """将 VLMEvalKit 的 msgs 转换为 VLLM 格式"""
        content = []
        for msg in msgs:
            if msg['type'] == 'image':
                image_path = msg['value']
                if os.path.exists(image_path):
                    image_base64 = self.vllm_client.encode_image_to_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    })
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        
        return [{"role": "user", "content": content}]


def print_summary(results: List[EvalResult], prompt_type: str, setting: str):
    """打印评测结果汇总"""
    print("\n" + "="*80)
    print(f"评测结果汇总 (Prompt 类型: {prompt_type}, Setting: {setting})")
    print("="*80)
    
    # 单个数据集结果
    for result in results:
        print(f"\n【{result.dataset_name}】")
        print(f"  样本数量: {result.num_samples}")
        print(f"  总评测时间: {result.total_time:.2f} 秒")
        print(f"  平均每个样本时间: {result.avg_time_per_sample:.2f} 秒")
        print(f"  平均输出长度: {result.avg_output_length:.2f} 字符")
        if result.accuracy > 0:
            print(f"  准确率: {result.accuracy:.2f}%")
    
    # 合并统计
    print("\n" + "-"*80)
    print("【合并统计】")
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / total_samples if total_samples > 0 else 0
    avg_output_length_overall = sum(r.avg_output_length * r.num_samples for r in results) / total_samples if total_samples > 0 else 0
    
    print(f"  Prompt 类型: {prompt_type}")
    print(f"  Setting: {setting}")
    print(f"  四个数据集总样本数: {total_samples}")
    print(f"  四个数据集总评测时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  四个数据集平均每个样本时间: {avg_time:.2f} 秒")
    print(f"  四个数据集平均输出长度: {avg_output_length_overall:.2f} 字符")
    
    # 各数据集指标
    print("\n  各数据集指标:")
    for result in results:
        print(f"    {result.dataset_name}:")
        print(f"      样本数: {result.num_samples}")
        print(f"      平均输出长度: {result.avg_output_length:.2f} 字符")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Qwen3-VL-7B VLLM 评测脚本 - 支持多种 Prompt 类型和评测 Setting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt 类型说明:
  short-cot-image  - Short-COT-Image: 短 CoT 图片 prompt
  short-cot-video  - Short-COT-Video: 短 CoT 视频 prompt
  directly-answer  - Directly-Answer: 直接回答
  custom-prompt    - Custom-Prompt: 使用 dataset 自带的 prompt

Setting 类型说明:
  sequential  - 串型评测，上一个问题的输出得到后再发下一个问题
  parallel    - 并行评测，并行发送多个问题，将模型利用率打满

示例:
  # 串型评测 (默认)
  python Efficiency-eval/legacy/eval_vllm_qwen3vl_full_legacy.py --setting sequential

  # 并行评测，32个并行请求
  python Efficiency-eval/legacy/eval_vllm_qwen3vl_full_legacy.py --setting parallel --parallel-workers 32

  # 使用 Short-COT-Image + 并行评测
  python Efficiency-eval/legacy/eval_vllm_qwen3vl_full_legacy.py --prompt-type short-cot-image --setting parallel
        """
    )
    
    parser.add_argument('--model', type=str, default='Qwen3-VL-7B-Instruct',
                        help='模型名称')
    parser.add_argument('--datasets', nargs='+',
                        default=['MathVista_MINI', 'MathVision', 'MathVerse_MINI', 'LogicVista'],
                        help='要评测的数据集列表')
    parser.add_argument('--prompt-type', type=str, 
                        default='custom-prompt',
                        choices=['short-cot-image', 'short-cot-video', 'directly-answer', 'custom-prompt'],
                        help='Prompt 类型 (默认: custom-prompt)')
    parser.add_argument('--setting', type=str,
                        default='sequential',
                        choices=['sequential', 'parallel'],
                        help='评测 Setting (默认: sequential)')
    parser.add_argument('--parallel-workers', type=int, default=32,
                        help='并行模式的工作线程数 (默认: 32)')
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
    evaluator = DatasetEvaluator(
        vllm_client, 
        args.output_dir, 
        args.prompt_type, 
        args.setting,
        args.parallel_workers
    )
    
    # 数据集映射
    dataset_class_map = {
        'MathVista_MINI': MathVista,
        'MathVision': MathVision,
        'MathVerse_MINI': MathVerse,
        'LogicVista': LogicVista,
    }
    
    # 执行评测
    results = []
    
    for dataset_name in args.datasets:
        if dataset_name in dataset_class_map:
            try:
                result = evaluator.evaluate_dataset(dataset_name, dataset_class_map[dataset_name])
                results.append(result)
            except Exception as e:
                print(f"评测 {dataset_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"未知的数据集: {dataset_name}")
    
    # 打印汇总结果
    print_summary(results, args.prompt_type, args.setting)
    
    # 保存汇总结果到文件
    summary_data = []
    for result in results:
        summary_data.append({
            'dataset': result.dataset_name,
            'num_samples': result.num_samples,
            'total_time_sec': result.total_time,
            'avg_time_per_sample_sec': result.avg_time_per_sample,
            'avg_output_length_chars': result.avg_output_length,
            'accuracy': result.accuracy
        })
    
    # 添加汇总行
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_output_length_overall = sum(r.avg_output_length * r.num_samples for r in results) / total_samples if total_samples > 0 else 0
    
    summary_data.append({
        'dataset': 'TOTAL',
        'num_samples': total_samples,
        'total_time_sec': total_time,
        'avg_time_per_sample_sec': total_time / total_samples if total_samples > 0 else 0,
        'avg_output_length_chars': avg_output_length_overall,
        'accuracy': None
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(args.output_dir) / f"summary_{args.prompt_type}_{args.setting}.xlsx"
    dump(summary_df, str(summary_path))
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == '__main__':
    main()
