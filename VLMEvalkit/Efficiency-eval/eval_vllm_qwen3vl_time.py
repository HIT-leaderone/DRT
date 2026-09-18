#!/usr/bin/env python3
"""
Qwen3-VL-7B VLLM 评测脚本 - 支持多种 Prompt 类型和评测 Setting
支持数据集: MathVista_MINI, MathVerse_MINI, LogicVista, GSM8K, Video_Holmes

Prompt 类型:
- short-cot-image: Short-COT-Image (短 CoT 图片 prompt)
- short-cot-video: Short-COT-Video (短 CoT 视频 prompt)
- directly-answer: Directly-Answer (直接回答)
- cod: CoD / Chain-of-Draft
- thinkless: ThinkLess-style concise post-regulated answer
- custom-prompt: Custom-Prompt (使用 dataset 自带的 prompt)

评测 Setting:
- sequential: 串行评测
- parallel: 并行评测（默认 32 并发线程）

新增指标:
- item_latency_sec: 每条样本一次 vLLM chat 调用耗时（秒）
- output_length_tokens: 使用本地 ./Qwen3-VL-8B-Instruct 的 tokenizer 统计输出 token 数
"""

import argparse
import base64
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset.image_vqa import MathVista, MathVerse, LogicVista
from vlmeval.dataset.text_math import MathReasoningDataset
from vlmeval.dataset.video_holmes import Video_Holmes
from vlmeval.smp import dump
from time_eval_prompt_utils import build_official_eval_message, set_official_prompt_env


@dataclass
class EvalResult:
    """评测结果数据结构"""
    dataset_name: str
    num_samples: int
    total_time: float  # 秒（整个数据集 wall time）
    avg_time_per_sample: float  # 秒（wall time / 样本数）
    avg_item_latency_sec: float  # 秒（仅 vllm chat 调用耗时均值；失败样本不计入）
    accuracy: float  # 百分比（此脚本仍为 0.0，占位）
    avg_output_tokens: float  # 平均输出长度（token 数）
    details: Dict[str, Any] = field(default_factory=dict)


class VLLMClient:
    """VLLM API 客户端"""

    def __init__(self, base_url: str, model_name: str = "Qwen3-VL-7B-Instruct"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 16384,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
    ) -> str:
        """发送聊天请求到 VLLM"""
        headers = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }

        # 兼容：有的服务端可能不支持 top_k；你原代码逻辑保留
        if top_k is not None and top_k > 0:
            payload["top_k"] = top_k

        response = None
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=30000
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            detail = ""
            if response is not None:
                try:
                    detail = response.text[:1000]
                except Exception:
                    detail = ""
            if detail:
                print(f"VLLM API 请求失败: {e}\n响应内容: {detail}")
            else:
                print(f"VLLM API 请求失败: {e}")
            raise

    def list_models(self) -> List[str]:
        response = requests.get(self.models_url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = payload.get('data', []) if isinstance(payload, dict) else []
        models = []
        for item in data:
            model_id = item.get('id') if isinstance(item, dict) else None
            if model_id:
                models.append(str(model_id))
        return models


class PromptBuilder:
    """Prompt 构建器 - 支持多种 prompt 类型"""

    SHORT_COT_IMAGE_TEMPLATE = (
        "{question}\n"
        "Analyze this question to provide a **Dense Cognitive Trace**.\n"
        "--- Requirements ---\n"
        "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols). NO conversational fillers.\n"
        "2. **Structure**: Your response MUST strictly follow this XML structure:\n"
        "   <visual>...concise visual evidence...</visual>  [思考过程]...[Priors] -> ...compressed reasoning... [思考结束] <answer>...final answer...</answer>\n"
        "3. **Content**: Deconstruct into visual observations, logical reasoning, and the final answer."
    )

    SHORT_COT_VIDEO_TEMPLATE = (
        "{question}\n\n"
        "Analyze this question with **maximum information density** and strict logical precision.\n"
        "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols).\n"
        "2. **Constraint**: STRICTLY FORBIDDEN to use conversational fillers (e.g., 'Let me think', 'Hmm', 'I see', 'Wait').\n"
        "3. **Content**: Focus only on: [Key Visual Evidence] -> [Logical Inference] -> [Conclusion].\n"
        "Provide your dense cognitive trace between the [思考过程] and [思考结束] tags, and then give your final answer."
    )

    DIRECTLY_ANSWER_TEMPLATE = (
        "{question}\n"
        "Please answer concisely with short words or phrases when possible."
    )

    COD_TEMPLATE = (
        "{question}\n"
        "Think step by step, but keep only a minimum draft for each step with at most 5 words.\n"
        "Give only the final answer at the end."
    )

    THINKLESS_TEMPLATE = (
        "{question}\n"
        "Think carefully but avoid redundant reasoning. "
        "Condense the reasoning as much as possible and provide only the final answer."
    )

    def __init__(self, prompt_type: str = "custom-prompt"):
        self.prompt_type = prompt_type.lower()

    def build_prompt(self, question: str, options: Dict[str, str] = None) -> str:
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
        elif self.prompt_type == "cod":
            return self.COD_TEMPLATE.format(question=question_with_options)
        elif self.prompt_type == "thinkless":
            return self.THINKLESS_TEMPLATE.format(question=question_with_options)
        elif self.prompt_type == "custom-prompt":
            return question_with_options
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")


class DatasetEvaluator:
    """数据集评测器 - 支持串行和并行两种评测模式；保存生成参数并下发到 vLLM"""

    def __init__(
        self,
        vllm_client: VLLMClient,
        output_dir: str,
        prompt_type: str = "custom-prompt",
        setting: str = "sequential",
        parallel_workers: int = 32,
        tokenizer_path: str = "./Qwen3-VL-8B-Instruct",
        # 生成参数（会保存在 evaluator 中）
        temperature: float = 0.7,
        max_tokens: int = 16384,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
    ):
        self.vllm_client = vllm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_builder = PromptBuilder(prompt_type)
        self.prompt_type = prompt_type
        set_official_prompt_env(None if prompt_type == "thinkless" else prompt_type)

        self.setting = setting  # "sequential" 或 "parallel"
        self.parallel_workers = parallel_workers

        # 保存生成参数：后续串行/并行统一下发
        self.gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )

        # tokenizer：用于把输出长度从 chars 变为 tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

    def build_messages(self, image_path: Optional[str], question: str, options: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """构建 VLLM 消息格式"""
        if self.prompt_type == "custom-prompt":
            prompt_text = question
        else:
            prompt_text = self.prompt_builder.build_prompt(question, options)

        content: List[Dict[str, Any]] = []
        if image_path is not None:
            image_base64 = self.vllm_client.encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            )
        content.append({"type": "text", "text": prompt_text})

        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def _build_eval_messages(self, dataset, row, dataset_name: str) -> List[Dict[str, Any]]:
        question = row.get('question', '')
        if pd.isna(question):
            question = ''

        options = None
        import string
        for cand in string.ascii_uppercase:
            if cand in row and not pd.isna(row[cand]):
                if options is None:
                    options = {}
                options[cand] = row[cand]

        if self.prompt_type == "thinkless":
            image_path = self._get_image_path(dataset, row, dataset_name, row.get('index', 0))
            return self.build_messages(image_path, question, options)

        official_message = build_official_eval_message(
            dataset,
            dataset_name,
            row,
            video_llm=True,
        )
        return self._msgs_to_vllm_format(official_message)

    def _infer_one(self, messages: List[Dict[str, Any]]) -> Tuple[str, float]:
        """一次 vLLM 推理：返回 (response_text, latency_sec)"""
        t0 = time.perf_counter()
        resp = self.vllm_client.chat(messages, **self.gen_kwargs)
        latency = time.perf_counter() - t0
        return resp, latency

    def _process_single_sample(self, args: Tuple[int, Any, Any, str]) -> Dict[str, Any]:
        """处理单个样本（用于并行模式）"""
        idx, row, dataset, dataset_name = args

        try:
            messages = self._build_eval_messages(dataset, row, dataset_name)

            response, latency = self._infer_one(messages)
            tok_len = len(self.tokenizer.encode(response, add_special_tokens=False))

            return {
                'index': row['index'],
                'prediction': response,
                'answer': row.get('answer', ''),
                'item_latency_sec': latency,
                'output_length_tokens': tok_len,
                'success': True
            }

        except Exception as e:
            print(f"处理第 {idx} 个样本时出错: {e}")
            return {
                'index': row.get('index', idx),
                'prediction': '',
                'answer': row.get('answer', ''),
                'item_latency_sec': None,
                'output_length_tokens': 0,
                'success': False,
                'error': str(e)
            }

    def _evaluate_sequential(self, dataset_name: str, dataset, data: pd.DataFrame) -> Tuple[List[Dict], List[int], List[Optional[float]], int]:
        """串行评测模式"""
        num_samples = len(data)
        predictions: List[Dict[str, Any]] = []
        output_token_lengths: List[int] = []
        item_latencies: List[Optional[float]] = []

        for idx in tqdm(range(num_samples), desc="评测进度(串行)"):
            row = data.iloc[idx]
            try:
                messages = self._build_eval_messages(dataset, row, dataset_name)

                response, latency = self._infer_one(messages)
                tok_len = len(self.tokenizer.encode(response, add_special_tokens=False))

                predictions.append({
                    'index': row['index'],
                    'prediction': response,
                    'answer': row.get('answer', ''),
                    'item_latency_sec': latency,
                    'output_length_tokens': tok_len
                })
                output_token_lengths.append(tok_len)
                item_latencies.append(latency)

            except Exception as e:
                print(f"处理第 {idx} 个样本时出错: {e}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    'index': row.get('index', idx),
                    'prediction': '',
                    'answer': row.get('answer', ''),
                    'item_latency_sec': None,
                    'output_length_tokens': 0
                })
                output_token_lengths.append(0)
                item_latencies.append(None)

        return predictions, output_token_lengths, item_latencies, num_samples

    def _evaluate_parallel(self, dataset_name: str, dataset, data: pd.DataFrame) -> Tuple[List[Dict], List[int], List[Optional[float]], int]:
        """并行评测模式"""
        num_samples = len(data)

        task_args = []
        for idx in range(num_samples):
            row = data.iloc[idx]
            task_args.append((idx, row, dataset, dataset_name))

        print(f"并行评测模式: 使用 {self.parallel_workers} 个并发线程")
        print(f"总样本数: {num_samples}, 批次数: {(num_samples + self.parallel_workers - 1) // self.parallel_workers}")

        results: List[Optional[Dict[str, Any]]] = [None] * num_samples
        output_token_lengths: List[int] = [0] * num_samples
        item_latencies: List[Optional[float]] = [None] * num_samples

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_sample, args): args[0]
                for args in task_args
            }

            with tqdm(total=num_samples, desc="评测进度(并行)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                        if result.get('success', False):
                            output_token_lengths[idx] = result.get('output_length_tokens', 0)
                            item_latencies[idx] = result.get('item_latency_sec', None)
                        else:
                            output_token_lengths[idx] = 0
                            item_latencies[idx] = None
                    except Exception as e:
                        print(f"处理第 {idx} 个样本时发生异常: {e}")
                        row = data.iloc[idx]
                        results[idx] = {
                            'index': row.get('index', idx),
                            'prediction': '',
                            'answer': row.get('answer', ''),
                            'item_latency_sec': None,
                            'output_length_tokens': 0,
                            'success': False,
                            'error': str(e)
                        }
                        output_token_lengths[idx] = 0
                        item_latencies[idx] = None
                    pbar.update(1)

        predictions: List[Dict[str, Any]] = []
        for idx, result in enumerate(results):
            if result is not None:
                predictions.append({
                    'index': result['index'],
                    'prediction': result['prediction'],
                    'answer': result['answer'],
                    'item_latency_sec': result.get('item_latency_sec', None),
                    'output_length_tokens': result.get('output_length_tokens', 0),
                })
            else:
                row = data.iloc[idx]
                predictions.append({
                    'index': row['index'],
                    'prediction': '',
                    'answer': row.get('answer', ''),
                    'item_latency_sec': None,
                    'output_length_tokens': 0,
                })

        return predictions, output_token_lengths, item_latencies, num_samples

    def evaluate_dataset(self, dataset_name: str, dataset_class) -> EvalResult:
        """通用数据集评测函数 - 支持串行和并行两种模式"""
        print("\n" + "=" * 60)
        print(f"评测 {dataset_name}")
        print(f"Prompt 类型: {self.prompt_type}")
        print(f"评测模式: {self.setting}")
        if self.setting == "parallel":
            print(f"并发线程数: {self.parallel_workers}")
        print(f"生成参数: {self.gen_kwargs}")
        print("=" * 60)

        dataset = dataset_class(dataset=dataset_name)
        data = dataset.data
        num_samples = len(data)
        print(f"样本数量: {num_samples}")

        start_time = time.time()

        if self.setting == "sequential":
            predictions, output_token_lengths, item_latencies, _ = self._evaluate_sequential(dataset_name, dataset, data)
        elif self.setting == "parallel":
            predictions, output_token_lengths, item_latencies, _ = self._evaluate_parallel(dataset_name, dataset, data)
        else:
            raise ValueError(f"Unknown setting: {self.setting}")

        total_time = time.time() - start_time

        avg_output_tokens = (sum(output_token_lengths) / len(output_token_lengths)) if output_token_lengths else 0.0
        valid_lat = [x for x in item_latencies if isinstance(x, (int, float))]
        avg_item_latency = (sum(valid_lat) / len(valid_lat)) if valid_lat else 0.0

        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / f"{dataset_name}_{self.prompt_type}_{self.setting}_predictions.xlsx"
        dump(pred_df, str(pred_path))

        return EvalResult(
            dataset_name=dataset_name,
            num_samples=num_samples,
            total_time=total_time,
            avg_time_per_sample=total_time / num_samples if num_samples > 0 else 0.0,
            avg_item_latency_sec=avg_item_latency,
            accuracy=0.0,
            avg_output_tokens=avg_output_tokens,
            details={"predictions_file": str(pred_path)}
        )

    def _get_image_path(self, dataset, row, dataset_name, idx):
        """获取图片路径"""
        try:
            image_path = dataset.dump_image(row)
            if image_path:
                return image_path[0] if isinstance(image_path, list) else image_path
        except Exception:
            pass

        if 'image_path' in row and not pd.isna(row['image_path']):
            return row['image_path']

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
            except Exception:
                pass

        return None

    def _msgs_to_vllm_format(self, msgs):
        """将 VLMEvalKit 的 msgs 转换为 VLLM 格式"""
        content = []
        for msg in msgs:
            if msg['type'] == 'image':
                image_values = msg['value'] if isinstance(msg['value'], list) else [msg['value']]
                for image_path in image_values:
                    if os.path.exists(image_path):
                        image_base64 = self.vllm_client.encode_image_to_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
            elif msg['type'] == 'video':
                video_value = msg['value']
                if isinstance(video_value, list):
                    for frame_path in video_value:
                        if os.path.exists(frame_path) and frame_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            image_base64 = self.vllm_client.encode_image_to_base64(frame_path)
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            })
                elif os.path.exists(video_value):
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": f"file://{video_value}"}
                    })
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})

        return [{"role": "user", "content": content}]


def print_summary(results: List[EvalResult], prompt_type: str, setting: str):
    """打印评测结果汇总"""
    print("\n" + "=" * 80)
    print(f"评测结果汇总 (Prompt 类型: {prompt_type}, Setting: {setting})")
    print("=" * 80)

    for result in results:
        print(f"\n【{result.dataset_name}】")
        print(f"  样本数量: {result.num_samples}")
        print(f"  总评测时间(wall): {result.total_time:.2f} 秒")
        print(f"  平均每样本时间(wall): {result.avg_time_per_sample:.2f} 秒")
        print(f"  平均 item latency(vllm chat): {result.avg_item_latency_sec:.3f} 秒")
        print(f"  平均输出长度: {result.avg_output_tokens:.2f} tokens")
        if result.accuracy > 0:
            print(f"  准确率: {result.accuracy:.2f}%")

    print("\n" + "-" * 80)
    print("【合并统计】")
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / total_samples if total_samples > 0 else 0.0

    avg_output_tokens_overall = (
        sum(r.avg_output_tokens * r.num_samples for r in results) / total_samples
        if total_samples > 0 else 0.0
    )
    avg_item_latency_overall = (
        sum(r.avg_item_latency_sec * r.num_samples for r in results) / total_samples
        if total_samples > 0 else 0.0
    )

    print(f"  Prompt 类型: {prompt_type}")
    print(f"  Setting: {setting}")
    print(f"  全部数据集总样本数: {total_samples}")
    print(f"  全部数据集总评测时间(wall): {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    print(f"  全部数据集平均每样本时间(wall): {avg_time:.2f} 秒")
    print(f"  全部数据集平均 item latency(加权均值): {avg_item_latency_overall:.3f} 秒")
    print(f"  全部数据集平均输出长度: {avg_output_tokens_overall:.2f} tokens")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Qwen3-VL-7B VLLM 评测脚本 - 支持多种 Prompt 类型和评测 Setting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, default='Qwen3-VL-7B-Instruct', help='模型名称')
    parser.add_argument('--datasets', nargs='+',
                        default=['MathVista_MINI', 'MathVerse_MINI', 'LogicVista', 'Video_Holmes', 'GSM8K'],
                        help='要评测的数据集列表')
    parser.add_argument('--prompt-type', type=str, default='custom-prompt',
                        choices=['short-cot-image', 'short-cot-video', 'directly-answer', 'cod', 'thinkless', 'custom-prompt'],
                        help='Prompt 类型 (默认: custom-prompt)')

    parser.add_argument('--setting', type=str, default='sequential',
                        choices=['sequential', 'parallel'],
                        help='评测 Setting (默认: sequential)')
    parser.add_argument('--parallel-workers', type=int, default=32,
                        help='并行模式的线程数 (默认: 32)')

    parser.add_argument('--vllm-url', type=str, default='http://localhost:8000', help='VLLM API 地址')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')

    # 生成参数（会真正下发到 vLLM）
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--max-tokens', type=int, default=16384, help='最大生成 token 数')
    parser.add_argument('--top-p', type=float, default=0.8, help='Top-p 采样')
    parser.add_argument('--top-k', type=int, default=20, help='Top-k 采样')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='repetition_penalty')
    parser.add_argument('--presence-penalty', type=float, default=1.5, help='presence_penalty')

    # tokenizer（用于 output_length_tokens）
    parser.add_argument('--tokenizer-path', type=str, default='./Qwen3-VL-8B-Instruct',
                        help='用于统计输出 token 长度的 tokenizer 路径')

    args = parser.parse_args()

    print(f"连接 VLLM Server: {args.vllm_url}")
    vllm_client = VLLMClient(args.vllm_url, args.model)

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

    try:
        served_models = vllm_client.list_models()
        if served_models:
            print(f"服务端注册模型: {served_models}")
            if args.model not in served_models:
                if len(served_models) == 1:
                    print(f"⚠ 请求模型名 {args.model} 不在服务端注册列表中，自动切换为 {served_models[0]}")
                    vllm_client.model_name = served_models[0]
                else:
                    print(f"⚠ 请求模型名 {args.model} 不在服务端注册列表中，请检查 served model name")
                    return
        else:
            print("⚠ /v1/models 返回空列表，继续使用请求中的模型名")
    except Exception as e:
        print(f"⚠ 获取 /v1/models 失败，将继续使用请求中的模型名: {e}")

    evaluator = DatasetEvaluator(
        vllm_client=vllm_client,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        setting=args.setting,
        parallel_workers=args.parallel_workers,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    dataset_class_map = {
        'MathVista_MINI': MathVista,
        'MathVerse_MINI': MathVerse,
        'LogicVista': LogicVista,
        'Video_Holmes': Video_Holmes,
        'GSM8K': MathReasoningDataset,
    }

    results: List[EvalResult] = []

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

    print_summary(results, args.prompt_type, args.setting)

    # 保存汇总结果到文件
    summary_data = []
    for r in results:
        summary_data.append({
            'dataset': r.dataset_name,
            'num_samples': r.num_samples,
            'total_time_sec_wall': r.total_time,
            'avg_time_per_sample_sec_wall': r.avg_time_per_sample,
            'avg_item_latency_sec': r.avg_item_latency_sec,
            'avg_output_length_tokens': r.avg_output_tokens,
            'accuracy': r.accuracy,
        })

    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_output_tokens_overall = (
        sum(r.avg_output_tokens * r.num_samples for r in results) / total_samples
        if total_samples > 0 else 0.0
    )
    avg_item_latency_overall = (
        sum(r.avg_item_latency_sec * r.num_samples for r in results) / total_samples
        if total_samples > 0 else 0.0
    )

    summary_data.append({
        'dataset': 'TOTAL',
        'num_samples': total_samples,
        'total_time_sec_wall': total_time,
        'avg_time_per_sample_sec_wall': (total_time / total_samples) if total_samples > 0 else 0.0,
        'avg_item_latency_sec': avg_item_latency_overall,
        'avg_output_length_tokens': avg_output_tokens_overall,
        'accuracy': None,
    })

    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(args.output_dir) / f"summary_{args.prompt_type}_{args.setting}.xlsx"
    dump(summary_df, str(summary_path))
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == '__main__':
    main()
