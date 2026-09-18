#!/usr/bin/env python3
"""
Transformers-based Thinkless time eval for Qwen3-VL models.

This runner is designed for fairer latency comparison against non-vLLM methods.
It uses a local HF model directly and executes batched generation on a single GPU.

Saved per-sample fields:
- item_latency_sec: approximated as batch wall time / batch size
- output_length_tokens: output token count from local tokenizer
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset.image_vqa import LogicVista, MathVerse, MathVista
from vlmeval.dataset.text_math import MathReasoningDataset
from vlmeval.dataset.video_holmes import Video_Holmes
from vlmeval.smp import dump
from time_eval_prompt_utils import (
    build_official_eval_message,
    set_official_prompt_env,
    to_processor_conversation,
)


@dataclass
class EvalResult:
    dataset_name: str
    num_samples: int
    total_time: float
    avg_time_per_sample: float
    avg_item_latency_sec: float
    accuracy: float
    avg_output_tokens: float
    details: Dict[str, Any] = field(default_factory=dict)


def sanitize_colon_env_var(name: str) -> None:
    raw = os.environ.get(name, "")
    if not raw:
        return
    cleaned = []
    for part in raw.split(":"):
        if not part:
            continue
        try:
            if os.path.isdir(part) and os.access(part, os.X_OK):
                cleaned.append(part)
        except OSError:
            continue
    os.environ[name] = ":".join(cleaned)


TAG_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue enclosed within <think> </think> "
    "and then provide the final answer."
)


def task_instruction(dataset_name: str, row) -> str:
    dataset_type = DATASET_TYPE(dataset_name, default=None)

    if dataset_name == "GSM8K":
        return "Solve the math problem step by step. Give only the final numerical answer."

    if dataset_name in {"LogicVista", "Video_Holmes"} or dataset_type == "MCQ":
        options = []
        if "candidates" in row and pd.notna(row["candidates"]):
            options = re.findall(r"([A-Z])\.", str(row["candidates"]))
        if not options:
            options = re.findall(r"([A-Z])[:.]", str(row.to_dict()))
        option_hint = ", ".join(sorted(set(options))) if options else "A, B, C, D"
        return (
            "Given the multiple-choice question above, think step by step, self-check your reasoning, "
            f"and output only the single final option ({option_hint})."
        )

    if dataset_name in {"MathVista_MINI", "MathVerse_MINI"}:
        if "choices" in row and pd.notna(row.get("choices", None)):
            return (
                "Solve the visual mathematical multiple-choice question above. "
                "Think step by step, self-check your reasoning, and output only the single final option letter."
            )
        hint = str(row.get("hint", "")).strip()
        if hint:
            return f"Solve the visual mathematical problem above. {hint} Give only the final answer."
        return "Solve the visual mathematical problem above. Give only the final answer."

    if dataset_type == "Y/N":
        return "Think step by step and output only Yes or No."

    return "Think step by step and give only the final answer."


class ThinklessBatchEvaluator:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        output_dir: str,
        batch_size: int = 32,
        token_budget: int = 300,
    ) -> None:
        sanitize_colon_env_var("PATH")
        sanitize_colon_env_var("LD_LIBRARY_PATH")
        set_official_prompt_env(None)

        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
            attn_implementation='flash_attention_2',
        )
        self.model.eval()

    def _build_sample(self, dataset, row, dataset_name: str, idx: int):
        source_message = build_official_eval_message(
            dataset,
            dataset_name,
            row,
            video_llm=True,
        )
        conversation = to_processor_conversation(source_message, system_prompt=TAG_SYSTEM_PROMPT)
        regulation_prompt = task_instruction(dataset_name, row)
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        text += f"<think></think>\n{regulation_prompt}\n"
        return {
            'index': row['index'],
            'answer': row.get('answer', ''),
            'conversation': conversation,
            'text': text,
        }

    def _generate_batch(self, batch_samples):
        from qwen_vl_utils import process_vision_info

        texts = [sample['text'] for sample in batch_samples]
        conversations = [sample['conversation'] for sample in batch_samples]
        images, videos, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors='pt',
            padding=True,
            **(video_kwargs or {}),
        )
        inputs = inputs.to(self.model.device)
        if hasattr(self.model, 'dtype'):
            inputs = inputs.to(self.model.dtype)

        start = time.perf_counter()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.token_budget,
            do_sample=False,
            repetition_penalty=1.0,
        )
        batch_latency = time.perf_counter() - start

        input_lengths = [len(x) for x in inputs.input_ids]
        trimmed_ids = [out_ids[in_len:] for out_ids, in_len in zip(generated_ids, input_lengths)]
        outputs = self.processor.tokenizer.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        per_item_latency = batch_latency / len(batch_samples) if batch_samples else 0.0
        return outputs, per_item_latency

    def evaluate_dataset(self, dataset_name: str, dataset_class, limit: Optional[int] = None) -> EvalResult:
        dataset = dataset_class(dataset=dataset_name)
        data = dataset.data
        if limit is not None:
            data = data.iloc[:limit].copy()
        num_samples = len(data)

        all_predictions = []
        output_token_lengths = []
        item_latencies = []
        start_time = time.time()

        prepared = []
        for idx in range(num_samples):
            row = data.iloc[idx]
            sample = self._build_sample(dataset, row, dataset_name, idx)
            if sample is not None:
                prepared.append(sample)

        for start_idx in tqdm(range(0, len(prepared), self.batch_size), desc=f"Thinkless {dataset_name}"):
            batch_samples = prepared[start_idx:start_idx + self.batch_size]
            outputs, per_item_latency = self._generate_batch(batch_samples)
            for sample, output in zip(batch_samples, outputs):
                tok_len = len(self.tokenizer.encode(output, add_special_tokens=False))
                all_predictions.append(
                    {
                        'index': sample['index'],
                        'prediction': output,
                        'answer': sample['answer'],
                        'item_latency_sec': per_item_latency,
                        'output_length_tokens': tok_len,
                    }
                )
                output_token_lengths.append(tok_len)
                item_latencies.append(per_item_latency)

        total_time = time.time() - start_time
        avg_output_tokens = (sum(output_token_lengths) / len(output_token_lengths)) if output_token_lengths else 0.0
        avg_item_latency = (sum(item_latencies) / len(item_latencies)) if item_latencies else 0.0

        pred_df = pd.DataFrame(all_predictions)
        pred_path = self.output_dir / f"{dataset_name}_thinkless_parallel_predictions.xlsx"
        dump(pred_df, str(pred_path))

        return EvalResult(
            dataset_name=dataset_name,
            num_samples=len(all_predictions),
            total_time=total_time,
            avg_time_per_sample=total_time / len(all_predictions) if all_predictions else 0.0,
            avg_item_latency_sec=avg_item_latency,
            accuracy=0.0,
            avg_output_tokens=avg_output_tokens,
            details={"predictions_file": str(pred_path)},
        )


def print_summary(results: List[EvalResult]):
    print("=" * 80)
    print("Thinkless Transformers Time Eval Summary")
    print("=" * 80)
    for result in results:
        print(result)


def main():
    parser = argparse.ArgumentParser(description='Transformers-based Thinkless time eval for Qwen3-VL.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--datasets', nargs='+', default=['MathVista_MINI', 'MathVerse_MINI', 'LogicVista', 'Video_Holmes', 'GSM8K'])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--parallel-workers', type=int, default=32, help='Mapped to batch size for transformers.')
    parser.add_argument('--token-budget', type=int, default=300)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    evaluator = ThinklessBatchEvaluator(
        model_name=args.model,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        batch_size=args.parallel_workers,
        token_budget=args.token_budget,
    )

    dataset_class_map = {
        'MathVista_MINI': MathVista,
        'MathVerse_MINI': MathVerse,
        'LogicVista': LogicVista,
        'Video_Holmes': Video_Holmes,
        'GSM8K': MathReasoningDataset,
    }
    results = []
    for dataset_name in args.datasets:
        if dataset_name in dataset_class_map:
            results.append(evaluator.evaluate_dataset(dataset_name, dataset_class_map[dataset_name], limit=args.limit))

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
    summary_path = Path(args.output_dir) / "summary_thinkless_parallel.xlsx"
    dump(summary_df, str(summary_path))
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
