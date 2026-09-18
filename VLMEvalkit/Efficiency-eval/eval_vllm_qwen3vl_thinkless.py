#!/usr/bin/env python3
"""
vLLM-based Thinkless time eval for Qwen3-VL models.

This runner keeps the vLLM server untouched and implements a two-stage
Thinkless workflow at the runner layer:
1. Build the official multimodal evaluation prompt.
2. Ask the model to continue an assistant prefix "<think>" for a fixed
   `think_budget`.
3. Force-close reasoning with "</think>" and append a lightweight
   regulation prompt.
4. Continue generation to obtain the final answer under `answer_budget`.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset.image_vqa import LogicVista, MathVerse, MathVista
from vlmeval.dataset.text_math import MathReasoningDataset
from vlmeval.dataset.video_holmes import Video_Holmes
from vlmeval.smp import dump
from time_eval_prompt_utils import build_official_eval_message, set_official_prompt_env


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


TAG_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue enclosed within <think> </think> "
    "and then provide the final answer."
)


def _nonempty_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return text not in {"", "[]", "None", "nan", "NaN"}


def _clean_think_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text)
    for marker in ["</think>", "<answer>", "</answer>"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def task_instruction(dataset_name: str, row) -> str:
    dataset_type = DATASET_TYPE(dataset_name, default=None)

    if dataset_name == "GSM8K":
        return (
            "Stop reasoning now. Output only the final numerical answer. "
            "Do not include any explanation, units, labels, or extra words.\n"
            "Final answer: "
        )

    if dataset_name in {"LogicVista", "Video_Holmes"} or dataset_type == "MCQ":
        options = []
        if "candidates" in row and pd.notna(row["candidates"]):
            options = re.findall(r"([A-Z])\.", str(row["candidates"]))
        if not options:
            options = re.findall(r"([A-Z])[:.]", str(row.to_dict()))
        option_hint = ", ".join(sorted(set(options))) if options else "A, B, C, D"
        return (
            "Stop reasoning now. "
            f"Output only the final option letters chosen from: {option_hint}. "
            "If there is a single correct option, output exactly one letter. "
            "If multiple options are correct, output only the letters in alphabetical order separated by commas, "
            "for example: A, C. "
            "Do not output any explanation, words, XML tags, or extra punctuation.\n"
            "Final answer: "
        )

    if dataset_name in {"MathVista_MINI", "MathVerse_MINI"}:
        is_multichoice = False
        if _nonempty_field(row.get("choices", None)):
            is_multichoice = True
        if str(row.get("question_type", "")).strip().lower() in {"multi_choice", "multi-choice"}:
            is_multichoice = True
        if _nonempty_field(row.get("answer_option", None)):
            is_multichoice = True

        if is_multichoice:
            return (
                "Stop reasoning now. "
                "Output only the single final option letter. "
                "Do not output the solution steps, equations, words, or any explanation.\n"
                "Final answer: "
            )
        answer_type = str(row.get("answer_type", "")).strip().lower()
        if answer_type == "integer":
            return (
                "Stop reasoning now. Output only the final integer answer. "
                "Do not include any explanation, words, or extra symbols.\n"
                "Final answer: "
            )
        if answer_type == "float":
            return (
                "Stop reasoning now. Output only the final numeric answer. "
                "Follow the required decimal precision exactly if the question specifies one. "
                "Do not include any explanation, words, or extra symbols.\n"
                "Final answer: "
            )
        return (
            "Stop reasoning now. Output only the final answer in the shortest correct form. "
            "Do not include any explanation or extra words.\n"
            "Final answer: "
        )

    if dataset_type == "Y/N":
        return (
            "Stop reasoning now. Output only Yes or No. "
            "Do not include any explanation or extra words.\n"
            "Final answer: "
        )

    return (
        "Stop reasoning now. Output only the final answer and nothing else.\n"
        "Final answer: "
    )


class VLLMClient:
    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def list_models(self) -> List[str]:
        response = requests.get(self.models_url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
        top_p: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "continue_final_message": continue_final_message,
            "add_generation_prompt": add_generation_prompt,
        }
        if stop:
            payload["stop"] = stop
        response = requests.post(self.chat_url, json=payload, timeout=30000)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _msg_value_to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    return [value]


def official_msgs_to_vllm_user_content(msgs: List[Dict[str, Any]], client: VLLMClient) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for msg in msgs:
        msg_type = msg["type"]
        value = msg["value"]
        if msg_type == "text":
            content.append({"type": "text", "text": value})
        elif msg_type == "image":
            for image_path in _msg_value_to_list(value):
                image_base64 = client.encode_image_to_base64(image_path)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
        elif msg_type == "video":
            video_items = _msg_value_to_list(value)
            for item in video_items:
                if os.path.exists(item) and item.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_base64 = client.encode_image_to_base64(item)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
                else:
                    content.append({"type": "video_url", "video_url": {"url": f"file://{item}"}})
        else:
            raise ValueError(f"Unsupported official message type: {msg_type}")
    return content


class ThinklessVLLMEvaluator:
    def __init__(
        self,
        model_name: str,
        vllm_url: str,
        tokenizer_path: str,
        output_dir: str,
        setting: str = "parallel",
        parallel_workers: int = 32,
        think_budget: int = 300,
        answer_budget: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
    ) -> None:
        set_official_prompt_env(None)
        self.model_name = model_name
        self.client = VLLMClient(vllm_url, model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setting = setting
        self.parallel_workers = parallel_workers
        self.think_budget = think_budget
        self.answer_budget = answer_budget
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _build_base_messages(self, dataset, row, dataset_name: str) -> tuple[List[Dict[str, Any]], str]:
        source_message = build_official_eval_message(
            dataset,
            dataset_name,
            row,
            video_llm=True,
        )
        user_content = official_msgs_to_vllm_user_content(source_message, self.client)
        regulation_prompt = task_instruction(dataset_name, row)
        base_messages = [
            {"role": "system", "content": TAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return base_messages, regulation_prompt

    def _generate_think(self, base_messages: List[Dict[str, Any]]) -> tuple[str, float]:
        t0 = time.perf_counter()
        messages = list(base_messages)
        messages.append({"role": "assistant", "content": "<think>"})
        response = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.think_budget,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            continue_final_message=True,
            add_generation_prompt=False,
        )
        return response, time.perf_counter() - t0

    def _generate_answer(
        self,
        base_messages: List[Dict[str, Any]],
        think_text: str,
        regulation_prompt: str,
    ) -> tuple[str, float]:
        t0 = time.perf_counter()
        messages = list(base_messages)
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{think_text}</think>\n{regulation_prompt}\n",
            }
        )
        response = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.answer_budget,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            continue_final_message=True,
            add_generation_prompt=False,
        )
        return response, time.perf_counter() - t0

    def _process_single_sample(self, idx: int, row, dataset, dataset_name: str) -> Dict[str, Any]:
        base_messages, regulation_prompt = self._build_base_messages(dataset, row, dataset_name)
        think_raw, think_latency = self._generate_think(base_messages)
        think_text = _clean_think_text(think_raw)
        answer_text, answer_latency = self._generate_answer(base_messages, think_text, regulation_prompt)

        think_tok_len = len(self.tokenizer.encode(think_text, add_special_tokens=False))
        answer_tok_len = len(self.tokenizer.encode(answer_text, add_special_tokens=False))
        total_tok_len = think_tok_len + answer_tok_len
        return {
            "index": row["index"],
            "prediction": answer_text,
            "answer": row.get("answer", ""),
            "item_latency_sec": think_latency + answer_latency,
            "thinking_latency_sec": think_latency,
            "answer_latency_sec": answer_latency,
            "thinking_text": think_text,
            "thinking_length_tokens": think_tok_len,
            "answer_length_tokens": answer_tok_len,
            "output_length_tokens": total_tok_len,
        }

    def evaluate_dataset(self, dataset_name: str, dataset_class, limit: Optional[int] = None) -> EvalResult:
        dataset = dataset_class(dataset=dataset_name)
        data = dataset.data
        if limit is not None:
            data = data.iloc[:limit].copy()

        num_samples = len(data)
        predictions: List[Dict[str, Any]] = []
        output_token_lengths: List[int] = []
        item_latencies: List[float] = []
        start_time = time.time()

        if self.setting == "sequential":
            for idx in tqdm(range(num_samples), desc=f"Thinkless-vLLM {dataset_name}"):
                row = data.iloc[idx]
                result = self._process_single_sample(idx, row, dataset, dataset_name)
                predictions.append(result)
                output_token_lengths.append(result["output_length_tokens"])
                item_latencies.append(result["item_latency_sec"])
        else:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_to_idx = {
                    executor.submit(self._process_single_sample, idx, data.iloc[idx], dataset, dataset_name): idx
                    for idx in range(num_samples)
                }
                results: List[Optional[Dict[str, Any]]] = [None] * num_samples
                for future in tqdm(as_completed(future_to_idx), total=num_samples, desc=f"Thinkless-vLLM {dataset_name}"):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
            for result in results:
                assert result is not None
                predictions.append(result)
                output_token_lengths.append(result["output_length_tokens"])
                item_latencies.append(result["item_latency_sec"])

        total_time = time.time() - start_time
        avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
        avg_item_latency = sum(item_latencies) / len(item_latencies) if item_latencies else 0.0

        pred_df = pd.DataFrame(predictions)
        pred_path = self.output_dir / f"{dataset_name}_thinkless_{self.setting}_predictions.xlsx"
        dump(pred_df, str(pred_path))

        return EvalResult(
            dataset_name=dataset_name,
            num_samples=len(predictions),
            total_time=total_time,
            avg_time_per_sample=total_time / len(predictions) if predictions else 0.0,
            avg_item_latency_sec=avg_item_latency,
            accuracy=0.0,
            avg_output_tokens=avg_output_tokens,
            details={"predictions_file": str(pred_path)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM-based Thinkless time eval for Qwen3-VL.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MathVista_MINI", "MathVerse_MINI", "LogicVista", "Video_Holmes", "GSM8K"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setting", choices=["sequential", "parallel"], default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=32)
    parser.add_argument("--think-budget", type=int, default=300)
    parser.add_argument("--answer-budget", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=None, help="Deprecated alias for --think-budget")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.token_budget is not None:
        args.think_budget = args.token_budget

    response = requests.get(f"{args.vllm_url}/health", timeout=10)
    response.raise_for_status()

    evaluator = ThinklessVLLMEvaluator(
        model_name=args.model,
        vllm_url=args.vllm_url,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        setting=args.setting,
        parallel_workers=args.parallel_workers,
        think_budget=args.think_budget,
        answer_budget=args.answer_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    served_models = evaluator.client.list_models()
    if served_models and args.model not in served_models:
        if len(served_models) == 1:
            evaluator.client.model_name = served_models[0]
        else:
            raise RuntimeError(f"Requested model {args.model} is not in served models: {served_models}")

    dataset_class_map = {
        "MathVista_MINI": MathVista,
        "MathVerse_MINI": MathVerse,
        "LogicVista": LogicVista,
        "Video_Holmes": Video_Holmes,
        "GSM8K": MathReasoningDataset,
    }

    results = []
    for dataset_name in args.datasets:
        if dataset_name in dataset_class_map:
            results.append(evaluator.evaluate_dataset(dataset_name, dataset_class_map[dataset_name], limit=args.limit))

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "dataset": result.dataset_name,
                "num_samples": result.num_samples,
                "total_time_sec_wall": result.total_time,
                "avg_time_per_sample_sec_wall": result.avg_time_per_sample,
                "avg_item_latency_sec": result.avg_item_latency_sec,
                "avg_output_length_tokens": result.avg_output_tokens,
                "accuracy": result.accuracy,
            }
        )

    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.total_time for r in results)
    avg_output_tokens_overall = (
        sum(r.avg_output_tokens * r.num_samples for r in results) / total_samples if total_samples > 0 else 0.0
    )
    avg_item_latency_overall = (
        sum(r.avg_item_latency_sec * r.num_samples for r in results) / total_samples if total_samples > 0 else 0.0
    )

    summary_rows.append(
        {
            "dataset": "TOTAL",
            "num_samples": total_samples,
            "total_time_sec_wall": total_time,
            "avg_time_per_sample_sec_wall": (total_time / total_samples) if total_samples > 0 else 0.0,
            "avg_item_latency_sec": avg_item_latency_overall,
            "avg_output_length_tokens": avg_output_tokens_overall,
            "accuracy": None,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.output_dir) / f"summary_thinkless_{args.setting}.xlsx"
    dump(summary_df, str(summary_path))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
