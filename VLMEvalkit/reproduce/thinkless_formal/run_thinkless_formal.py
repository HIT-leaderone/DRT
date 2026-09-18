#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch

from vlmeval.config import supported_VLM
from vlmeval.dataset import DATASET_TYPE, build_dataset
from vlmeval.smp import dump, get_logger, load


TAG_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue enclosed within <think> </think> "
    "and then provide the final answer."
)


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


def build_formal_message(model, dataset, dataset_name: str, row):
    if getattr(dataset, "MODALITY", None) == "VIDEO":
        sample = int(row["index"]) if "index" in row else 0
        return dataset.build_prompt(sample, video_llm=getattr(model, "VIDEO_LLM", False))

    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        model.set_dump_image(dataset.dump_image)
        return model.build_prompt(row, dataset=dataset_name)

    return dataset.build_prompt(row)


class FormalThinkLessRunner:
    def __init__(
        self,
        model_name: str,
        token_budget: int,
        temperature: float = 0.0,
    ) -> None:
        sanitize_colon_env_var("PATH")
        sanitize_colon_env_var("LD_LIBRARY_PATH")
        self.logger = get_logger("THINKLESS")
        self.model_name = model_name
        self.token_budget = token_budget
        self.temperature = temperature
        self.model = supported_VLM[model_name](
            use_vllm=False,
            system_prompt=TAG_SYSTEM_PROMPT,
            temperature=temperature,
        )
        self.model.max_new_tokens = token_budget
        self.model.generate_kwargs["max_new_tokens"] = token_budget

    def _generation_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": self.token_budget,
            "repetition_penalty": self.model.repetition_penalty,
        }
        if self.temperature is None or self.temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.model.top_p,
                    "top_k": self.model.top_k,
                }
            )
        return kwargs

    def _prepare_messages(self, message: list[dict], dataset: str, regulation_prompt: str):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError("qwen_vl_utils is required") from err

        messages = []
        if self.model.system_prompt is not None:
            messages.append({"role": "system", "content": self.model.system_prompt})
        messages.append({"role": "user", "content": self.model._prepare_content(message, dataset=dataset)})
        think_prefix = "<think>"
        prefix_text = self.model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = prefix_text + think_prefix

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.model.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        try:
            inputs = inputs.to(self.model.model.device)
            if hasattr(self.model.model, "dtype"):
                inputs = inputs.to(self.model.model.dtype)
        except Exception:
            inputs = inputs.to("cuda")

        regulation_text = f"</think>\n{regulation_prompt}\n"
        extra_ids = self.model.processor.tokenizer(
            regulation_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], extra_ids], dim=1)
        if "attention_mask" in inputs:
            extra_mask = torch.ones_like(extra_ids, device=inputs["attention_mask"].device)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], extra_mask], dim=1)
        return inputs

    def generate(self, message: list[dict], dataset: str, regulation_prompt: str) -> str:
        inputs = self._prepare_messages(message, dataset, regulation_prompt)
        generated_ids = self.model.model.generate(
            **inputs,
            **self._generation_kwargs(),
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.model.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Formal isolated ThinkLess reproduction on VLMEvalKit datasets.")
    parser.add_argument("--model", required=True, choices=["Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Thinking"])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--token-budget", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", default="gpt-4.1-2025-04-14")
    parser.add_argument("--api-nproc", type=int, default=4)
    return parser.parse_args()


def output_file_for_dataset(work_dir: Path, model_name: str, dataset_name: str) -> Path:
    suffix = ".xlsx"
    if dataset_name == "GSM8K":
        suffix = ".xlsx"
    return work_dir / f"{model_name}_{dataset_name}{suffix}"


def run_dataset(runner: FormalThinkLessRunner, dataset_name: str, work_dir: Path, limit: int | None, judge: str, api_nproc: int):
    dataset = build_dataset(dataset_name)
    data = dataset.data.copy()
    if limit is not None:
        data = data.iloc[:limit].copy()

    predictions = []
    records = []
    for _, row in data.iterrows():
        message = build_formal_message(runner.model, dataset, dataset_name, row)
        regulation_prompt = task_instruction(dataset_name, row)
        start = time.time()
        response = runner.generate(message, dataset_name, regulation_prompt)
        elapsed = time.time() - start
        predictions.append(str(response))
        rec = row.to_dict()
        rec["prediction"] = str(response)
        rec["elapsed_sec"] = round(elapsed, 3)
        records.append(rec)
        print(f"[{runner.model_name}] {dataset_name} index={row['index']} elapsed={elapsed:.2f}s")

    result_df = pd.DataFrame(records)
    result_path = output_file_for_dataset(work_dir, runner.model_name, dataset_name)
    dump(result_df, str(result_path))

    judge_kwargs = {
        "model": judge,
        "nproc": api_nproc,
        "verbose": False,
        "retry": 3,
    }
    score = dataset.evaluate(str(result_path), **judge_kwargs)
    return {
        "dataset": dataset_name,
        "result_file": str(result_path),
        "score": score.to_dict() if hasattr(score, "to_dict") else score,
    }


def main():
    args = parse_args()
    os.environ.pop("PROMPT_TYPE", None)
    work_dir = Path(args.work_dir) / args.model
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = FormalThinkLessRunner(
        model_name=args.model,
        token_budget=args.token_budget,
        temperature=args.temperature,
    )

    summary = {
        "model": args.model,
        "token_budget": args.token_budget,
        "top_k": 1,
        "datasets": [],
    }
    for dataset_name in args.data:
        summary["datasets"].append(
            run_dataset(
                runner=runner,
                dataset_name=dataset_name,
                work_dir=work_dir,
                limit=args.limit,
                judge=args.judge,
                api_nproc=args.api_nproc,
            )
        )

    summary_path = work_dir / "thinkless_summary.json"
    dump(summary, str(summary_path))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
