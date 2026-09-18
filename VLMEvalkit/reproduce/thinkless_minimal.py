#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from vlmeval.config import supported_VLM
from vlmeval.dataset import DATASET_TYPE, build_dataset
from vlmeval.smp import dump


THINKLESS_SYSTEM_PROMPT = (
    "You are a careful multimodal reasoning assistant. "
    "First think about the reasoning process as an internal monologue enclosed "
    "within <think> </think> tags, then provide the final answer enclosed within "
    "<answer> </answer> tags."
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


def append_tag_instruction(messages: list[dict], dataset_name: str) -> list[dict]:
    dataset_type = DATASET_TYPE(dataset_name, default=None)
    suffix = (
        "\n\nThink carefully. Put the reasoning in <think> </think> and the final answer in <answer> </answer>."
    )
    if dataset_type == "MCQ" or dataset_name == "Video_Holmes":
        suffix += " For the final answer, output only the single option letter inside <answer>."
    elif dataset_type == "Y/N":
        suffix += " For the final answer, output only yes or no inside <answer>."
    else:
        suffix += " Keep the final answer concise inside <answer>."

    updated = []
    appended = False
    for msg in messages:
        copied = dict(msg)
        if copied["type"] == "text":
            copied["value"] = f'{copied["value"]}{suffix}'
            appended = True
        updated.append(copied)
    if not appended:
        updated.append(dict(type="text", value=suffix.strip()))
    return updated


class ThinkLessRunner:
    def __init__(
        self,
        model_name: str,
        think_budget: int,
        answer_budget: int,
        temperature: float = 0.0,
    ) -> None:
        sanitize_colon_env_var("PATH")
        sanitize_colon_env_var("LD_LIBRARY_PATH")
        self.model_name = model_name
        self.think_budget = think_budget
        self.answer_budget = answer_budget
        self.temperature = temperature
        self.model = supported_VLM[model_name](
            use_vllm=False,
            system_prompt=THINKLESS_SYSTEM_PROMPT,
            temperature=temperature,
        )
        self.model.max_new_tokens = think_budget + answer_budget
        self.model.generate_kwargs["max_new_tokens"] = think_budget + answer_budget

    def _generation_kwargs(self, max_new_tokens: int) -> dict:
        kwargs = {
            "max_new_tokens": max_new_tokens,
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

    def _prepare_messages(self, message: list[dict], dataset: str):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError("qwen_vl_utils is required for ThinkLess minimal runner") from err

        messages = []
        if self.model.system_prompt is not None:
            messages.append({"role": "system", "content": self.model.system_prompt})
        messages.append({"role": "user", "content": self.model._prepare_content(message, dataset=dataset)})
        text = self.model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        return messages, inputs

    def _decode_generated(self, inputs, generated_ids):
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.model.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0]

    def _generate_from_messages(self, messages, max_new_tokens):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError("qwen_vl_utils is required for ThinkLess minimal runner") from err

        text = self.model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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

        generated_ids = self.model.model.generate(
            **inputs,
            **self._generation_kwargs(max_new_tokens),
        )
        return self._decode_generated(inputs, generated_ids)

    def generate_thinkless(self, message: list[dict], dataset: str) -> dict:
        base_messages, inputs = self._prepare_messages(message, dataset)

        stage1_ids = self.model.model.generate(
            **inputs,
            **self._generation_kwargs(self.think_budget),
        )
        stage1_text = self._decode_generated(inputs, stage1_ids)

        if "</think>" in stage1_text and "<answer>" in stage1_text:
            final_text = stage1_text
        else:
            trimmed_think = stage1_text
            if "<answer>" in trimmed_think:
                trimmed_think = trimmed_think.split("<answer>", 1)[0]
            trimmed_think = trimmed_think.replace("</think>", "").strip()
            continuation_messages = list(base_messages)
            continuation_messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{trimmed_think}</think>\n<answer>",
                }
            )
            stage2_tail = self._generate_from_messages(continuation_messages, self.answer_budget)
            final_text = f"<think>{trimmed_think}</think>\n<answer>{stage2_tail}"
            if "</answer>" not in final_text:
                final_text += "</answer>"

        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", final_text, re.S | re.I)
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", final_text, re.S | re.I)
        return {
            "raw_output": final_text,
            "thinking": think_match.group(1).strip() if think_match else "",
            "answer": answer_match.group(1).strip() if answer_match else final_text.strip(),
        }


def build_message(model, dataset, dataset_name: str, row):
    if getattr(dataset, "MODALITY", None) == "VIDEO":
        sample = int(row["index"]) if "index" in row else 0
        message = dataset.build_prompt(sample, video_llm=getattr(model, "VIDEO_LLM", False))
        return append_tag_instruction(message, dataset_name)

    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        model.set_dump_image(dataset.dump_image)
        message = model.build_prompt(row, dataset=dataset_name)
    else:
        message = dataset.build_prompt(row)
    return append_tag_instruction(message, dataset_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ThinkLess verification on VLMEvalKit tasks.")
    parser.add_argument("--model", required=True, choices=["Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Thinking"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--think-budget", type=int, default=96)
    parser.add_argument("--answer-budget", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "thinkless_minimal.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = ThinkLessRunner(
        model_name=args.model,
        think_budget=args.think_budget,
        answer_budget=args.answer_budget,
        temperature=args.temperature,
    )

    results = []
    for dataset_name in args.datasets:
        dataset = build_dataset(dataset_name)
        sample_count = min(args.limit, len(dataset.data))
        for idx in range(sample_count):
            row = dataset.data.iloc[idx]
            message = build_message(runner.model, dataset, dataset_name, row)
            start = time.time()
            output = runner.generate_thinkless(message, dataset_name)
            elapsed = time.time() - start
            results.append(
                {
                    "model": args.model,
                    "dataset": dataset_name,
                    "index": int(row["index"]),
                    "question": str(row.get("question", "")),
                    "elapsed_sec": round(elapsed, 3),
                    **output,
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(results, str(output_path))
    print(f"Saved ThinkLess minimal results to {output_path}")


if __name__ == "__main__":
    main()
