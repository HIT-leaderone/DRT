#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

try:
    import vllm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("vllm is required for VisionThink bridge evaluation") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = OFFICIAL_ROOT / "results"

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.smp import dump  # noqa: E402


TOOL_CALL_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "resize_image", "name": "resize_image", "description": "Resize the image resolution.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `resize`: Double the resolution of the current image. You should only use this tool if you are unable to obtain the critical information needed to answer the question from the current resolution.", "enum": ["resize"], "type": "string"}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

TOOL_CALL_MULTI_TRUN_PROMPT = (
    "Please carefully analyze the content returned from the image resize tool in combination "
    "with the original question and image from the user, continue your reasoning process inside "
    "<think> and </think> and then write your final answer inside <answer> and </answer>."
)

ERROR_INFO_MULTI_TURN_PROMPT = (
    "Please analyze the error information obtained from the function tool and adjust your response. "
    "Countinue your reasoning process inside <think> and </think>."
)

VISIONTHINK_PROMPT_TEMPLATE = (
    "Answer the question based on the image provided. You must conduct reasoning within <think> and "
    "</think> first in each of your reasoning steps. You may call ONE function tool per step to help "
    "you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the "
    "end of each step to perform a function call. You should continue your reasoning process based on "
    "the content returned by the function tool. Once you confirm your final answer, place the final "
    "answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer "
    "value or choice with \\boxed{{}}. Here is the image and question:\n{Question}"
)

TEXT_ONLY_PROMPT_TEMPLATE = (
    "Answer the question carefully. You must conduct reasoning within <think> and </think> first, "
    "and then write your final answer inside <answer> and </answer>. For mathematical or multiple-choice "
    "problem, wrap the answer value or choice with \\boxed{{}}.\n{Question}"
)

DATASETS = [
    "MathVista_MINI",
    "MathVerse_MINI",
    "LogicVista",
    "GSM8K",
    "Video_Holmes",
]


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


@dataclass
class SampleInput:
    index: Any
    prompt_token_ids: list[int]
    multi_modal_data: dict[str, list[Image.Image]]
    original_images: list[Image.Image]
    answer: Any
    question: str


def ensure_min_side(image: Image.Image, min_side: int = 28) -> Image.Image:
    width, height = image.size
    if width >= min_side and height >= min_side:
        return image
    scale = max(min_side / max(width, 1), min_side / max(height, 1))
    new_width = max(min_side, int(width * scale))
    new_height = max(min_side, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def downsample_half(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = max(28, width // 2)
    new_height = max(28, height // 2)
    return ensure_min_side(image.resize((new_width, new_height), Image.Resampling.LANCZOS))


def load_image(image_ref: str | Image.Image) -> Image.Image:
    if isinstance(image_ref, Image.Image):
        return image_ref.convert("RGB")
    return Image.open(image_ref).convert("RGB")


def parse_candidates(raw_value: Any) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return []
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if isinstance(raw_value, tuple):
        return [str(x) for x in raw_value]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, tuple):
            return [str(x) for x in parsed]
        return [text]
    return [str(raw_value)]


def collect_options(row: pd.Series) -> list[str]:
    options: list[str] = []
    if "candidates" in row:
        options.extend(parse_candidates(row["candidates"]))

    if "options" in row and row["options"] not in [None, ""]:
        raw = row["options"]
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                pass
        if isinstance(raw, dict):
            options.extend([f"{k}. {v}" for k, v in raw.items()])
        elif isinstance(raw, (list, tuple)):
            options.extend([f"{chr(65 + i)}. {v}" for i, v in enumerate(raw)])

    if "choices" in row and row["choices"] not in [None, ""]:
        raw = row["choices"]
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                pass
        if isinstance(raw, dict):
            options.extend([f"{k}. {v}" for k, v in raw.items()])
        elif isinstance(raw, (list, tuple)):
            options.extend([f"{chr(65 + i)}. {v}" for i, v in enumerate(raw)])

    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if label in row and not pd.isna(row[label]):
            options.append(f"{label}. {row[label]}")

    deduped: list[str] = []
    seen = set()
    for option in options:
        norm = str(option).strip()
        if not norm or norm in seen:
            continue
        deduped.append(norm)
        seen.add(norm)
    return deduped


def build_plain_question(dataset_name: str, row: pd.Series) -> str:
    question = str(row.get("question", "") if not pd.isna(row.get("question", "")) else "").strip()
    hint = str(row.get("hint", "") if not pd.isna(row.get("hint", "")) else "").strip()
    options = collect_options(row)

    parts: list[str] = []
    if hint:
        parts.append(f"Hint: {hint}")

    if dataset_name == "GSM8K":
        parts.append(f"Question: {question}")
        parts.append("Provide the final numerical answer.")
        return "\n".join(parts).strip()

    if question:
        parts.append(f"Question: {question}")

    if options:
        parts.append("Options:")
        parts.extend(options)
        parts.append("Select the best option.")

    return "\n".join(parts).strip()


def extract_visuals(dataset_name: str, dataset, row: pd.Series) -> list[Image.Image]:
    if dataset_name == "Video_Holmes":
        message = dataset.build_prompt(row, video_llm=False)
        images = [load_image(item["value"]) for item in message if item["type"] == "image"]
        return images

    try:
        dumped = dataset.dump_image(row)
    except Exception:
        dumped = None

    if dumped is None:
        return []

    if isinstance(dumped, list):
        return [load_image(item) for item in dumped]
    return [load_image(dumped)]


def prepare_tool_call_inputs(json_objects: list[dict[str, Any]]) -> str:
    for obj in json_objects:
        action_type = obj["arguments"]["action"]
        if action_type not in {"resize"}:
            raise AssertionError(f"Unknown Tool Type: {action_type}.")
        if len(json_objects) != 1:
            raise AssertionError("You should only call function `resize` once per function call.")
    return "resize"


class VisionThinkRunner:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        batch_size: int,
        max_images: int,
        max_generation_round: int,
        downsample_image: bool,
        gpu_memory_utilization: float,
    ) -> None:
        sanitize_colon_env_var("PATH")
        sanitize_colon_env_var("LD_LIBRARY_PATH")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_images = max_images
        self.max_generation_round = max_generation_round
        self.downsample_image = downsample_image
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.inference_engine = vllm.LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=32768,
            enable_prefix_caching=True,
            max_model_len=32768,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max_images},
        )

    def make_conversation(self, problem: str, system_prompt: str, prompt_template: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template.format(Question=problem)},
        ]

    def build_sample_input(self, dataset_name: str, dataset, row: pd.Series) -> SampleInput:
        visuals = extract_visuals(dataset_name, dataset, row)
        prompt_question = build_plain_question(dataset_name, row)
        has_visual = len(visuals) > 0

        if has_visual:
            image_prefix = "<image>\n" * len(visuals)
            prompt = self.make_conversation(f"{image_prefix}{prompt_question}", TOOL_CALL_SYSTEM_PROMPT, VISIONTHINK_PROMPT_TEMPLATE)
        else:
            prompt = self.make_conversation(prompt_question, "You are a helpful assistant.", TEXT_ONLY_PROMPT_TEMPLATE)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

        if has_visual:
            raw_prompt = prompt_with_chat_template.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            downsampled = [downsample_half(image) if self.downsample_image else ensure_min_side(image) for image in visuals]
            originals = [ensure_min_side(image) for image in visuals]
            mm_data = {"image": downsampled}
        else:
            raw_prompt = prompt_with_chat_template
            originals = []
            mm_data = {}

        return SampleInput(
            index=row["index"],
            prompt_token_ids=self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            multi_modal_data=mm_data,
            original_images=originals,
            answer=row.get("answer", ""),
            question=prompt_question,
        )

    def _sampling_params(self):
        return vllm.SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            top_p=0.95,
            n=1,
        )

    def _decode_outputs(self, outputs) -> list[list[int]]:
        response_ids: list[list[int]] = []
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            filtered = [token_id for token_id in token_ids if token_id <= 151664]
            if 151645 not in filtered and filtered:
                filtered[-1] = 151645
            response_ids.append(filtered)
        return response_ids

    def _extract_answer(self, text: str) -> str:
        if "<answer>" in text and "</answer>" in text:
            return text.split("</answer>")[0].split("<answer>")[-1].strip()
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
        return text.strip()

    def generate_batch(self, batch_samples: list[SampleInput]) -> tuple[list[str], list[int]]:
        batched_inputs = []
        response_masks: list[list[Any]] = []
        prefix_lengths: list[int] = []

        for sample in batch_samples:
            sample_input = {
                "prompt_token_ids": list(sample.prompt_token_ids),
                "original_images": list(sample.original_images),
            }
            if sample.multi_modal_data:
                sample_input["multi_modal_data"] = {k: list(v) for k, v in sample.multi_modal_data.items()}
            batched_inputs.append(sample_input)
            prefix_lengths.append(len(sample.prompt_token_ids))
            response_masks.append([0] * len(sample.prompt_token_ids))

        to_generate = list(range(len(batched_inputs)))
        round_idx = 0

        while round_idx < self.max_generation_round and to_generate:
            prompts = [batched_inputs[i] for i in to_generate]
            outputs = self.inference_engine.generate(prompts=prompts, sampling_params=self._sampling_params(), use_tqdm=False)
            response_token_ids = self._decode_outputs(outputs)

            remove_indices: list[int] = []
            tool_queries: dict[int, dict[str, Any]] = {}

            for request_idx, generated_ids in zip(to_generate, response_token_ids):
                batched_inputs[request_idx]["prompt_token_ids"].extend(generated_ids)
                response_masks[request_idx].extend([1] * len(generated_ids))
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                tool_matches = re.findall(r"<tool_call>(.*?)</tool_call>", decoded, flags=re.DOTALL)

                if not tool_matches:
                    remove_indices.append(request_idx)
                    continue

                try:
                    json_pattern = re.compile(r"\{.*?\}\}", re.DOTALL)
                    parsed = []
                    for match in tool_matches:
                        parsed.extend([json.loads(item) for item in json_pattern.findall(match)])
                    tool_type = prepare_tool_call_inputs(parsed)
                    tool_queries[request_idx] = {"tool_type": tool_type, "error": None}
                except Exception as exc:
                    tool_queries[request_idx] = {"tool_type": None, "error": str(exc)}

            for done_idx in remove_indices:
                to_generate.remove(done_idx)

            next_round = list(to_generate)
            for request_idx in next_round:
                query = tool_queries.get(request_idx)
                if query is None:
                    to_generate.remove(request_idx)
                    continue

                if query["tool_type"] == "resize" and batched_inputs[request_idx]["original_images"]:
                    resized_images = batched_inputs[request_idx]["original_images"]
                    tool_prompt = "<|im_start|>user\n<tool_response>\nThe resized image is shown below:\n"
                    for _ in resized_images:
                        tool_prompt += "<|vision_start|><|image_pad|><|vision_end|>\n"
                    tool_prompt += "</tool_response>\n" + TOOL_CALL_MULTI_TRUN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                    next_prompt_ids = self.tokenizer.encode(tool_prompt, add_special_tokens=False)
                    batched_inputs[request_idx]["prompt_token_ids"].extend(next_prompt_ids)
                    response_masks[request_idx].extend([0] * len(next_prompt_ids))
                    batched_inputs[request_idx]["multi_modal_data"].setdefault("image", []).extend(resized_images)
                else:
                    error_text = query["error"] or "Tool call failed."
                    tool_prompt = (
                        "<|im_start|>user\n"
                        + error_text
                        + ERROR_INFO_MULTI_TURN_PROMPT
                        + "<|im_end|>\n<|im_start|>assistant\n"
                    )
                    next_prompt_ids = self.tokenizer.encode(tool_prompt, add_special_tokens=False)
                    batched_inputs[request_idx]["prompt_token_ids"].extend(next_prompt_ids)
                    response_masks[request_idx].extend([0] * len(next_prompt_ids))

            round_idx += 1

        final_texts: list[str] = []
        final_lengths: list[int] = []
        for sample_idx, sample in enumerate(batch_samples):
            all_ids = batched_inputs[sample_idx]["prompt_token_ids"][prefix_lengths[sample_idx]:]
            valid_ids = [token_id for token_id, mask in zip(all_ids, response_masks[sample_idx][prefix_lengths[sample_idx]:]) if mask == 1]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            final_texts.append(text)
            final_lengths.append(len(valid_ids))
        return final_texts, final_lengths

    def evaluate_dataset(
        self,
        dataset_name: str,
        limit: int | None,
        work_dir: Path,
        judge_model: str,
        api_nproc: int,
    ) -> dict[str, Any]:
        dataset = build_dataset(dataset_name)
        data = dataset.data.copy()
        if limit is not None:
            data = data.iloc[:limit].copy()

        prepared = [self.build_sample_input(dataset_name, dataset, row) for _, row in data.iterrows()]
        predictions = []

        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start:start + self.batch_size]
            raw_texts, token_lengths = self.generate_batch(batch)
            for sample, raw_text, token_length in zip(batch, raw_texts, token_lengths):
                predictions.append(
                    {
                        "index": sample.index,
                        "prediction": self._extract_answer(raw_text),
                        "raw_prediction": raw_text,
                        "answer": sample.answer,
                        "output_length_tokens": token_length,
                    }
                )

        pred_map = {item["index"]: item for item in predictions}
        records = []
        for _, row in data.iterrows():
            row_dict = row.to_dict()
            pred_item = pred_map[row_dict["index"]]
            row_dict["prediction"] = pred_item["prediction"]
            row_dict["raw_prediction"] = pred_item["raw_prediction"]
            row_dict["output_length_tokens"] = pred_item["output_length_tokens"]
            records.append(row_dict)

        pred_df = pd.DataFrame(records)
        model_slug = Path(self.model_path).name.replace("/", "_")
        pred_path = work_dir / f"{model_slug}_{dataset_name}.xlsx"
        dump(pred_df, str(pred_path))

        judge_kwargs = {
            "model": judge_model,
            "nproc": api_nproc,
            "verbose": False,
            "retry": 3,
        }
        score = dataset.evaluate(str(pred_path), **judge_kwargs)
        avg_output_tokens = float(pred_df["output_length_tokens"].mean()) if len(pred_df) else 0.0

        return {
            "dataset": dataset_name,
            "result_file": str(pred_path),
            "num_samples": int(len(pred_df)),
            "avg_output_tokens": avg_output_tokens,
            "score": score.to_dict() if hasattr(score, "to_dict") else score,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisionThink official inference on VLMEvalKit datasets.")
    parser.add_argument("--model-path", default="Senqiao/VisionThink-Efficient")
    parser.add_argument("--data", nargs="+", default=DATASETS)
    parser.add_argument("--work-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-generation-round", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--no-downsample-image", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", default="gpt-4.1-2025-04-14")
    parser.add_argument("--api-nproc", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_slug = Path(args.model_path).name.replace("/", "_")
    model_dir = work_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    runner = VisionThinkRunner(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_images=args.max_images,
        max_generation_round=args.max_generation_round,
        downsample_image=not args.no_downsample_image,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    summary = {
        "model": args.model_path,
        "datasets": [],
        "downsample_image": not args.no_downsample_image,
        "max_generation_round": args.max_generation_round,
    }
    for dataset_name in args.data:
        result = runner.evaluate_dataset(
            dataset_name=dataset_name,
            limit=args.limit,
            work_dir=model_dir,
            judge_model=args.judge,
            api_nproc=args.api_nproc,
        )
        summary["datasets"].append(result)

    summary_path = model_dir / "visionthink_vlmeval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
