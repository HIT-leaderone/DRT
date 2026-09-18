import argparse
import asyncio
import importlib.util
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERL_ROOT = Path(".")
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score import short_cot_qwen

_gpt_model_spec = importlib.util.spec_from_file_location("verl_root_gpt_model", VERL_ROOT / "gpt_model.py")
_gpt_model_module = importlib.util.module_from_spec(_gpt_model_spec)
_gpt_model_spec.loader.exec_module(_gpt_model_module)
GPT4o = _gpt_model_module.GPT4o


DEFAULT_GT_PARQUET = Path("./data/mathvista/testmini.parquet")
DEFAULT_OUTPUT_DIR = Path("./fig")
DEFAULT_ROUTER_JUDGE_MODEL = "./Qwen3-235B-A22B-Instruct-2507"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.1-2025-11-13"
DROP_GT_STEP_COUNTS = {25}
MAX_GT_STEP_BUCKET = 12
ANALYSIS_VERSION = "mathvista_separate_prompts_v2"
CACHE_FILENAME = f"{ANALYSIS_VERSION}_cache.json"
MODEL_PLOT_ORDER = [
    "Qwen-Standard",
    "DRT-SFT-800it",
    "w/o process reward",
    "w/o step bouns",
    "DRT-RL",
]
MODEL_PLOT_TITLE_MAP = {
    "Qwen-Standard": "Qwen",
    "DRT-SFT-800it": "DRT-SFT",
    "w/o process reward": "w/o process reward",
    "w/o step bouns": "w/o step reward",
    "DRT-RL": "DRT-RL",
}


REASON_STEP_COUNT_PROMPT = """
You are a strict reasoning annotator for a visual math benchmark.

Task: count the student's `reason_step_count`.

Definition:
- `reason_step_count` means the number of distinct atomic reasoning acts in the student's response.
- Count semantically distinct atomic acts such as:
  1. reading or restating a given condition if it is used in the reasoning flow,
  2. extracting a visual fact,
  3. making an assumption,
  4. making a guess or hypothesis,
  5. deriving an intermediate relation,
  6. checking or confirming a previous result,
  7. revising a previous inference after recalculation.
- In DRT-style traces, an atomic act often looks like one `->` segment, but do NOT rely only on formatting.
- In long natural-language responses, judge semantically.
- Do NOT count pure filler, repeated paraphrases of the exact same act, stylistic padding, or copied boilerplate.

Original Problem:
{problem_text}

Student Reasoning:
{pred_think}

Student Final Answer:
{pred_final}

Output strict JSON only:
{{
  "reason_step_count": 0,
  "reason": "brief explanation"
}}
"""


EFFECTIVE_REASONING_STEP_PROMPT = """
Y√
"""


HALLUCINATED_EFFECTIVE_STEP_PROMPT = """
You are a strict hallucination annotator for a visual math benchmark.

Task: count the student's `hallucinated_effective_step_count`.

Definition:
- Only consider steps that would qualify as effective reasoning steps under this definition:
  distinct, meaningful, solution-advancing process steps that either match GT intermediate milestones
  or form a genuinely useful alternative path toward the answer.
- Among those effective steps only, count how many contain hallucination-like behavior:
  1. unsupported guesses,
  2. unjustified assumptions,
  3. fabricated visual observations,
  4. invented numeric values,
  5. logical leaps not rigorously derived from the problem.
- Do NOT count hallucinations that occur only inside non-effective filler text.
- The count must be based on the effective-step subset, not the whole response.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Reasoning:
{pred_think}

Student Final Answer:
{pred_final}

Output strict JSON only:
{{
  "hallucinated_effective_step_count": 0,
  "reason": "brief explanation"
}}
"""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    prediction_path: Path
    extract_pkl_path: Path


DEFAULT_MODEL_SPECS = [
    ModelSpec(
        label="Qwen-Standard",
        prediction_path=Path(
            "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
            "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVista_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
            "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
        ),
    ),
    ModelSpec(
        label="w/o process reward",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVista_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
        ),
    ),
    ModelSpec(
        label="w/o step bouns",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVista_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
        ),
    ),
    ModelSpec(
        label="DRT-RL",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVista_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
        ),
    ),
    ModelSpec(
        label="DRT-SFT-800it",
        prediction_path=Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVista_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
        ),
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MathVista benchmark outputs with the GPT step-bonus judge and "
            "plot step metrics grouped by GT reference step count."
        )
    )
    parser.add_argument("--gt_parquet", type=Path, default=DEFAULT_GT_PARQUET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge_backend",
        choices=["openai", "router"],
        default="openai",
        help="Use OPENAI_API_* env vars or a local reward router endpoint.",
    )
    parser.add_argument(
        "--reward_router_address",
        type=str,
        default=None,
        help="OpenAI-compatible router address like 127.0.0.1:8000 for the original reward path.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="auto",
        help="Judge model name. 'auto' selects a backend-appropriate default.",
    )
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ignore_cache", action="store_true")
    return parser.parse_args()


def load_gt_records(gt_parquet: Path):
    df = pd.read_parquet(gt_parquet)
    records = {}
    for row in df.to_dict("records"):
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        gt_steps, gt_answer = short_cot_qwen.parse_ground_truth(reward_model.get("ground_truth", ""))
        original_index = extra_info.get("original_index")
        if original_index is None:
            continue
        eval_index = int(original_index) + 1
        records[eval_index] = {
            "gt_answer": gt_answer,
            "gt_steps": gt_steps,
            "gt_step_count": len(gt_steps),
            "problem": extra_info.get("problem", ""),
            "extra_info": extra_info,
        }
    return records


def load_extract_results(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        return {int(k): v for k, v in obj.items()}

    raise ValueError(f"Unsupported extract result format in {path}: {type(obj).__name__}")


def _clean_prediction_text(text):
    text = str(text or "").strip()
    text = text.replace("_x000D_", "\n")
    return text.strip()


def _extract_final_answer_heuristic(text):
    text = _clean_prediction_text(text)
    if not text:
        return ""

    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return short_cot_qwen.clean_final_answer(answer_match[-1])

    cue_match = re.findall(
        r"(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if cue_match:
        return short_cot_qwen.clean_final_answer(cue_match[-1])

    tail = text.strip().splitlines()[-1]
    return short_cot_qwen.clean_final_answer(tail)


def build_reasoning_view(raw_prediction, extracted_answer):
    raw_prediction = _clean_prediction_text(raw_prediction)
    extracted_answer = short_cot_qwen.clean_final_answer("" if extracted_answer is None else str(extracted_answer))

    pred_think, pred_final = short_cot_qwen.extract_parts(raw_prediction)
    final_answer = short_cot_qwen.clean_final_answer(extracted_answer or pred_final or _extract_final_answer_heuristic(raw_prediction))

    if pred_think is None:
        think_text = raw_prediction
        think_text = re.sub(r"<answer>.*?</answer>", " ", think_text, flags=re.IGNORECASE | re.DOTALL)
        if final_answer:
            think_text = re.sub(
                r"(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*.*$",
                " ",
                think_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        think_text = think_text.strip()
    else:
        think_text = pred_think.strip()

    return {
        "pred_think": think_text,
        "pred_final": final_answer,
        "normalized_solution": f"<think>{think_text}</think><answer>{final_answer}</answer>",
    }


def load_model_samples(model_spec: ModelSpec, gt_records: dict, limit=None):
    df = pd.read_excel(model_spec.prediction_path)
    extract_results = load_extract_results(model_spec.extract_pkl_path)

    samples = []
    for row in df.to_dict("records"):
        index_value = row.get("index")
        if pd.isna(index_value):
            continue

        eval_index = int(index_value)
        gt_item = gt_records.get(eval_index)
        if gt_item is None:
            continue

        extract_item = extract_results.get(eval_index, {})
        extracted_answer = ""
        if isinstance(extract_item, dict):
            extracted_answer = extract_item.get("res", "")
        elif extract_item is not None:
            extracted_answer = str(extract_item)

        reasoning_view = build_reasoning_view(row.get("prediction", ""), extracted_answer)
        samples.append(
            {
                "model_label": model_spec.label,
                "eval_index": eval_index,
                "question": row.get("question", ""),
                "raw_prediction": row.get("prediction", ""),
                "pred_think": reasoning_view["pred_think"],
                "pred_final": reasoning_view["pred_final"],
                "normalized_solution": reasoning_view["normalized_solution"],
                "judge_extract_answer": extracted_answer,
                "gt_answer": gt_item["gt_answer"],
                "gt_steps": gt_item["gt_steps"],
                "gt_step_count": gt_item["gt_step_count"],
                "problem": gt_item["problem"],
                "extra_info": gt_item["extra_info"],
            }
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples


def get_default_result(gt_steps):
    return {
        "reason_step_count": 0,
        "effective_reasoning_step_count": 0,
        "hallucinated_effective_step_count": 0,
    }


def format_gt_steps(gt_steps):
    return "".join(f"Step {idx + 1}: {step}\n" for idx, step in enumerate(gt_steps))


class StepBonusAnalyzer:
    def __init__(
        self,
        judge_backend,
        reward_router_address=None,
        judge_model="auto",
        max_concurrency=4,
    ):
        self.judge_backend = judge_backend
        self.reward_router_address = reward_router_address
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.openai_client = None

        if judge_backend == "openai":
            resolved_judge_model = judge_model if judge_model != "auto" else DEFAULT_OPENAI_JUDGE_MODEL
            api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_base or not api_key:
                raise ValueError("OPENAI_API_BASE/OPENAI_BASE_URL and OPENAI_API_KEY are required for --judge_backend openai")
            self.openai_client = GPT4o(
                deployment_name=resolved_judge_model,
                api_base=api_base,
                api_key=api_key,
            )
        elif judge_backend == "router":
            resolved_judge_model = judge_model if judge_model != "auto" else DEFAULT_ROUTER_JUDGE_MODEL
            if not reward_router_address:
                raise ValueError("--reward_router_address is required for --judge_backend router")
            if resolved_judge_model != DEFAULT_ROUTER_JUDGE_MODEL:
                print(
                    f"Warning: router backend ignores custom judge model '{resolved_judge_model}' "
                    f"and uses the server-side configured model."
                )
        else:
            raise ValueError(f"Unsupported judge backend: {judge_backend}")

    async def _send_messages(self, messages, max_tokens=512):
        if self.judge_backend == "openai":
            return await asyncio.to_thread(self.openai_client.send_stable_request, messages, temperature=0.0)
        return await short_cot_qwen.generate_chat_aiohttp(
            router_address=self.reward_router_address,
            messages=messages,
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
        )

    @staticmethod
    def _extract_json(text):
        match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    async def _judge_int(self, prompt, field_name, max_tokens=512):
        try:
            response_text = await self._send_messages([{"role": "user", "content": prompt}], max_tokens=max_tokens)
            result = self._extract_json(response_text)
            if not isinstance(result, dict):
                return 0
            return max(0, int(result.get(field_name, 0)))
        except Exception as exc:
            print(f"[Judge Error] field={field_name} error={exc}")
            return 0

    async def judge_sample(self, sample):
        async with self.semaphore:
            gt_steps_str = format_gt_steps(sample["gt_steps"])
            problem_text = (sample["problem"] or "N/A")[:4000]
            pred_think = (sample["pred_think"] or "")[:6000]
            pred_final = sample["pred_final"] or ""

            reason_prompt = REASON_STEP_COUNT_PROMPT.format(
                problem_text=problem_text,
                pred_think=pred_think,
                pred_final=pred_final,
            )
            effective_prompt = EFFECTIVE_REASONING_STEP_PROMPT.format(
                problem_text=problem_text,
                gt_steps_str=gt_steps_str,
                pred_think=pred_think,
                pred_final=pred_final,
            )
            hallucination_prompt = HALLUCINATED_EFFECTIVE_STEP_PROMPT.format(
                problem_text=problem_text,
                gt_steps_str=gt_steps_str,
                pred_think=pred_think,
                pred_final=pred_final,
            )

            reason_step_count = await self._judge_int(reason_prompt, "reason_step_count", max_tokens=384)
            effective_reasoning_step_count = await self._judge_int(
                effective_prompt,
                "effective_reasoning_step_count",
                max_tokens=512,
            )
            hallucinated_effective_step_count = await self._judge_int(
                hallucination_prompt,
                "hallucinated_effective_step_count",
                max_tokens=384,
            )

            effective_reasoning_step_count = min(effective_reasoning_step_count, reason_step_count)
            hallucinated_effective_step_count = min(
                hallucinated_effective_step_count,
                effective_reasoning_step_count,
            )

            return {
                "reason_step_count": reason_step_count,
                "effective_reasoning_step_count": effective_reasoning_step_count,
                "hallucinated_effective_step_count": hallucinated_effective_step_count,
            }


def enrich_metrics(sample, judge_result):
    reason_step_count = int(judge_result.get("reason_step_count", 0))
    effective_reasoning_step_count = int(judge_result.get("effective_reasoning_step_count", 0))
    hallucinated_effective_step_count = int(judge_result.get("hallucinated_effective_step_count", 0))
    hallucination_ratio = 0.0
    if effective_reasoning_step_count > 0:
        hallucination_ratio = min(hallucinated_effective_step_count / effective_reasoning_step_count, 1.0)

    gt_step_count = int(sample["gt_step_count"])
    return {
        "model_label": sample["model_label"],
        "eval_index": sample["eval_index"],
        "original_gt_step_count": gt_step_count,
        "gt_step_count": gt_step_count,
        "reason_step_count": reason_step_count,
        "effective_reasoning_step_count": effective_reasoning_step_count,
        "hallucinated_effective_step_count": hallucinated_effective_step_count,
        "hallucination_ratio": hallucination_ratio,
        "judge_extract_answer": sample["judge_extract_answer"],
        "gt_answer": sample["gt_answer"],
        "pred_final": sample["pred_final"],
        "prediction_has_think_tag": "<think>" in str(sample["raw_prediction"]).lower(),
        "prediction_length_chars": len(str(sample["raw_prediction"] or "")),
    }


def apply_gt_step_bucketing(per_sample_df: pd.DataFrame):
    df = per_sample_df.copy()
    if "original_gt_step_count" not in df.columns:
        df["original_gt_step_count"] = df["gt_step_count"]

    df["original_gt_step_count"] = pd.to_numeric(df["original_gt_step_count"], errors="coerce")
    if "eval_index" in df.columns and df["original_gt_step_count"].isna().any():
        gt_records = load_gt_records(DEFAULT_GT_PARQUET)
        df.loc[df["original_gt_step_count"].isna(), "original_gt_step_count"] = (
            df.loc[df["original_gt_step_count"].isna(), "eval_index"]
            .map(lambda idx: gt_records.get(int(idx), {}).get("gt_step_count"))
        )

    df = df[df["original_gt_step_count"].notna()].copy()
    df["original_gt_step_count"] = df["original_gt_step_count"].astype(int)
    df = df[~df["original_gt_step_count"].isin(DROP_GT_STEP_COUNTS)].copy()
    df["gt_step_count"] = df["original_gt_step_count"].clip(upper=MAX_GT_STEP_BUCKET)
    return df


def load_cache(cache_path: Path, ignore_cache=False):
    if ignore_cache or not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, cache_data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


async def run_analysis(samples, analyzer: StepBonusAnalyzer, cache_path: Path, ignore_cache=False):
    cache_data = load_cache(cache_path, ignore_cache=ignore_cache)
    results = []
    pending = []

    for sample in samples:
        cache_key = f"{sample['model_label']}::{sample['eval_index']}"
        cached = cache_data.get(cache_key)
        if cached is not None:
            results.append(cached)
            continue
        pending.append((cache_key, sample))

    if pending and analyzer is None:
        raise ValueError("Pending samples require a configured analyzer, but analyzer is None.")

    if pending:
        progress = tqdm(total=len(pending), desc="Judging samples")
        save_interval = 20
        completed_since_save = 0

        async def _run_single(cache_key, sample):
            judge_result = await analyzer.judge_sample(sample)
            metrics = enrich_metrics(sample, judge_result)
            return cache_key, metrics

        tasks = [asyncio.create_task(_run_single(cache_key, sample)) for cache_key, sample in pending]
        for task in asyncio.as_completed(tasks):
            cache_key, metrics = await task
            cache_data[cache_key] = metrics
            results.append(metrics)
            progress.update(1)
            completed_since_save += 1
            if completed_since_save >= save_interval:
                save_cache(cache_path, cache_data)
                completed_since_save = 0

        progress.close()
        save_cache(cache_path, cache_data)

    return pd.DataFrame(results)


def build_grouped_summary(per_sample_df: pd.DataFrame):
    grouped = (
        per_sample_df.groupby(["model_label", "gt_step_count"], as_index=False)
        .agg(
            sample_count=("eval_index", "size"),
            mean_reason_step_count=("reason_step_count", "mean"),
            mean_effective_reasoning_step_count=("effective_reasoning_step_count", "mean"),
            mean_hallucination_ratio=("hallucination_ratio", "mean"),
            mean_tokens_per_reason_step=("avg_tokens_per_reason_step", "mean"),
        )
        .sort_values(["model_label", "gt_step_count"])
    )
    return grouped


def ordered_model_labels(values):
    present = list(dict.fromkeys(values))
    ordered = [label for label in MODEL_PLOT_ORDER if label in present]
    ordered.extend(label for label in present if label not in ordered)
    return ordered


def plot_grouped_summary_by_model(grouped_df: pd.DataFrame, distribution_df: pd.DataFrame, output_path: Path):
    model_labels = ordered_model_labels(grouped_df["model_label"].tolist())
    n_rows = 2
    n_cols = 3

    left_axis_max = float(
        max(
            grouped_df["mean_reason_step_count"].max(),
            grouped_df["mean_effective_reasoning_step_count"].max(),
        )
    )
    left_axis_min = float(
        min(
            grouped_df["mean_reason_step_count"].min(),
            grouped_df["mean_effective_reasoning_step_count"].min(),
        )
    )
    left_axis_min = min(0.0, left_axis_min)
    left_axis_padding = max(0.5, (left_axis_max - left_axis_min) * 0.08)
    shared_left_ylim = (left_axis_min, left_axis_max + left_axis_padding)

    right_axis_max = float(grouped_df["mean_hallucination_ratio"].max())
    right_axis_padding = max(0.02, right_axis_max * 0.12)
    shared_right_ylim = (0.0, right_axis_max + right_axis_padding)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9), sharex=False)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    dist_axis = axes[0]
    dist_axis.bar(distribution_df["gt_step_count"], distribution_df["sample_count"], color="#4c78a8")
    dist_axis.set_title("GT Step Distribution")
    dist_axis.set_xlabel("GT reference step count")
    dist_axis.set_ylabel("Sample count")
    dist_axis.grid(axis="y", alpha=0.3)

    model_axes = axes[1:]
    for axis, model_label in zip(model_axes, model_labels):
        subset = grouped_df[grouped_df["model_label"] == model_label].sort_values("gt_step_count")
        mean_tokens_per_reason_step = subset["mean_tokens_per_reason_step"].mean()
        axis.plot(
            subset["gt_step_count"],
            subset["mean_reason_step_count"],
            marker="o",
            linewidth=2,
            label="Reason Step",
            color="#1f77b4",
        )
        axis.plot(
            subset["gt_step_count"],
            subset["mean_effective_reasoning_step_count"],
            marker="s",
            linewidth=2,
            label="Effective Reason Step",
            color="#ff7f0e",
        )
        axis.set_title(MODEL_PLOT_TITLE_MAP.get(model_label, model_label))
        axis.set_ylabel("Mean step count")
        axis.set_ylim(*shared_left_ylim)
        axis.grid(alpha=0.3)

        axis_right = axis.twinx()
        axis_right.plot(
            subset["gt_step_count"],
            subset["mean_hallucination_ratio"],
            marker="^",
            linewidth=2,
            linestyle="--",
            label="Hallucination Ratio",
            color="#2ca02c",
        )
        axis_right.set_ylabel("Mean hallucination ratio")
        axis_right.set_ylim(*shared_right_ylim)

        axis.text(
            0.98,
            0.98,
            f"Avg Tok/Step: {mean_tokens_per_reason_step:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
        )

        left_handles, left_labels = axis.get_legend_handles_labels()
        right_handles, right_labels = axis_right.get_legend_handles_labels()
        axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="best",
            fontsize=8,
        )

    for axis in model_axes[len(model_labels):]:
        axis.remove()

    for axis in model_axes[:len(model_labels)]:
        axis.set_xlabel("GT reference step count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grouped_summary_by_metric(grouped_df: pd.DataFrame, output_path: Path):
    metric_specs = [
        ("mean_reason_step_count", "Reason Step", "Mean atomic reasoning step count"),
        ("mean_effective_reasoning_step_count", "Effective Reason Step", "Mean effective reasoning step count"),
        ("mean_hallucination_ratio", "Hallucination Ratio", "Mean hallucination ratio"),
    ]

    model_labels = ordered_model_labels(grouped_df["model_label"].tolist())
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(12, 14), sharex=True)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for axis, (metric_key, title, ylabel) in zip(axes, metric_specs):
        for idx, model_label in enumerate(model_labels):
            subset = grouped_df[grouped_df["model_label"] == model_label].sort_values("gt_step_count")
            axis.plot(
                subset["gt_step_count"],
                subset[metric_key],
                marker="o",
                linewidth=2,
                label=model_label,
                color=palette[idx % len(palette)],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("GT reference step count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_gt_step_distribution(per_sample_df: pd.DataFrame):
    distribution = (
        per_sample_df[["eval_index", "gt_step_count"]]
        .drop_duplicates()
        .groupby("gt_step_count", as_index=False)
        .agg(sample_count=("eval_index", "size"))
        .sort_values("gt_step_count")
    )
    return distribution


def build_sample_feature_df(samples):
    rows = []
    for sample in samples:
        pred_think_token_count = short_cot_qwen.estimate_tokens(sample.get("pred_think", ""))
        rows.append(
            {
                "model_label": sample["model_label"],
                "eval_index": sample["eval_index"],
                "pred_think_token_count": pred_think_token_count,
            }
        )
    return pd.DataFrame(rows)


def plot_gt_step_distribution(distribution_df: pd.DataFrame, output_path: Path):
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(distribution_df["gt_step_count"], distribution_df["sample_count"], color="#4c78a8")
    axis.set_title("MathVista GT Step Distribution")
    axis.set_xlabel("GT reference step count")
    axis.set_ylabel("Sample count")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_records = load_gt_records(args.gt_parquet)

    samples = []
    for model_spec in DEFAULT_MODEL_SPECS:
        if not model_spec.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {model_spec.prediction_path}")
        if not model_spec.extract_pkl_path.exists():
            raise FileNotFoundError(f"Extract PKL not found: {model_spec.extract_pkl_path}")
        samples.extend(load_model_samples(model_spec, gt_records, limit=args.limit))

    if not samples:
        raise ValueError("No overlapping samples found between benchmark predictions and GT parquet.")

    sample_feature_df = build_sample_feature_df(samples)

    cache_path = args.output_dir / CACHE_FILENAME
    cache_data = load_cache(cache_path, ignore_cache=args.ignore_cache)
    pending_count = 0
    for sample in samples:
        cache_key = f"{sample['model_label']}::{sample['eval_index']}"
        if cache_key not in cache_data:
            pending_count += 1

    analyzer = None
    if pending_count > 0:
        analyzer = StepBonusAnalyzer(
            judge_backend=args.judge_backend,
            reward_router_address=args.reward_router_address,
            judge_model=args.judge_model,
            max_concurrency=args.max_concurrency,
        )

    per_sample_df = asyncio.run(
        run_analysis(samples, analyzer, cache_path=cache_path, ignore_cache=args.ignore_cache)
    ).sort_values(["model_label", "eval_index"])
    per_sample_df = per_sample_df.merge(sample_feature_df, on=["model_label", "eval_index"], how="left")
    per_sample_df["avg_tokens_per_reason_step"] = float("nan")
    valid_reason_mask = per_sample_df["reason_step_count"].fillna(0) > 0
    per_sample_df.loc[valid_reason_mask, "avg_tokens_per_reason_step"] = (
        per_sample_df.loc[valid_reason_mask, "pred_think_token_count"]
        / per_sample_df.loc[valid_reason_mask, "reason_step_count"]
    )
    per_sample_df = apply_gt_step_bucketing(per_sample_df)

    grouped_df = build_grouped_summary(per_sample_df)
    gt_distribution_df = build_gt_step_distribution(per_sample_df)

    per_sample_path = args.output_dir / "mathvista_step_bonus_per_sample.csv"
    grouped_path = args.output_dir / "mathvista_step_bonus_grouped_by_gt_steps.csv"
    gt_distribution_path = args.output_dir / "mathvista_gt_step_distribution.csv"
    figure_path = args.output_dir / "mathvista_step_bonus_grouped_by_model.png"
    metric_figure_path = args.output_dir / "mathvista_step_bonus_grouped_by_metric.png"
    gt_distribution_figure_path = args.output_dir / "mathvista_gt_step_distribution.png"

    per_sample_df.to_csv(per_sample_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)
    gt_distribution_df.to_csv(gt_distribution_path, index=False)
    plot_grouped_summary_by_model(grouped_df, gt_distribution_df, figure_path)
    plot_grouped_summary_by_metric(grouped_df, metric_figure_path)
    plot_gt_step_distribution(gt_distribution_df, gt_distribution_figure_path)

    print(f"Saved per-sample metrics to: {per_sample_path}")
    print(f"Saved grouped metrics to: {grouped_path}")
    print(f"Saved GT step distribution to: {gt_distribution_path}")
    print(f"Saved model-grouped plot to: {figure_path}")
    print(f"Saved metric-grouped plot to: {metric_figure_path}")
    print(f"Saved GT step distribution plot to: {gt_distribution_figure_path}")


if __name__ == "__main__":
    main()
