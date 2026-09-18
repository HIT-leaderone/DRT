import argparse
import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERL_ROOT = Path(".")
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
_mathvista_spec = importlib.util.spec_from_file_location(
    "analyze_mathvista_step_bonus_base",
    SCRIPT_DIR / "analyze_mathvista_step_bonus.py",
)
base = importlib.util.module_from_spec(_mathvista_spec)
_mathvista_spec.loader.exec_module(base)


DEFAULT_GT_PARQUET = Path("./data/mathverse/testmini.parquet")
DEFAULT_OUTPUT_DIR = Path("./fig")
ANALYSIS_VERSION = "mathverse_separate_prompts_v2"
CACHE_FILENAME = f"{ANALYSIS_VERSION}_cache.json"


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
            "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVerse_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
            "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
        ),
    ),
    ModelSpec(
        label="w/o process reward",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVerse_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
        ),
    ),
    ModelSpec(
        label="w/o step bouns",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVerse_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
        ),
    ),
    ModelSpec(
        label="DRT-RL",
        prediction_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVerse_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
        ),
    ),
    ModelSpec(
        label="DRT-SFT-800it",
        prediction_path=Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVerse_MINI.xlsx"
        ),
        extract_pkl_path=Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
        ),
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MathVerse benchmark outputs with the GPT step-level judge."
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


def apply_gt_step_identity(per_sample_df: pd.DataFrame):
    df = per_sample_df.copy()
    if "original_gt_step_count" not in df.columns:
        df["original_gt_step_count"] = df["gt_step_count"]
    df["original_gt_step_count"] = pd.to_numeric(df["original_gt_step_count"], errors="coerce")
    df = df[df["original_gt_step_count"].notna()].copy()
    df["original_gt_step_count"] = df["original_gt_step_count"].astype(int)
    df["gt_step_count"] = df["original_gt_step_count"]
    return df


def plot_gt_step_distribution(distribution_df: pd.DataFrame, output_path: Path):
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(distribution_df["gt_step_count"], distribution_df["sample_count"], color="#4c78a8")
    axis.set_title("MathVerse GT Step Distribution")
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

    gt_records = base.load_gt_records(args.gt_parquet)

    samples = []
    for model_spec in DEFAULT_MODEL_SPECS:
        if not model_spec.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {model_spec.prediction_path}")
        if not model_spec.extract_pkl_path.exists():
            raise FileNotFoundError(f"Extract PKL not found: {model_spec.extract_pkl_path}")
        samples.extend(base.load_model_samples(model_spec, gt_records, limit=args.limit))

    if not samples:
        raise ValueError("No overlapping samples found between benchmark predictions and GT parquet.")

    sample_feature_df = base.build_sample_feature_df(samples)

    cache_path = args.output_dir / CACHE_FILENAME
    cache_data = base.load_cache(cache_path, ignore_cache=args.ignore_cache)
    pending_count = 0
    for sample in samples:
        cache_key = f"{sample['model_label']}::{sample['eval_index']}"
        if cache_key not in cache_data:
            pending_count += 1

    analyzer = None
    if pending_count > 0:
        analyzer = base.StepBonusAnalyzer(
            judge_backend=args.judge_backend,
            reward_router_address=args.reward_router_address,
            judge_model=args.judge_model,
            max_concurrency=args.max_concurrency,
        )

    per_sample_df = asyncio.run(
        base.run_analysis(samples, analyzer, cache_path=cache_path, ignore_cache=args.ignore_cache)
    ).sort_values(["model_label", "eval_index"])
    per_sample_df = per_sample_df.merge(sample_feature_df, on=["model_label", "eval_index"], how="left")
    per_sample_df["avg_tokens_per_reason_step"] = float("nan")
    valid_reason_mask = per_sample_df["reason_step_count"].fillna(0) > 0
    per_sample_df.loc[valid_reason_mask, "avg_tokens_per_reason_step"] = (
        per_sample_df.loc[valid_reason_mask, "pred_think_token_count"]
        / per_sample_df.loc[valid_reason_mask, "reason_step_count"]
    )
    per_sample_df = apply_gt_step_identity(per_sample_df)

    grouped_df = base.build_grouped_summary(per_sample_df)
    gt_distribution_df = base.build_gt_step_distribution(per_sample_df)

    per_sample_path = args.output_dir / "mathverse_step_bonus_per_sample.csv"
    grouped_path = args.output_dir / "mathverse_step_bonus_grouped_by_gt_steps.csv"
    gt_distribution_path = args.output_dir / "mathverse_gt_step_distribution.csv"
    figure_path = args.output_dir / "mathverse_step_bonus_grouped_by_model.png"
    metric_figure_path = args.output_dir / "mathverse_step_bonus_grouped_by_metric.png"
    gt_distribution_figure_path = args.output_dir / "mathverse_gt_step_distribution.png"

    per_sample_df.to_csv(per_sample_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)
    gt_distribution_df.to_csv(gt_distribution_path, index=False)
    base.plot_grouped_summary_by_model(grouped_df, gt_distribution_df, figure_path)
    base.plot_grouped_summary_by_metric(grouped_df, metric_figure_path)
    plot_gt_step_distribution(gt_distribution_df, gt_distribution_figure_path)

    print(f"Saved per-sample metrics to: {per_sample_path}")
    print(f"Saved grouped metrics to: {grouped_path}")
    print(f"Saved GT step distribution to: {gt_distribution_path}")
    print(f"Saved model-grouped plot to: {figure_path}")
    print(f"Saved metric-grouped plot to: {metric_figure_path}")
    print(f"Saved GT step distribution plot to: {gt_distribution_figure_path}")


if __name__ == "__main__":
    main()
