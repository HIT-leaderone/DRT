import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

import pandas as pd


VERL_ROOT = Path(".")
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))


SCRIPT_PATHS = {
    "mathvista": Path("./analyze_mathvista_step_bonus.py"),
    "mathverse": Path("./analyze_mathverse_step_bonus.py"),
    "logicvista": Path("./analyze_logicvista_step_bonus.py"),
}

PREFIXES = {
    "mathvista": "mathvista",
    "mathverse": "mathverse",
    "logicvista": "logicvista",
}


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun a step-bonus benchmark for selected model labels and merge them back into full outputs."
    )
    parser.add_argument("--benchmark", choices=sorted(SCRIPT_PATHS), required=True)
    parser.add_argument("--model_labels", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./Step-level-LLM-judge/assets"),
    )
    parser.add_argument("--gt_parquet", type=Path, default=None)
    parser.add_argument("--judge_backend", choices=["openai", "router"], default="openai")
    parser.add_argument("--reward_router_address", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="auto")
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--ignore_cache", action="store_true")
    return parser.parse_args()


def parse_model_labels(text):
    return {item.strip() for item in str(text).split(",") if item.strip()}


def build_selected_samples(module, base_module, selected_labels, gt_records):
    selected_specs = [spec for spec in module.DEFAULT_MODEL_SPECS if spec.label in selected_labels]
    if not selected_specs:
        raise ValueError(f"No model specs matched labels: {sorted(selected_labels)}")

    samples = []
    for model_spec in selected_specs:
        if not model_spec.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {model_spec.prediction_path}")
        if not model_spec.extract_pkl_path.exists():
            raise FileNotFoundError(f"Extract PKL not found: {model_spec.extract_pkl_path}")
        samples.extend(base_module.load_model_samples(model_spec, gt_records, limit=None))
    return samples


def enrich_selected_per_sample_df(module, per_sample_df, sample_feature_df):
    df = per_sample_df.sort_values(["model_label", "eval_index"]).copy()
    df = df.merge(sample_feature_df, on=["model_label", "eval_index"], how="left")
    df["avg_tokens_per_reason_step"] = float("nan")
    valid_reason_mask = df["reason_step_count"].fillna(0) > 0
    df.loc[valid_reason_mask, "avg_tokens_per_reason_step"] = (
        df.loc[valid_reason_mask, "pred_think_token_count"] / df.loc[valid_reason_mask, "reason_step_count"]
    )
    if hasattr(module, "apply_gt_step_bucketing"):
        df = module.apply_gt_step_bucketing(df)
    elif hasattr(module, "apply_gt_step_identity"):
        df = module.apply_gt_step_identity(df)
    return df


def main():
    args = parse_args()
    module = load_module(SCRIPT_PATHS[args.benchmark])
    base_module = getattr(module, "base", module)
    selected_labels = parse_model_labels(args.model_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_parquet = args.gt_parquet or module.DEFAULT_GT_PARQUET
    gt_records = base_module.load_gt_records(gt_parquet)
    selected_samples = build_selected_samples(module, base_module, selected_labels, gt_records)
    sample_feature_df = base_module.build_sample_feature_df(selected_samples)

    cache_path = args.output_dir / module.CACHE_FILENAME
    if args.ignore_cache and cache_path.exists():
        cache_data = base_module.load_cache(cache_path, ignore_cache=False)
        cache_data = {
            key: value
            for key, value in cache_data.items()
            if key.split("::", 1)[0] not in selected_labels
        }
        base_module.save_cache(cache_path, cache_data)
    cache_data = base_module.load_cache(cache_path, ignore_cache=False)
    pending_count = 0
    for sample in selected_samples:
        cache_key = f"{sample['model_label']}::{sample['eval_index']}"
        if cache_key not in cache_data:
            pending_count += 1

    print(f"Benchmark: {args.benchmark}")
    print(f"Selected labels: {sorted(selected_labels)}")
    print(f"Selected samples: {len(selected_samples)}")
    print(f"Pending cache entries for selected labels: {pending_count}")

    analyzer = None
    if pending_count > 0:
        analyzer = base_module.StepBonusAnalyzer(
            judge_backend=args.judge_backend,
            reward_router_address=args.reward_router_address,
            judge_model=args.judge_model,
            max_concurrency=args.max_concurrency,
        )

    selected_per_sample_df = asyncio.run(
        base_module.run_analysis(
            selected_samples,
            analyzer,
            cache_path=cache_path,
            ignore_cache=False,
        )
    )
    selected_per_sample_df = enrich_selected_per_sample_df(module, selected_per_sample_df, sample_feature_df)

    prefix = PREFIXES[args.benchmark]
    per_sample_path = args.output_dir / f"{prefix}_step_bonus_per_sample.csv"
    grouped_path = args.output_dir / f"{prefix}_step_bonus_grouped_by_gt_steps.csv"
    gt_distribution_path = args.output_dir / f"{prefix}_gt_step_distribution.csv"
    figure_path = args.output_dir / f"{prefix}_step_bonus_grouped_by_model.png"
    metric_figure_path = args.output_dir / f"{prefix}_step_bonus_grouped_by_metric.png"
    gt_distribution_figure_path = args.output_dir / f"{prefix}_gt_step_distribution.png"

    if per_sample_path.exists():
        existing_df = pd.read_csv(per_sample_path)
        existing_df = existing_df[~existing_df["model_label"].isin(selected_labels)].copy()
        full_per_sample_df = pd.concat([existing_df, selected_per_sample_df], ignore_index=True)
    else:
        full_per_sample_df = selected_per_sample_df.copy()

    full_per_sample_df = full_per_sample_df.sort_values(["model_label", "eval_index"])
    grouped_df = base_module.build_grouped_summary(full_per_sample_df)
    gt_distribution_df = base_module.build_gt_step_distribution(full_per_sample_df)

    full_per_sample_df.to_csv(per_sample_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)
    gt_distribution_df.to_csv(gt_distribution_path, index=False)
    base_module.plot_grouped_summary_by_model(grouped_df, gt_distribution_df, figure_path)
    base_module.plot_grouped_summary_by_metric(grouped_df, metric_figure_path)
    module.plot_gt_step_distribution(gt_distribution_df, gt_distribution_figure_path)

    print(f"Saved merged per-sample metrics to: {per_sample_path}")
    print(f"Saved merged grouped metrics to: {grouped_path}")
    print(f"Saved merged GT step distribution to: {gt_distribution_path}")
    print(f"Saved merged model-grouped plot to: {figure_path}")
    print(f"Saved merged metric-grouped plot to: {metric_figure_path}")
    print(f"Saved merged GT step distribution plot to: {gt_distribution_figure_path}")


if __name__ == "__main__":
    main()
