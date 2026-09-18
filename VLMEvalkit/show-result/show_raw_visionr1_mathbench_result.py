#!/usr/bin/env python3
"""Summarize Raw-VisionR1 SFT evaluation results across MathVista/MathVerse/LogicVista."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd


MODEL_NAME = "Qwen3-VL-8B-Thinking-VisionR1-SFT-2000it_Raw_Data"
DEFAULT_ROOT = Path(os.environ.get("VLMEVALKIT_QWEN3VL_RESULT_ROOT", "./Qwen3-VL"))
OUTPUT_FILE = "raw_visionr1_mathbench_summary.csv"
DATASET_NAME = "Raw-VisionR1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show MathVista/MathVerse/LogicVista summary for Raw-VisionR1 SFT checkpoints."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing {XX}it checkpoint folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_FILE),
        help="CSV path used to save the summary table.",
    )
    return parser.parse_args()


def score_to_display(value: float) -> float:
    value = float(value)
    if -1.0 <= value <= 1.0:
        value *= 100.0
    return round(value, 2)


def parse_common_benchmark(filepath: Path) -> float | None:
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    df.columns = [str(col).strip() for col in df.columns]

    acc_col = next((col for col in df.columns if col.lower() == "acc"), None)
    if acc_col:
        first_col = df.columns[0]
        overall_rows = df[df[first_col].astype(str).str.contains("Overall", case=False, na=False)]
        if not overall_rows.empty:
            return score_to_display(overall_rows.iloc[0][acc_col])

    overall_col = next((col for col in df.columns if col.lower() == "overall"), None)
    if overall_col and not df.empty:
        return score_to_display(df.iloc[0][overall_col])

    return None


def parse_mathverse(filepath: Path) -> float | None:
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    df.columns = [str(col).strip() for col in df.columns]
    if "Overall" not in df.columns or df.empty:
        return None

    target_splits = [
        "Text Dominant",
        "Vision Intensive",
        "Vision Only",
        "Text Lite",
        "Vision Dominant",
    ]
    first_col = df.columns[0]
    filtered = df[df[first_col].isin(target_splits)]
    score = filtered["Overall"].mean() if not filtered.empty else df["Overall"].mean()
    return score_to_display(score)


def find_single_score_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(Path(path) for path in glob.glob(str(folder / pattern)))
    return matches[0] if matches else None


def collect_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for checkpoint_dir in sorted(root.glob("*it")):
        if not checkpoint_dir.is_dir():
            continue

        match = re.fullmatch(r"(\d+)it", checkpoint_dir.name)
        if match is None:
            continue

        iteration = int(match.group(1))
        model_dir = checkpoint_dir / MODEL_NAME
        if not model_dir.is_dir():
            continue

        mathvista_file = find_single_score_file(model_dir, f"{MODEL_NAME}_MathVista*_score.csv")
        mathverse_file = find_single_score_file(model_dir, f"{MODEL_NAME}_MathVerse*_score.csv")
        logicvista_file = find_single_score_file(model_dir, f"{MODEL_NAME}_LogicVista*_score.csv")

        mathvista = parse_common_benchmark(mathvista_file) if mathvista_file else None
        mathverse = parse_mathverse(mathverse_file) if mathverse_file else None
        logicvista = parse_common_benchmark(logicvista_file) if logicvista_file else None

        scores = [score for score in [mathvista, mathverse, logicvista] if score is not None]
        average = round(sum(scores) / len(scores), 2) if scores else None

        rows.append(
            {
                "SFT Iteration": iteration,
                "Dataset": DATASET_NAME,
                "MathVista": mathvista if mathvista is not None else "-",
                "MathVerse": mathverse if mathverse is not None else "-",
                "LogicVista": logicvista if logicvista is not None else "-",
                "AVG": average if average is not None else "-",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("SFT Iteration").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    df = collect_results(args.root)

    if df.empty:
        print("No matching benchmark result files found.")
        return

    print("\n" + "=" * 96)
    print("Raw-VisionR1 MathBench Summary")
    print("=" * 96)
    print(df.to_string(index=False, col_space=12, justify="center"))
    print("=" * 96 + "\n")

    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
