import math
import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results"
RESULTS_ROOT = Path(os.environ.get("TIME_EVAL_RESULTS_ROOT", SCRIPT_DIR / "results"))
VLLM0191_TF4571_ROOT = RESULTS_ROOT / "vllm0191_tf4571"
MODE = "parallel"

METHODS = [
    {"label": "Qwen-DA", "models": ["Qwen3-VL-8B-Instruct", "Qwen3-8B-Instruct"], "prompt": "directly-answer"},
    {"label": "Qwen-DRT", "models": ["Qwen3-VL-8B-Instruct", "Qwen3-8B-Instruct"], "prompt": "short-cot-image"},
    {"label": "DRT-SFT", "models": ["DRT-SFT"], "prompt": "short-cot-image"},
    {"label": "DRT-RL", "models": ["DRT-RL"], "prompt": "short-cot-image"},
    {"label": "VisionThink", "models": ["VisionThink-Efficient"], "prompt": "visionthink"},
    {"label": "Qwen-Standard", "models": ["Qwen3-VL-8B-Instruct", "Qwen3-8B-Instruct"], "prompt": "custom-prompt"},
    {"label": "Qwen-Thinking", "models": ["Qwen3-VL-8B-Thinking"], "prompt": "custom-prompt", "root": VLLM0191_TF4571_ROOT},
    {"label": "CoD", "models": ["Qwen3-VL-8B-Instruct", "Qwen3-8B-Instruct"], "prompt": "cod"},
    {"label": "Thinkless", "models": ["Qwen3-VL-8B-Instruct", "Qwen3-8B-Instruct"], "prompt": "thinkless"},
]

BENCHMARKS = [
    {
        "label": "MathVista",
        "latex": r"\textbf{MathVista~\cite{lu2024mathvista}}",
        "dataset_name": "MathVista_MINI",
    },
    {
        "label": "MathVerse",
        "latex": r"\textbf{MathVerse~\cite{zhang2024mathverse}}",
        "dataset_name": "MathVerse_MINI",
    },
    {
        "label": "LogicVista",
        "latex": r"\textbf{LogicVista~\cite{xiao2024logicvista}}",
        "dataset_name": "LogicVista",
    },
    {
        "label": "GSM8K",
        "latex": r"\textbf{GSM8K~\cite{cobbe2021training}}",
        "dataset_name": "GSM8K",
    },
    {
        "label": "Video-Holmes",
        "latex": r"\textbf{Video-Holmes~\cite{cheng2025video_holmes}}",
        "dataset_name": "Video_Holmes",
    },
]


PROMPT_DIR_CANDIDATES = {
    "directly-answer": ["directly-answer", "Directly-Answer"],
    "short-cot-image": ["short-cot-image", "Short-COT-Image"],
    "custom-prompt": ["custom-prompt", "Custom-Prompt"],
    "cod": ["cod", "CoD"],
    "thinkless": ["thinkless", "Thinkless"],
    "visionthink": ["visionthink", "VisionThink"],
}


def _summary_paths(models, prompt: str, mode: str = MODE, root: Path = RESULTS_ROOT) -> list[Path]:
    if isinstance(models, str):
        models = [models]
    dir_candidates = PROMPT_DIR_CANDIDATES.get(prompt, [prompt])
    file_name = f"summary_{prompt}_{mode}.xlsx"
    for model in models:
        mode_root = root / model / mode
        if not mode_root.exists():
            continue

        paths = []
        seen = set()
        for prompt_dir in dir_candidates:
            base_path = mode_root / prompt_dir / file_name
            if base_path.exists() and base_path not in seen:
                paths.append(base_path)
                seen.add(base_path)

            override_dirs = sorted(
                (
                    path for path in mode_root.iterdir()
                    if path.is_dir() and path.name.startswith(f"{prompt_dir}-")
                ),
                key=lambda path: (path.stat().st_mtime, path.name),
            )
            for override_dir in override_dirs:
                override_path = override_dir / file_name
                if override_path.exists() and override_path not in seen:
                    paths.append(override_path)
                    seen.add(override_path)

        if paths:
            return paths
    return []


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _format_num(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _extract_dataset_metrics(summary_paths: list[Path]):
    if not summary_paths:
        return {}, "missing summary"

    metrics = {}
    warnings = []
    for summary_path in summary_paths:
        try:
            df = pd.read_excel(summary_path)
        except Exception as exc:
            warnings.append(f"failed to read summary: {summary_path} ({exc})")
            continue

        for _, row in df.iterrows():
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue
            output_tokens = _safe_float(row.get("avg_output_length_tokens"))
            time_per_sample_wall = _safe_float(row.get("avg_time_per_sample_sec_wall"))
            item_latency = _safe_float(row.get("avg_item_latency_sec"))
            model_exec_latency = _safe_float(row.get("avg_model_exec_sec"))
            qps = None
            if time_per_sample_wall not in (None, 0):
                qps = 1.0 / time_per_sample_wall
            metrics[dataset] = {
                "output_tokens": output_tokens,
                "qps": qps,
                "avg_latency": model_exec_latency if model_exec_latency is not None else item_latency,
            }
    return metrics, "; ".join(warnings) if warnings else None


def _mean_metric(entries, key):
    values = [item[key] for item in entries if item.get(key) is not None and not math.isnan(item[key])]
    if not values:
        return None
    return sum(values) / len(values)


def collect_results():
    rows = []
    warnings = []

    for method in METHODS:
        summary_paths = _summary_paths(method["models"], method["prompt"], root=method.get("root", RESULTS_ROOT))
        metrics_by_dataset, warning = _extract_dataset_metrics(summary_paths)
        if warning:
            warnings.append({"method": method["label"], "warning": warning})

        row = {"Method": method["label"]}
        bench_metric_values = []

        for bench in BENCHMARKS:
            dataset_name = bench["dataset_name"]
            item = metrics_by_dataset.get(dataset_name)
            if item is None:
                row[f"{bench['label']}_tokens"] = "-"
                row[f"{bench['label']}_qps"] = "-"
                row[f"{bench['label']}_latency"] = "-"
                continue

            row[f"{bench['label']}_tokens"] = _format_num(item["output_tokens"])
            row[f"{bench['label']}_qps"] = _format_num(item["qps"])
            row[f"{bench['label']}_latency"] = _format_num(item["avg_latency"])
            bench_metric_values.append(item)

        total_metrics = metrics_by_dataset.get("TOTAL")
        if total_metrics is not None:
            row["AVG_tokens"] = _format_num(total_metrics.get("output_tokens"))
            row["AVG_qps"] = _format_num(total_metrics.get("qps"))
            row["AVG_latency"] = _format_num(total_metrics.get("avg_latency"))
        else:
            row["AVG_tokens"] = _format_num(_mean_metric(bench_metric_values, "output_tokens"))
            row["AVG_qps"] = _format_num(_mean_metric(bench_metric_values, "qps"))
            row["AVG_latency"] = _format_num(_mean_metric(bench_metric_values, "avg_latency"))
        rows.append(row)

    return rows, warnings


def build_latex_table(df: pd.DataFrame) -> str:
    header_top = (
        r"\multirow{2}{*}{\textbf{Method}}"
        + "".join(
            f"\n& \\multicolumn{{3}}{{c}}{{{bench['latex']}}}" for bench in BENCHMARKS
        )
        + "\n& \\multicolumn{3}{c}{\\textbf{AVG.}} \\\\"
    )
    header_bottom = (
        " "
        + "".join("\n& \\textbf{Tok.} & \\textbf{QPS} & \\textbf{Lat.}" for _ in BENCHMARKS)
        + "\n& \\textbf{Tok.} & \\textbf{QPS} & \\textbf{Lat.} \\\\"
    )

    body_lines = []
    for _, row in df.iterrows():
        parts = [str(row["Method"])]
        for bench in BENCHMARKS:
            parts.extend([
                str(row[f"{bench['label']}_tokens"]),
                str(row[f"{bench['label']}_qps"]),
                str(row[f"{bench['label']}_latency"]),
            ])
        parts.extend([
            str(row["AVG_tokens"]),
            str(row["AVG_qps"]),
            str(row["AVG_latency"]),
        ])
        body_lines.append(" & ".join(parts) + r" \\")

    return "\n".join([header_top, header_bottom, r"\midrule"] + body_lines)


def build_two_line_terminal_table(df: pd.DataFrame) -> str:
    subheaders = ["Method"]
    for _ in BENCHMARKS:
        subheaders.extend(["Tok", "QPS", "Lat"])
    subheaders.extend(["Tok", "QPS", "Lat"])

    rows = []
    for _, row in df.iterrows():
        current = [str(row["Method"])]
        for bench in BENCHMARKS:
            current.extend([
                str(row[f"{bench['label']}_tokens"]),
                str(row[f"{bench['label']}_qps"]),
                str(row[f"{bench['label']}_latency"]),
            ])
        current.extend([
            str(row["AVG_tokens"]),
            str(row["AVG_qps"]),
            str(row["AVG_latency"]),
        ])
        rows.append(current)

    all_rows = [subheaders] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(subheaders))]

    def fmt_regular(row_values):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row_values))

    spans = [("Method", 1)] + [(bench["label"], 3) for bench in BENCHMARKS] + [("AVG.", 3)]

    top_cells = []
    cursor = 0
    for title, span_cols in spans:
        span_width = sum(widths[cursor:cursor + span_cols]) + 3 * (span_cols - 1)
        top_cells.append(title.center(span_width))
        cursor += span_cols
    top_line = " | ".join(top_cells)

    separator = "-+-".join("-" * w for w in widths)
    lines = [top_line, fmt_regular(subheaders), separator]
    lines.extend(fmt_regular(r) for r in rows)
    return "\n".join(lines)


def main():
    rows, warnings = collect_results()
    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_csv = OUTPUT_DIR / "time_eval_summary_methods.csv"
    df.to_csv(wide_csv, index=False)

    latex_path = OUTPUT_DIR / "time_eval_table.tex"
    latex_path.write_text(build_latex_table(df), encoding="utf-8")

    print("=" * 80)
    print("Time Eval Method Summary")
    print("=" * 80)
    print(build_two_line_terminal_table(df))
    print(f"\nSaved wide summary to {wide_csv}")
    print(f"Saved LaTeX table to {latex_path}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item['method']}: {item['warning']}")


if __name__ == "__main__":
    main()
