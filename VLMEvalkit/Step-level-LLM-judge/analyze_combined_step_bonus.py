import argparse
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent
VERL_ROOT = Path(os.environ.get("VERL_QWEN3VL_ROOT", "."))
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

BASE_MODULE_PATH = REPO_ROOT / "analyze_mathvista_step_bonus.py"

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


def _ordered_model_labels(values):
    present = list(dict.fromkeys(values))
    ordered = [label for label in MODEL_PLOT_ORDER if label in present]
    ordered.extend(label for label in present if label not in ordered)
    return ordered


def _load_base_module():
    try:
        base_spec = importlib.util.spec_from_file_location(
            "analyze_mathvista_step_bonus_base",
            BASE_MODULE_PATH,
        )
        module = importlib.util.module_from_spec(base_spec)
        base_spec.loader.exec_module(module)
        return module
    except Exception:
        short_cot_path = VERL_ROOT / "verl" / "utils" / "reward_score" / "short_cot_qwen.py"
        short_cot_spec = importlib.util.spec_from_file_location(
            "short_cot_qwen_direct",
            short_cot_path,
        )
        short_cot_qwen = importlib.util.module_from_spec(short_cot_spec)
        short_cot_spec.loader.exec_module(short_cot_qwen)
        return types.SimpleNamespace(
            short_cot_qwen=short_cot_qwen,
            MODEL_PLOT_TITLE_MAP=MODEL_PLOT_TITLE_MAP,
            ordered_model_labels=_ordered_model_labels,
        )


base = _load_base_module()


DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets"
DEFAULT_BACKUP_DIR = Path("./Step-level-LLM-judge/assets")
DEFAULT_INPUT_DIR = VERL_ROOT / "fig"
FALLBACK_INPUT_DIR = DEFAULT_BACKUP_DIR


def default_input_path(filename: str) -> Path:
    primary = DEFAULT_INPUT_DIR / filename
    if primary.exists():
        return primary
    return FALLBACK_INPUT_DIR / filename


DEFAULT_INPUTS = {
    "mathvista": default_input_path("mathvista_step_bonus_per_sample.csv"),
    "mathverse": default_input_path("mathverse_step_bonus_per_sample.csv"),
    "logicvista": default_input_path("logicvista_step_bonus_per_sample.csv"),
}
OFFICIAL_SCORE_PATHS = {
    "mathverse": {
        "Qwen-Standard": Path(
            "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
            "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVerse_MINI_gpt-4.1-2025-04-14_score.pkl"
        ),
        "w/o process reward": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVerse_MINI_gpt-4.1-2025-04-14_score.pkl"
        ),
        "w/o step bouns": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVerse_MINI_gpt-4.1-2025-04-14_score.pkl"
        ),
        "DRT-RL": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVerse_MINI_gpt-4.1-2025-04-14_score.pkl"
        ),
        "DRT-SFT-800it": Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVerse_MINI_gpt-4.1-2025-04-14_score.pkl"
        ),
    },
    "logicvista": {
        "Qwen-Standard": Path(
            "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
            "T20260112_G624a0651/Qwen3-VL-8B-Instruct_LogicVista_gpt4.1.pkl"
        ),
        "w/o process reward": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
            "T20260406_G6d220c41/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_LogicVista_gpt4.1.pkl"
        ),
        "w/o step bouns": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_LogicVista_gpt4.1.pkl"
        ),
        "DRT-RL": Path(
            "./results/Short-COT-Image/170it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
            "T20260411_Gf7429a73/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_LogicVista_gpt4.1.pkl"
        ),
        "DRT-SFT-800it": Path(
            "./results/Short-COT-Image/800it/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
            "T20260409_Gc7c180bc/"
            "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_LogicVista_gpt4.1.pkl"
        ),
    },
}
MIN_GT_STEP = 3
MAX_GT_STEP = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a combined step-level eval view for MathVista, MathVerse, and LogicVista."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backup_dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--min_gt_step", type=int, default=MIN_GT_STEP)
    parser.add_argument("--max_gt_step", type=int, default=MAX_GT_STEP)
    return parser.parse_args()


def load_and_concat(inputs: dict[str, Path], min_gt_step: int, max_gt_step: int):
    frames = []
    for benchmark, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing per-sample CSV for {benchmark}: {path}")
        df = pd.read_csv(path)
        df["benchmark"] = benchmark
        df = df[(df["gt_step_count"] >= min_gt_step) & (df["gt_step_count"] <= max_gt_step)].copy()
        df["sample_uid"] = df["benchmark"].astype(str) + "::" + df["eval_index"].astype(str)
        frames.append(df)
    if not frames:
        raise ValueError("No input dataframes loaded.")
    return pd.concat(frames, ignore_index=True)


def build_combined_distribution(per_sample_df: pd.DataFrame):
    distribution = (
        per_sample_df[["sample_uid", "gt_step_count"]]
        .drop_duplicates()
        .groupby("gt_step_count", as_index=False)
        .agg(sample_count=("sample_uid", "size"))
        .sort_values("gt_step_count")
    )
    return distribution


def load_official_accuracy_maps():
    import pickle

    result = {}
    for benchmark, model_map in OFFICIAL_SCORE_PATHS.items():
        result[benchmark] = {}
        for model_label, path in model_map.items():
            if not path.exists():
                continue
            with open(path, "rb") as f:
                obj = pickle.load(f)
            acc_map = {}
            if benchmark == "mathverse":
                for key, value in obj.items():
                    if isinstance(value, dict):
                        acc_map[int(key)] = 1.0 if bool(value.get("score", False)) else 0.0
            elif benchmark == "logicvista":
                for key, value in obj.items():
                    if isinstance(value, dict):
                        acc_map[int(key)] = 1.0 if int(value.get("hit", 0)) == 1 else 0.0
            result[benchmark][model_label] = acc_map
    return result


def resolve_eval_answer(row: pd.Series) -> str:
    candidate = row.get("judge_extract_answer", "")
    if pd.notna(candidate) and str(candidate).strip():
        return str(candidate).strip()
    candidate = row.get("pred_final", "")
    if pd.notna(candidate):
        return str(candidate).strip()
    return ""


def compute_accuracy_column(per_sample_df: pd.DataFrame):
    df = per_sample_df.copy()
    official_maps = load_official_accuracy_maps()
    resolved_answers = []
    correctness = []
    for _, row in df.iterrows():
        pred = resolve_eval_answer(row)
        gt = str(row.get("gt_answer", "") or "").strip()
        pred_clean = base.short_cot_qwen.clean_final_answer(pred)
        gt_clean = base.short_cot_qwen.clean_final_answer(gt)
        resolved_answers.append(pred_clean)
        benchmark = str(row.get("benchmark", "") or "")
        model_label = str(row.get("model_label", "") or "")
        eval_index = int(row.get("eval_index"))
        official = (
            official_maps.get(benchmark, {})
            .get(model_label, {})
            .get(eval_index)
        )
        if official is None:
            try:
                hit = bool(base.short_cot_qwen.grade_answer(pred_clean, gt_clean))
            except Exception:
                hit = pred_clean.strip().lower() == gt_clean.strip().lower()
        else:
            hit = bool(official)
        correctness.append(1.0 if hit else 0.0)
    df["resolved_eval_answer"] = resolved_answers
    df["is_correct"] = correctness
    return df


def build_grouped_summary_with_accuracy(per_sample_df: pd.DataFrame):
    grouped = (
        per_sample_df.groupby(["model_label", "gt_step_count"], as_index=False)
        .agg(
            sample_count=("sample_uid", "size"),
            mean_reason_step_count=("reason_step_count", "mean"),
            mean_effective_reasoning_step_count=("effective_reasoning_step_count", "mean"),
            mean_hallucination_ratio=("hallucination_ratio", "mean"),
            mean_tokens_per_reason_step=("avg_tokens_per_reason_step", "mean"),
            mean_accuracy=("is_correct", "mean"),
        )
        .sort_values(["model_label", "gt_step_count"])
    )
    return grouped


def plot_combined_gt_step_distribution(distribution_df: pd.DataFrame, output_path: Path):
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(distribution_df["gt_step_count"], distribution_df["sample_count"], color="#4c78a8")
    axis.set_title("Combined GT Step Distribution")
    axis.set_xlabel("GT reference step count")
    axis.set_ylabel("Sample count")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grouped_summary_by_metric_with_accuracy(
    grouped_df: pd.DataFrame,
    output_path: Path,
    n_rows: int = 1,
    n_cols: int = 4,
    figsize: tuple[float, float] = (24, 4.8),
):
    metric_specs = [
        ("mean_reason_step_count", "Reason Step", "Mean atomic reasoning step count"),
        ("mean_effective_reasoning_step_count", "Effective Reason Step", "Mean effective reasoning step count"),
        ("mean_hallucination_ratio", "Hallucination Ratio", "Mean hallucination ratio"),
        ("mean_accuracy", "Accuracy", "Mean accuracy"),
    ]

    model_labels = base.ordered_model_labels(grouped_df["model_label"].tolist())
    if n_rows * n_cols < len(metric_specs):
        raise ValueError("Metric subplot grid is too small for the number of metrics.")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for axis, (metric_key, title, ylabel) in zip(axes, metric_specs):
        metric_min = float(grouped_df[metric_key].min())
        metric_max = float(grouped_df[metric_key].max())
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
        if metric_key == "mean_accuracy":
            span = max(metric_max - metric_min, 0.08)
            pad = max(span * 0.12, 0.03)
            ymin = max(0.0, metric_min - pad)
            ymax = min(1.0, metric_max + pad)
            axis.set_ylim(ymin, ymax)
        if metric_key == "mean_hallucination_ratio":
            span = max(metric_max - metric_min, 0.08)
            pad = max(span * 0.12, 0.03)
            ymin = max(0.0, metric_min - pad)
            ymax = min(1.0, metric_max + pad)
            axis.set_ylim(ymin, ymax)
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=9)
        axis.set_xlabel("GT reference step count")

    for axis in axes[len(metric_specs):]:
        axis.remove()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grouped_summary_with_side_panels(
    grouped_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    output_path: Path,
):
    font_dir = REPO_ROOT / "assets" / "fonts"
    regular_font = font_dir / "ComicNeue-Regular.ttf"
    bold_font = font_dir / "ComicNeue-Bold.ttf"
    for font_path in (regular_font, bold_font):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    font_name = (
        font_manager.FontProperties(fname=str(regular_font)).get_name()
        if regular_font.exists()
        else "DejaVu Sans"
    )
    with plt.rc_context(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "font.weight": "bold",
            "axes.titlesize": 18.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 13.6,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 15.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.default": "regular",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    ):
        _plot_grouped_summary_with_side_panels(grouped_df, distribution_df, output_path)


def _plot_grouped_summary_with_side_panels(
    grouped_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    output_path: Path,
):
    model_labels = base.ordered_model_labels(grouped_df["model_label"].tolist())
    color_map = {
        "Qwen-Standard": "#4C6FAE",
        "DRT-SFT-800it": "#F39B2B",
        "w/o process reward": "#8CBDB8",
        "w/o step bouns": "#AF8EAA",
        "DRT-RL": "#E34F4F",
    }
    marker_map = {
        "Qwen-Standard": "o",
        "DRT-SFT-800it": "o",
        "w/o process reward": "D",
        "w/o step bouns": "s",
        "DRT-RL": "*",
    }
    label_map = {
        "Qwen-Standard": "Qwen-Standard",
        "DRT-SFT-800it": "DRT-SFT",
        "w/o process reward": "w/o process reward",
        "w/o step bouns": "w/o step bonus",
        "DRT-RL": "DRT-RL",
    }
    formula_map = {
        "w/o process reward": r"($R=R_{format}+R_{ans}$)",
        "w/o step bouns": r"($R=R_{format}+R_{main}$)",
    }
    regular_font_path = REPO_ROOT / "assets" / "fonts" / "ComicNeue-Regular.ttf"
    legend_font_path = REPO_ROOT / "assets" / "fonts" / "ComicNeue-Bold.ttf"
    legend_font = (
        font_manager.FontProperties(fname=str(legend_font_path), size=15.2)
        if legend_font_path.exists()
        else font_manager.FontProperties(weight="bold", size=15.2)
    )
    formula_font = (
        font_manager.FontProperties(fname=str(regular_font_path), size=13.4)
        if regular_font_path.exists()
        else font_manager.FontProperties(weight="normal", size=13.4)
    )

    fig, axes = plt.subplots(2, 3, figsize=(17.4, 9.3), sharex=False)
    axes = list(axes.flat)

    def style_panel(axis, panel_label: str, title: str):
        axis.set_title(f"{panel_label}. {title}", loc="left", pad=5)
        axis.grid(True, alpha=0.18, linewidth=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.95)
        axis.spines["bottom"].set_linewidth(0.95)
        axis.tick_params(axis="both", width=0.9, length=3.2)
        axis.set_xticks(range(3, 11))
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontweight("bold")

    def set_right_xlabel(axis, label: str = "GT reference step count"):
        axis.set_xlabel(label, labelpad=1)
        axis.xaxis.set_label_coords(0.83, -0.078)
        axis.xaxis.label.set_horizontalalignment("right")

    def plot_model_lines(axis, metric_key: str):
        handles = []
        labels = []
        for model_label in model_labels:
            subset = grouped_df[grouped_df["model_label"] == model_label].sort_values("gt_step_count")
            line = axis.plot(
                subset["gt_step_count"],
                subset[metric_key],
                marker=marker_map.get(model_label, "o"),
                markersize=5.4 if model_label != "DRT-RL" else 7.2,
                linewidth=1.95 if model_label != "DRT-RL" else 2.45,
                label=label_map.get(model_label, model_label),
                color=color_map.get(model_label, "#666666"),
                alpha=0.98,
            )[0]
            handles.append(line)
            labels.append(label_map.get(model_label, model_label))
        return handles, labels

    def draw_top_legend(handles):
        items = [
            (handle, label_map.get(model_label, model_label), formula_map.get(model_label, ""))
            for handle, model_label in zip(handles, model_labels)
        ]
        measure_texts = []
        measured = []
        for _, label, formula in items:
            label_text = fig.text(0, 0, label, fontproperties=legend_font, va="center", ha="left", alpha=0)
            formula_text = fig.text(0, 0, formula, fontproperties=formula_font, va="center", ha="left", alpha=0)
            formula_text.set_math_fontfamily("dejavusans")
            measure_texts.extend([label_text, formula_text])
            measured.append((label_text, formula_text))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_width = fig.bbox.width
        fig_height = fig.bbox.height
        label_widths = []
        formula_widths = []
        text_heights = []
        for label_text, formula_text in measured:
            label_box = label_text.get_window_extent(renderer=renderer)
            formula_box = formula_text.get_window_extent(renderer=renderer)
            label_widths.append(label_box.width / fig_width)
            formula_widths.append(formula_box.width / fig_width)
            text_heights.append(max(label_box.height, formula_box.height) / fig_height)
        for text in measure_texts:
            text.remove()

        handle_width = 0.018
        handle_text_gap = 0.004
        label_formula_gap = 0.0035
        item_gap = 0.022
        segment_widths = []
        for label_width, formula_width in zip(label_widths, formula_widths):
            formula_gap = label_formula_gap if formula_width else 0.0
            segment_widths.append(handle_width + handle_text_gap + label_width + formula_gap + formula_width)
        total_width = sum(segment_widths) + item_gap * (len(segment_widths) - 1)
        if total_width > 0.955:
            item_gap = max(0.008, item_gap - (total_width - 0.955) / max(1, len(segment_widths) - 1))
            total_width = sum(segment_widths) + item_gap * (len(segment_widths) - 1)

        y_center = 0.968
        pad_x = 0.010
        pad_y = 0.012
        box_height = max(text_heights) + 2 * pad_y
        box_left = 0.5 - total_width / 2 - pad_x
        box_bottom = y_center - box_height / 2
        box = FancyBboxPatch(
            (box_left, box_bottom),
            total_width + 2 * pad_x,
            box_height,
            boxstyle="round,pad=0.004,rounding_size=0.004",
            transform=fig.transFigure,
            linewidth=0.8,
            edgecolor="#DFE3EA",
            facecolor="white",
            clip_on=False,
            zorder=20,
        )
        fig.patches.append(box)

        cursor = box_left + pad_x
        for idx, ((handle, label, formula), label_width, formula_width, segment_width) in enumerate(
            zip(items, label_widths, formula_widths, segment_widths)
        ):
            color = handle.get_color()
            marker = handle.get_marker()
            marker_size = handle.get_markersize()
            line_width = handle.get_linewidth()
            legend_line = Line2D(
                [cursor, cursor + handle_width],
                [y_center, y_center],
                transform=fig.transFigure,
                color=color,
                linewidth=line_width,
                solid_capstyle="round",
                clip_on=False,
                zorder=21,
            )
            fig.add_artist(legend_line)
            legend_marker = Line2D(
                [cursor + handle_width / 2],
                [y_center],
                transform=fig.transFigure,
                color=color,
                marker=marker,
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="None",
                clip_on=False,
                zorder=22,
            )
            fig.add_artist(legend_marker)
            text_x = cursor + handle_width + handle_text_gap
            fig.text(
                text_x,
                y_center,
                label,
                fontproperties=legend_font,
                color="black",
                va="center",
                ha="left",
                zorder=21,
            )
            if formula:
                formula_text = fig.text(
                    text_x + label_width + label_formula_gap,
                    y_center,
                    formula,
                    fontproperties=formula_font,
                    color="black",
                    va="center",
                    ha="left",
                    zorder=21,
                )
                formula_text.set_math_fontfamily("dejavusans")
                formula_text.set_fontweight("normal")
            cursor += segment_width
            if idx < len(items) - 1:
                cursor += item_gap

    def set_padded_ylim(axis, values: pd.Series, *, floor: float = 0.0, cap=None, min_span: float = 1.0):
        metric_min = float(values.min())
        metric_max = float(values.max())
        span = max(metric_max - metric_min, min_span)
        pad = max(span * 0.10, min_span * 0.08)
        ymin = max(floor, metric_min - pad)
        ymax = metric_max + pad
        if cap is not None:
            ymax = min(cap, ymax)
        axis.set_ylim(ymin, ymax)

    dist_axis = axes[0]
    dist_axis.bar(
        distribution_df["gt_step_count"],
        distribution_df["sample_count"],
        color="#4C6FAE",
        alpha=0.90,
        width=0.72,
    )
    style_panel(dist_axis, "A", "GT Step Distribution")
    set_right_xlabel(dist_axis)
    dist_axis.set_ylabel("Sample count")
    dist_axis.grid(axis="y", alpha=0.20, linewidth=0.65)

    token_axis = axes[1]
    legend_handles, _ = plot_model_lines(token_axis, "mean_tokens_per_reason_step")
    style_panel(token_axis, "B", "Token / Reason Step")
    set_right_xlabel(token_axis)
    token_axis.set_ylabel("Mean tokens per reason step")
    set_padded_ylim(token_axis, grouped_df["mean_tokens_per_reason_step"], min_span=1.0)

    halluc_axis = axes[2]
    plot_model_lines(halluc_axis, "mean_hallucination_ratio")
    style_panel(halluc_axis, "C", "Hallucination Ratio")
    set_right_xlabel(halluc_axis)
    halluc_axis.set_ylabel("Mean hallucination ratio")
    set_padded_ylim(halluc_axis, grouped_df["mean_hallucination_ratio"], cap=1.0, min_span=0.08)

    reason_axis = axes[3]
    plot_model_lines(reason_axis, "mean_reason_step_count")
    style_panel(reason_axis, "D", "Reason Step")
    set_right_xlabel(reason_axis)
    reason_axis.set_ylabel("Mean atomic reasoning step count")
    set_padded_ylim(reason_axis, grouped_df["mean_reason_step_count"], min_span=1.0)

    metric_axes = [axes[4], axes[5]]
    metric_specs_reordered = [
        ("mean_effective_reasoning_step_count", "Effective Reason Step", "Mean effective reasoning step count"),
        ("mean_accuracy", "Accuracy", "Mean accuracy"),
    ]
    panel_labels = ["E", "F"]

    for axis, panel_label, (metric_key, title, ylabel) in zip(metric_axes, panel_labels, metric_specs_reordered):
        plot_model_lines(axis, metric_key)
        style_panel(axis, panel_label, title)
        axis.set_ylabel(ylabel)
        set_right_xlabel(axis)
        if metric_key == "mean_accuracy":
            set_padded_ylim(axis, grouped_df[metric_key], cap=1.0, min_span=0.08)
        else:
            set_padded_ylim(axis, grouped_df[metric_key], min_span=1.0)

    draw_top_legend(legend_handles)

    fig.subplots_adjust(left=0.068, right=0.994, bottom=0.078, top=0.880, wspace=0.130, hspace=0.260)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def plot_grouped_summary_by_model_with_accuracy(grouped_df: pd.DataFrame, distribution_df: pd.DataFrame, output_path: Path):
    model_labels = base.ordered_model_labels(grouped_df["model_label"].tolist())
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
    shared_right_ylim = (0.0, 1.0)

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
        axis.set_title(base.MODEL_PLOT_TITLE_MAP.get(model_label, model_label))
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
        axis_right.plot(
            subset["gt_step_count"],
            subset["mean_accuracy"],
            marker="D",
            linewidth=2,
            linestyle="-.",
            label="Accuracy",
            color="#d62728",
        )
        axis_right.set_ylabel("Ratio / Accuracy")
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


def sync_outputs(output_dir: Path, backup_dir: Path, filenames: list[str]):
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = output_dir / name
        if src.exists():
            dst = backup_dir / name
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    combined_df = load_and_concat(DEFAULT_INPUTS, args.min_gt_step, args.max_gt_step)
    combined_df = compute_accuracy_column(combined_df)
    grouped_df = build_grouped_summary_with_accuracy(combined_df)
    dist_df = build_combined_distribution(combined_df)

    suffix = f"gt{args.min_gt_step}to{args.max_gt_step}"
    outputs = {
        "per_sample": f"combined_step_bonus_per_sample_{suffix}.csv",
        "grouped": f"combined_step_bonus_grouped_by_gt_steps_{suffix}.csv",
        "model_png": f"combined_step_bonus_grouped_by_model_{suffix}.png",
        "metric_png": f"combined_step_bonus_grouped_by_metric_{suffix}.png",
        "metric_png_2x2": f"combined_step_bonus_grouped_by_metric_2x2_{suffix}.png",
        "metric_png_with_side_panels": f"combined_step_bonus_grouped_by_metric_with_side_panels_{suffix}.png",
        "metric_pdf_with_side_panels": f"combined_step_bonus_grouped_by_metric_with_side_panels_{suffix}.pdf",
        "metric_svg_with_side_panels": f"combined_step_bonus_grouped_by_metric_with_side_panels_{suffix}.svg",
        "dist_csv": f"combined_gt_step_distribution_{suffix}.csv",
        "dist_png": f"combined_gt_step_distribution_{suffix}.png",
    }

    combined_df.to_csv(args.output_dir / outputs["per_sample"], index=False)
    grouped_df.to_csv(args.output_dir / outputs["grouped"], index=False)
    dist_df.to_csv(args.output_dir / outputs["dist_csv"], index=False)
    plot_grouped_summary_by_model_with_accuracy(grouped_df, dist_df, args.output_dir / outputs["model_png"])
    plot_grouped_summary_by_metric_with_accuracy(grouped_df, args.output_dir / outputs["metric_png"])
    plot_grouped_summary_by_metric_with_accuracy(
        grouped_df,
        args.output_dir / outputs["metric_png_2x2"],
        n_rows=2,
        n_cols=2,
        figsize=(14, 9),
    )
    plot_grouped_summary_with_side_panels(
        grouped_df,
        dist_df,
        args.output_dir / outputs["metric_png_with_side_panels"],
    )
    plot_grouped_summary_with_side_panels(
        grouped_df,
        dist_df,
        args.output_dir / outputs["metric_pdf_with_side_panels"],
    )
    plot_grouped_summary_with_side_panels(
        grouped_df,
        dist_df,
        args.output_dir / outputs["metric_svg_with_side_panels"],
    )
    plot_combined_gt_step_distribution(dist_df, args.output_dir / outputs["dist_png"])

    sync_outputs(args.output_dir, args.backup_dir, list(outputs.values()))

    print(f"Saved combined per-sample metrics to: {args.output_dir / outputs['per_sample']}")
    print(f"Saved combined grouped metrics to: {args.output_dir / outputs['grouped']}")
    print(f"Saved combined GT step distribution to: {args.output_dir / outputs['dist_csv']}")
    print(f"Saved combined model-grouped plot to: {args.output_dir / outputs['model_png']}")
    print(f"Saved combined metric-grouped plot to: {args.output_dir / outputs['metric_png']}")
    print(f"Saved combined metric-grouped 2x2 plot to: {args.output_dir / outputs['metric_png_2x2']}")
    print(f"Saved combined metric plot with side panels to: {args.output_dir / outputs['metric_png_with_side_panels']}")
    print(f"Saved combined metric plot with side panels PDF to: {args.output_dir / outputs['metric_pdf_with_side_panels']}")
    print(f"Saved combined metric plot with side panels SVG to: {args.output_dir / outputs['metric_svg_with_side_panels']}")
    print(f"Saved combined GT step distribution plot to: {args.output_dir / outputs['dist_png']}")


if __name__ == "__main__":
    main()
