#!/usr/bin/env python3
"""Plot a compact v3 DRT performance-vs-token-efficiency teaser."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from main_result_parser import parse_main_result_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_MAIN_RESULT_TEX = REPO_ROOT / "main_result.tex"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets"
SHOWCASE_FONT_FILES = ("ComicNeue-Regular.ttf", "ComicNeue-Bold.ttf")
ACCENT_COLOR = "#A32020"
OUTLIER_METHOD = "Qwen-Thinking"
TRADEOFF_TEXT_SIZE = 18
POINT_SIZE = 320
TRADEOFF_LEGEND_TEXT_SIZE = TRADEOFF_TEXT_SIZE * 0.9
LEGEND_MARKER_SIZE = POINT_SIZE**0.5 * 0.9
ANNOTATED_METHODS = {"Qwen-Standard", "Qwen-Thinking", "DRT-RL"}

METHOD_ORDER = (
    "Qwen-DA",
    "Qwen-DRT",
    "Qwen-Standard",
    "Qwen-Thinking",
    "CoD",
    "ThinkLess",
    "VisionThink",
    "DRT-SFT",
    "DRT-RL",
)

METHOD_STYLE = {
    "Qwen-DA": {"color": "#93A3B5", "marker": "o", "size": POINT_SIZE, "alpha": 0.72},
    "Qwen-DRT": {"color": "#8CBDB8", "marker": "o", "size": POINT_SIZE, "alpha": 0.72},
    "Qwen-Standard": {"color": "#4C6FAE", "marker": "o", "size": POINT_SIZE, "alpha": 0.88},
    "Qwen-Thinking": {"color": "#1F3552", "marker": "^", "size": POINT_SIZE, "alpha": 0.88},
    "CoD": {"color": "#87A87A", "marker": "D", "size": POINT_SIZE, "alpha": 0.72},
    "ThinkLess": {"color": "#B39A8B", "marker": "P", "size": POINT_SIZE, "alpha": 0.72},
    "VisionThink": {"color": "#AF8EAA", "marker": "s", "size": POINT_SIZE, "alpha": 0.72},
    "DRT-SFT": {"color": "#F39B2B", "marker": "o", "size": POINT_SIZE, "alpha": 0.95},
    "DRT-RL": {"color": "#E34F4F", "marker": "*", "size": POINT_SIZE, "alpha": 1.0},
}

ANNOTATION_OFFSETS = {
    "Qwen-DA": (8, -2),
    "Qwen-DRT": (8, -12),
    "Qwen-Standard": (0, 16),
    "Qwen-Thinking": (-10, 12),
    "CoD": (8, -18),
    "ThinkLess": (8, -12),
    "VisionThink": (8, 8),
    "DRT-SFT": (0, 16),
    "DRT-RL": (0, 18),
}

ANNOTATION_ALIGN = {
    "Qwen-Standard": "center",
    "Qwen-Thinking": "right",
    "DRT-SFT": "center",
    "DRT-RL": "center",
}


@dataclass(frozen=True)
class Point:
    key: str
    display_name: str
    acc: float
    tokens: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-result-tex",
        type=Path,
        default=DEFAULT_MAIN_RESULT_TEX,
        help="LaTeX table containing method accuracy/token values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the generated figure.",
    )
    parser.add_argument(
        "--stem",
        default="drt_token_efficiency_v3",
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output dpi.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=DEFAULT_FONT_DIR,
        help="Directory containing showcase-style ComicNeue font files.",
    )
    parser.add_argument("--fig-width", type=float, default=8.6)
    parser.add_argument("--fig-height", type=float, default=7.4)
    parser.add_argument("--title-size", type=float, default=18)
    parser.add_argument("--hide-title", action="store_true")
    return parser.parse_args()


def register_showcase_font(font_dir: Path) -> str:
    font_name = "DejaVu Sans"
    regular_path = font_dir / SHOWCASE_FONT_FILES[0]

    for filename in SHOWCASE_FONT_FILES:
        font_path = font_dir / filename
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    if regular_path.exists():
        font_name = font_manager.FontProperties(fname=str(regular_path)).get_name()

    return font_name


def set_showcase_style(font_dir: Path, title_size: float = 18) -> None:
    font_name = register_showcase_font(font_dir)
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.titlesize": title_size,
            "axes.labelsize": TRADEOFF_TEXT_SIZE,
            "xtick.labelsize": TRADEOFF_TEXT_SIZE,
            "ytick.labelsize": TRADEOFF_TEXT_SIZE,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        }
    )


def build_points(main_result_tex: Path) -> list[Point]:
    rows = parse_main_result_table(main_result_tex)
    missing = [method for method in METHOD_ORDER if method not in rows]
    if missing:
        raise ValueError(f"Missing methods in {main_result_tex}: {', '.join(missing)}")

    points: list[Point] = []
    for method in METHOD_ORDER:
        avg = rows[method].metrics["AVG."]
        points.append(
            Point(
                key=method,
                display_name=method,
                acc=avg.acc,
                tokens=avg.tokens,
            )
        )
    return points


def make_legend_handle(point: Point) -> Line2D:
    style = METHOD_STYLE[point.key]
    label = point.display_name

    return Line2D(
        [0],
        [0],
        linestyle="",
        marker=style["marker"],
        markerfacecolor=style["color"],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=LEGEND_MARKER_SIZE,
        label=label,
        alpha=style["alpha"],
    )


def add_arc_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float,
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=22,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=1.7,
        color=ACCENT_COLOR,
        alpha=0.9,
        zorder=2,
    )
    ax.add_patch(arrow)

    if text and text_xy:
        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            color=ACCENT_COLOR,
            fontsize=TRADEOFF_TEXT_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#FFF8F7",
                "edgecolor": "#F0C6C6",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
        )


def draw_offscale_break(
    ax: plt.Axes,
    *,
    break_x: float,
    outlier_x: float,
    outlier: Point,
    y_min: float,
) -> None:
    axis_y = y_min
    ax.plot(
        [outlier_x, outlier_x],
        [y_min + 0.25, outlier.acc - 0.35],
        color="#8C95A3",
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        alpha=0.75,
        zorder=1,
    )
    axis_break_x = break_x + (outlier_x - break_x) * 0.52
    axis_color = "#000000"
    mask_left = axis_break_x - 25
    mask_right = axis_break_x + 25
    slash_offsets = (-5.2, 5.2)
    slash_dx = 4.8
    slash_height = 0.58
    ax.plot(
        [mask_left, mask_right],
        [axis_y, axis_y],
        color="#FFFFFF",
        linewidth=4.8,
        solid_capstyle="butt",
        clip_on=False,
        zorder=6,
    )
    ax.plot(
        [mask_left, axis_break_x + slash_offsets[0]],
        [axis_y, axis_y],
        color=axis_color,
        linewidth=1.35,
        solid_capstyle="butt",
        clip_on=False,
        zorder=7,
    )
    ax.plot(
        [axis_break_x + slash_offsets[1], mask_right],
        [axis_y, axis_y],
        color=axis_color,
        linewidth=1.35,
        solid_capstyle="butt",
        clip_on=False,
        zorder=7,
    )
    for offset in slash_offsets:
        ax.plot(
            [axis_break_x + offset - slash_dx, axis_break_x + offset + slash_dx],
            [axis_y - slash_height / 2, axis_y + slash_height / 2],
            color=axis_color,
            linewidth=1.6,
            solid_capstyle="round",
            clip_on=False,
            zorder=8,
        )
    ax.text(
        outlier_x,
        y_min - 0.62,
        f"{outlier.tokens:.0f}",
        color="#000000",
        fontsize=TRADEOFF_TEXT_SIZE,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
        zorder=5,
    )


def build_figure(
    points: list[Point],
    font_dir: Path,
    *,
    figsize: tuple[float, float] = (8.6, 7.4),
    title_size: float = 18,
    title_text: str | None = "Accuracy-Token Tradeoff Across Reasoning Methods",
) -> plt.Figure:
    set_showcase_style(font_dir, title_size=title_size)

    fig, ax = plt.subplots(figsize=figsize)

    point_by_key = {point.key: point for point in points}
    non_outlier_points = [point for point in points if point.key != OUTLIER_METHOD]
    non_outlier_token_max = max(point.tokens for point in non_outlier_points)
    outlier = point_by_key.get(OUTLIER_METHOD)
    clip_outlier = bool(outlier and outlier.tokens > non_outlier_token_max * 1.35)
    x_max = (non_outlier_token_max if clip_outlier else max(point.tokens for point in points)) * 1.20
    break_x = non_outlier_token_max * 1.08
    outlier_x = break_x + 105
    if clip_outlier:
        x_max = outlier_x + 70

    for point in points:
        style = METHOD_STYLE[point.key]
        x_value = outlier_x if clip_outlier and point.key == OUTLIER_METHOD else point.tokens
        if point.key == "DRT-RL":
            ax.scatter(
                [x_value],
                [point.acc],
                s=1840,
                marker="o",
                color=style["color"],
                alpha=0.18,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                [x_value],
                [point.acc],
                s=2720,
                marker="o",
                color=style["color"],
                alpha=0.08,
                edgecolors="none",
                zorder=1,
            )
        ax.scatter(
            [x_value],
            [point.acc],
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolors="#1A1A1A" if point.key in {"DRT-RL", "DRT-SFT", "Qwen-Standard"} else "#333333",
            linewidths=1.0 if point.key in {"DRT-RL", "DRT-SFT", "Qwen-Standard"} else 0.75,
            alpha=style["alpha"],
            zorder=5 if point.key == "DRT-RL" else 3,
        )
        if point.key in ANNOTATED_METHODS:
            label = point.display_name
            if clip_outlier and point.key == OUTLIER_METHOD:
                label = f"{point.display_name}\n(off-scale)"
            ax.annotate(
                label,
                xy=(x_value, point.acc),
                xytext=ANNOTATION_OFFSETS[point.key],
                textcoords="offset points",
                fontsize=TRADEOFF_TEXT_SIZE,
                fontweight="bold",
                color=style["color"],
                ha=ANNOTATION_ALIGN.get(point.key, "left"),
                va="bottom",
                zorder=6,
            )

    standard = point_by_key["Qwen-Standard"]
    drt_rl = point_by_key["DRT-RL"]
    token_ratio = standard.tokens / drt_rl.tokens
    acc_delta = drt_rl.acc - standard.acc
    add_arc_arrow(
        ax,
        start=(standard.tokens - 105, standard.acc + 0.78),
        end=(drt_rl.tokens + 145, drt_rl.acc + 0.62),
        rad=0.12,
        text=f"{token_ratio:.1f}x fewer tokens, {acc_delta:+.1f} Acc",
        text_xy=(640, 69.82),
    )

    ax.set_xlabel("Average Output Tokens", fontweight="bold")
    ax.set_ylabel("Average Accuracy (%)", fontweight="bold")
    if title_text:
        ax.set_title(title_text, fontweight="bold", pad=10)

    acc_values = [point.acc for point in points]
    y_min = min(acc_values) - 3.0
    y_max = max(acc_values) + 3.6
    ax.set_xlim(-x_max * 0.03, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(0, int(break_x) + 1, 200))

    if clip_outlier and outlier:
        draw_offscale_break(
            ax,
            break_x=break_x,
            outlier_x=outlier_x,
            outlier=outlier,
            y_min=y_min,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", width=1.15, length=5.8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(True, which="major", alpha=0.16, linewidth=0.8)

    legend_handles = [make_legend_handle(point_by_key[method]) for method in METHOD_ORDER]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.93, 0.005),
        frameon=True,
        fancybox=True,
        facecolor=(1.0, 1.0, 1.0, 0.9),
        edgecolor="#DFE3EA",
        prop={"size": TRADEOFF_LEGEND_TEXT_SIZE, "weight": "bold"},
        ncol=2,
        columnspacing=1.45,
        handlelength=1.35,
        handletextpad=0.62,
        labelspacing=0.72,
        borderpad=0.78,
        borderaxespad=0.20,
    )

    return fig


def plot(
    output_base: Path,
    dpi: int,
    points: list[Point],
    font_dir: Path,
    *,
    figsize: tuple[float, float] = (8.6, 7.4),
    title_size: float = 18,
    title_text: str | None = "Accuracy-Token Tradeoff Across Reasoning Methods",
) -> list[Path]:
    fig = build_figure(
        points,
        font_dir,
        figsize=figsize,
        title_size=title_size,
        title_text=title_text,
    )

    output_paths = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".svg"),
    ]
    fig.savefig(output_paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(output_paths[1], bbox_inches="tight")
    fig.savefig(output_paths[2], bbox_inches="tight")
    plt.close(fig)
    return output_paths


def main() -> None:
    args = parse_args()
    points = build_points(args.main_result_tex)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.stem
    output_paths = plot(
        output_base=output_base,
        dpi=args.dpi,
        points=points,
        font_dir=args.font_dir,
        figsize=(args.fig_width, args.fig_height),
        title_size=args.title_size,
        title_text=None if args.hide_title else "Accuracy-Token Tradeoff Across Reasoning Methods",
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
