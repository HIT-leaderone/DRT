#!/usr/bin/env python3
"""Rebuild the DRT reward-design figure as native vector PDF/SVG/PNG."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle


WIDTH = 1672
HEIGHT = 941
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets"
DEFAULT_FONT_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_STEM = "drt_reward_design_diagram"

BLACK = "#050505"
RED = "#d70d0d"
GREEN = "#0b8d24"
ORANGE = "#ff6f00"
BLUE = "#0d67bd"
DEEP_BLUE = "#0e5285"
PURPLE = "#6d46e8"
CYAN_BOX = "#e7f6f8"
GREEN_BOX = "#eef8df"
PEACH_BOX = "#fff2e9"
YELLOW_BOX = "#fff5d8"
BLUE_BOX = "#eaf4ff"

FONT_REGULAR: font_manager.FontProperties | None = None
FONT_BOLD: font_manager.FontProperties | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    return parser.parse_args()


def configure_matplotlib(font_dir: Path) -> None:
    global FONT_REGULAR, FONT_BOLD
    regular = font_dir / "ComicNeue-Regular.ttf"
    bold = font_dir / "ComicNeue-Bold.ttf"
    if regular.exists():
        font_manager.fontManager.addfont(str(regular))
        FONT_REGULAR = font_manager.FontProperties(fname=str(regular))
    if bold.exists():
        font_manager.fontManager.addfont(str(bold))
        FONT_BOLD = font_manager.FontProperties(fname=str(bold))

    family = FONT_REGULAR.get_name() if FONT_REGULAR is not None else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family, "DejaVu Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        }
    )


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 20,
    color: str = BLACK,
    weight: str = "bold",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.05,
    zorder: int = 10,
    stroke: float = 0.0,
) -> None:
    kwargs = {}
    if weight == "bold" and FONT_BOLD is not None:
        kwargs["fontproperties"] = FONT_BOLD
    elif FONT_REGULAR is not None:
        kwargs["fontproperties"] = FONT_REGULAR
    else:
        kwargs["fontweight"] = weight

    effects = []
    if stroke > 0:
        effects.append(path_effects.Stroke(linewidth=stroke, foreground=color))
    effects.append(path_effects.Normal())
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
        path_effects=effects,
        **kwargs,
    )


def round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = "white",
    edge: str = "none",
    lw: float = 1.6,
    radius: float = 16,
    alpha: float = 1.0,
    linestyle: str | tuple[int, tuple[int, ...]] = "solid",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    lw: float = 3,
    color: str = BLACK,
    mutation: float = 22,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "solid",
    zorder: int = 8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def line(ax: plt.Axes, pts: list[tuple[float, float]], *, color: str = BLACK, lw: float = 3, zorder: int = 7) -> None:
    ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_capstyle="round", zorder=zorder))


def draw_check(ax: plt.Axes, cx: float, cy: float, r: float = 17, *, color: str = GREEN) -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color, lw=3, zorder=8))
    line(ax, [(cx - r * 0.44, cy - 1), (cx - r * 0.13, cy + r * 0.35), (cx + r * 0.52, cy - r * 0.45)], color=color, lw=3, zorder=9)


def draw_cross(ax: plt.Axes, cx: float, cy: float, r: float = 17, *, color: str = "#ff1111") -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color, lw=3, zorder=8))
    line(ax, [(cx - r * 0.42, cy - r * 0.42), (cx + r * 0.42, cy + r * 0.42)], color=color, lw=3, zorder=9)
    line(ax, [(cx + r * 0.42, cy - r * 0.42), (cx - r * 0.42, cy + r * 0.42)], color=color, lw=3, zorder=9)


def draw_warning_step(ax: plt.Axes, cx: float, cy: float, r: float = 17) -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=ORANGE, lw=3, zorder=8))
    add_text(ax, cx, cy + 1, "!", size=26, color=ORANGE, weight="bold", ha="center", zorder=9, stroke=0.2)


def draw_small_arrow_sequence(ax: plt.Axes, xs: list[float], y: float, symbols: list[str]) -> None:
    for i, (x, symbol) in enumerate(zip(xs, symbols)):
        if symbol == "check":
            draw_check(ax, x, y)
        elif symbol == "warn":
            draw_warning_step(ax, x, y)
        else:
            draw_cross(ax, x, y)
        if i < len(xs) - 1:
            add_arrow(ax, (x + 27, y), (xs[i + 1] - 27, y), lw=2.3, mutation=16, style="->")


def draw_flame(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    outer = [
        (cx - 17 * scale, cy + 35 * scale),
        (cx - 34 * scale, cy + 5 * scale),
        (cx - 11 * scale, cy - 18 * scale),
        (cx - 4 * scale, cy - 52 * scale),
        (cx + 16 * scale, cy - 18 * scale),
        (cx + 31 * scale, cy - 29 * scale),
        (cx + 27 * scale, cy + 5 * scale),
        (cx + 14 * scale, cy + 35 * scale),
    ]
    inner = [
        (cx - 4 * scale, cy + 34 * scale),
        (cx - 15 * scale, cy + 11 * scale),
        (cx + 2 * scale, cy - 8 * scale),
        (cx + 8 * scale, cy - 29 * scale),
        (cx + 21 * scale, cy - 2 * scale),
        (cx + 16 * scale, cy + 23 * scale),
    ]
    ax.add_patch(Polygon(outer, closed=True, facecolor="#ff2a2a", edgecolor="none", zorder=6))
    ax.add_patch(Polygon(inner, closed=True, facecolor="#ffb000", edgecolor="none", zorder=7))


def draw_snowflake(ax: plt.Axes, cx: float, cy: float, r: float = 31) -> None:
    for angle in range(0, 180, 30):
        rad = math.radians(angle)
        dx = math.cos(rad) * r
        dy = math.sin(rad) * r
        line(ax, [(cx - dx, cy - dy), (cx + dx, cy + dy)], color="#6bd4f2", lw=4, zorder=8)
    ax.add_patch(Circle((cx, cy), 4, facecolor="#6bd4f2", edgecolor="none", zorder=9))


def draw_file_icon(ax: plt.Axes, x: float, y: float, w: float = 69, h: float = 122) -> None:
    fold = 20
    pts = [(x + fold, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y + fold), (x + fold, y + fold)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=BLACK, lw=3, zorder=6))
    line(ax, [(x + fold, y), (x + fold, y + fold), (x, y + fold)], lw=3, zorder=7)
    add_text(ax, x + 34, y + 44, "</>", size=25, ha="center", color=BLACK, stroke=0.3, zorder=8)
    round_box(ax, x + 15, y + 72, 50, 34, face="#b8eef5", edge=BLACK, lw=2.5, radius=4, zorder=7)
    add_text(ax, x + 40, y + 90, "DRT", size=16, ha="center", color=BLACK, stroke=0.2, zorder=8)


def draw_robot(ax: plt.Axes, cx: float, y: float) -> None:
    for dx in [-24, 0, 24]:
        ax.add_patch(Circle((cx + dx, y + 6), 8, facecolor="#ef2525", edgecolor=BLACK, lw=3, zorder=5))
    ax.add_patch(Circle((cx - 75, y + 59), 20, facecolor="#318cc5", edgecolor=BLACK, lw=4, zorder=4))
    ax.add_patch(Circle((cx + 75, y + 59), 20, facecolor="#318cc5", edgecolor=BLACK, lw=4, zorder=4))
    round_box(ax, cx - 63, y + 14, 126, 112, face="#aeeeff", edge=BLACK, lw=4, radius=23, zorder=5)
    round_box(ax, cx - 45, y + 29, 90, 70, face="#f9ffff", edge=BLACK, lw=3, radius=12, zorder=6)
    ax.add_patch(Circle((cx - 27, y + 49), 10, facecolor="#eefcff", edgecolor=BLACK, lw=3, zorder=7))
    ax.add_patch(Circle((cx + 27, y + 49), 10, facecolor="#eefcff", edgecolor=BLACK, lw=3, zorder=7))
    ax.add_patch(FancyArrowPatch((cx - 11, y + 64), (cx + 11, y + 64), arrowstyle="-", connectionstyle="arc3,rad=0.8", color=BLACK, lw=3, zorder=7))
    for dx in [-37, -31, 31, 37]:
        line(ax, [(cx + dx, y + 68), (cx + dx + (4 if dx < 0 else -4), y + 68)], lw=2.2, zorder=7)
        line(ax, [(cx + dx, y + 76), (cx + dx + (4 if dx < 0 else -4), y + 76)], lw=2.2, zorder=7)
    round_box(ax, cx - 13, y + 126, 26, 23, face="#62c6ef", edge=BLACK, lw=4, radius=8, zorder=4)
    ax.add_patch(Polygon([(cx - 48, y + 150), (cx + 48, y + 150), (cx + 48, y + 183), (cx - 48, y + 183)], facecolor="#b7efff", edgecolor=BLACK, lw=3, zorder=3))
    add_text(ax, cx - 27, y + 171, "AI", size=25, color=BLACK, ha="center", zorder=8, stroke=0.2)


def draw_qwen_mark(ax: plt.Axes, cx: float, cy: float, size: float = 61) -> None:
    colors = ["#8159ff", "#6b45ea", "#4f7ff0", "#213f9a", "#7bd7df", "#9a72ff"]
    for i in range(6):
        angle = math.radians(i * 60 + 30)
        ox = math.cos(angle) * size * 0.24
        oy = math.sin(angle) * size * 0.24
        rot = angle + math.pi / 4
        c, s = math.cos(rot), math.sin(rot)
        base = [(-14, -7), (12, -7), (18, 0), (12, 7), (-14, 7), (-20, 0)]
        pts = [(cx + ox + px * c - py * s, cy + oy + px * s + py * c) for px, py in base]
        ax.add_patch(Polygon(pts, closed=True, facecolor=colors[i], edgecolor=colors[i], lw=1, zorder=8))


def draw_left_side(ax: plt.Axes) -> None:
    draw_robot(ax, 246, 98)

    round_box(ax, 49, 280, 374, 128, face=CYAN_BOX, edge="none", radius=15, zorder=1)
    add_text(ax, 235, 325, "DRT Policy", size=34, ha="center", color=BLACK, stroke=0.35)
    add_text(ax, 226, 375, "(init from DRT-SFT)", size=25, ha="center", color=BLACK, stroke=0.22)
    draw_flame(ax, 381, 340, 0.66)

    round_box(ax, 51, 637, 269, 92, face=GREEN_BOX, edge="none", radius=13, zorder=1)
    add_text(ax, 154, 684, "Ref Model", size=38, ha="center", color=BLACK, stroke=0.35)
    draw_snowflake(ax, 273, 682, 31)

    add_arrow(ax, (164, 611), (164, 412), lw=3.2, mutation=21)
    add_arrow(ax, (164, 417), (164, 623), lw=3.2, mutation=21)
    add_text(ax, 130, 522, "KL", size=31, ha="center", color=BLACK, stroke=0.25)

    add_text(ax, 572, 296, "Rollout", size=30, ha="center", color=BLACK, stroke=0.25)
    add_arrow(ax, (435, 342), (688, 342), lw=3, mutation=23)

    line(ax, [(300, 412), (300, 457), (585, 457)], lw=3)
    add_arrow(ax, (300, 457), (300, 412), lw=3, mutation=21)
    add_text(ax, 461, 457, "Reward", size=30, ha="center", color=BLACK, stroke=0.25)
    ax.add_patch(Circle((615, 474), 32, facecolor="white", edgecolor=BLACK, lw=4, zorder=8))
    add_text(ax, 615, 474, "+", size=45, ha="center", color=BLACK, stroke=0.2, zorder=9)
    line(ax, [(585, 474), (700, 474)], lw=3)
    add_arrow(ax, (615, 506), (615, 581), lw=3, mutation=21)

    round_box(ax, 400, 586, 298, 238, face=PEACH_BOX, edge="none", radius=18, zorder=1)
    add_text(ax, 548, 624, "(2) Format Reward", size=23, color=RED, ha="center", stroke=0.2)
    draw_file_icon(ax, 418, 656)
    add_text(ax, 510, 697, "<visual> ... </visual>", size=13.5, color=BLACK, stroke=0.08)
    add_text(ax, 510, 739, "<think> ... </think>", size=13.5, color=BLACK, stroke=0.08)
    add_text(ax, 510, 786, "<answer> ... </answer>", size=13.5, color=BLACK, stroke=0.08)


def draw_score_box(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    round_box(ax, x, y, 76, 91, face=color, edge="white", lw=2, radius=10, zorder=5)
    add_text(ax, x + 38, y + 47, text, size=30, ha="center", color=BLACK, stroke=0.15)


def draw_main_reward(ax: plt.Axes) -> None:
    round_box(ax, 698, 96, 928, 678, face="#fff1ea", edge="none", radius=25, alpha=0.72, zorder=0)
    add_text(ax, 733, 153, "(1) Main Reward + Step Bonus", size=38, color=RED, stroke=0.35)
    draw_qwen_mark(ax, 1553, 145, 64)

    round_box(ax, 719, 214, 518, 548, face=YELLOW_BOX, edge="none", radius=20, alpha=0.78, zorder=1)
    add_text(ax, 978, 253, r"Process Reward ($R_{\mathrm{main}}$)", size=28, color=BLACK, ha="center", stroke=0.18)

    rows = [
        (294, ["check", "check", "check", "check", "check"], "1.0", "#f0f8e8"),
        (420, ["check", "check", "check", "check", "cross"], "0.9", "#fff0c9"),
        (545, ["check", "warn", "warn", "warn", "warn"], "0.2", "#ffd9d2"),
    ]
    xs = [771, 850, 929, 1008, 1087]
    for y, symbols, score, score_color in rows:
        round_box(ax, 735, y, 386, 95, face="white", edge="none", radius=11, alpha=0.82, zorder=3)
        draw_small_arrow_sequence(ax, xs, y + 48, symbols)
        draw_score_box(ax, 1142, y, score, score_color)

    round_box(ax, 735, 669, 487, 65, face="white", edge="none", radius=10, alpha=0.82, zorder=3)
    draw_check(ax, 758, 701, 14)
    add_text(ax, 780, 701, "matched step", size=14.5, color=BLACK, stroke=0.06)
    draw_warning_step(ax, 912, 701, 14)
    add_text(ax, 934, 701, "hallucinated step", size=14.5, color=BLACK, stroke=0.06)
    draw_cross(ax, 1086, 701, 14)
    add_text(ax, 1108, 701, "mismatch step", size=14.5, color=BLACK, stroke=0.06)


def draw_blue_node(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Circle((cx, cy), 10, facecolor="#0974db", edgecolor=BLACK, lw=2, zorder=8))


def draw_step_bonus(ax: plt.Axes) -> None:
    round_box(ax, 1255, 214, 354, 548, face=BLUE_BOX, edge="none", radius=20, alpha=0.92, zorder=1)
    add_text(ax, 1432, 251, "Step Bonus", size=31, color=BLUE, ha="center", stroke=0.25)

    round_box(ax, 1270, 291, 325, 153, face="white", edge="none", radius=13, alpha=0.78, zorder=3)
    add_text(ax, 1286, 322, "effective multi-step", size=20, color=BLACK, stroke=0.15)
    top_xs = [1295, 1342, 1389, 1436, 1482]
    for i, x in enumerate(top_xs):
        draw_blue_node(ax, x, 379)
        if i < len(top_xs) - 1:
            add_arrow(ax, (x + 15, 379), (top_xs[i + 1] - 15, 379), lw=2.2, mutation=13, style="->")
    ax.add_patch(Ellipse((1543, 372), 79, 99, facecolor="white", edgecolor=GREEN, lw=3, zorder=8))
    add_text(ax, 1543, 354, "+", size=29, color=GREEN, ha="center", stroke=0.15)
    add_text(ax, 1543, 391, "bonus", size=19, color=GREEN, ha="center", stroke=0.1)

    round_box(ax, 1270, 469, 325, 257, face="white", edge="none", radius=13, alpha=0.78, zorder=3)
    add_text(ax, 1283, 504, "deep exploration", size=20, color=BLACK, stroke=0.15)
    pts = [(1290, 550), (1345, 550), (1406, 550), (1468, 550), (1337, 614), (1413, 661)]
    for p in pts:
        draw_blue_node(ax, *p)
    for a, b in [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]:
        add_arrow(ax, (a[0] + 15, a[1]), (b[0] - 15, b[1]), lw=2.2, mutation=13, style="->")
    add_arrow(ax, (1302, 573), (1328, 603), lw=2, mutation=12, style="->", linestyle=(0, (4, 5)))
    add_arrow(ax, (1346, 626), (1402, 653), lw=2, mutation=12, style="->", linestyle=(0, (4, 5)))
    add_arrow(ax, (1427, 645), (1460, 573), lw=2, mutation=12, style="->", linestyle=(0, (4, 5)))
    ax.add_patch(Ellipse((1543, 568), 79, 108, facecolor="white", edgecolor=GREEN, lw=3, zorder=8))
    add_text(ax, 1543, 553, "++", size=30, color=GREEN, ha="center", stroke=0.15)
    add_text(ax, 1543, 590, "bonus", size=19, color=GREEN, ha="center", stroke=0.1)


def draw_figure(font_dir: Path) -> plt.Figure:
    configure_matplotlib(font_dir)
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, facecolor="white", edgecolor="none", zorder=-10))
    ax.add_patch(Rectangle((8, 12), WIDTH - 18, HEIGHT - 24, facecolor="none", edgecolor=BLACK, lw=2.5, linestyle=(0, (4, 8)), zorder=20))

    draw_left_side(ax)
    draw_main_reward(ax)
    draw_step_bonus(ax)
    return fig


def save_outputs(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(outputs[0], format="pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(outputs[1], format="svg", bbox_inches="tight", pad_inches=0)
    fig.savefig(outputs[2], format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    return outputs


def main() -> None:
    args = parse_args()
    fig = draw_figure(args.font_dir)
    outputs = save_outputs(fig, args.output_dir, args.stem, args.dpi)
    plt.close(fig)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
