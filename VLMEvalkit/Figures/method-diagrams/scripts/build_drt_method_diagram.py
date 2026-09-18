#!/usr/bin/env python3
"""Build the DRT method diagram as PDF/SVG/PNG plus reusable icon assets."""

from __future__ import annotations

import argparse
import functools
import math
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from PIL import Image


WIDTH = 2048
HEIGHT = 645
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_DIR = REPO_ROOT / "assets" / "fonts"
QWEN_LOGO_PATH = REPO_ROOT / "assets" / "LOGO.svg"
QWEN_ICON_PATH = REPO_ROOT / "assets" / "Qwen icon.png"
GPT_ICON_PATH = REPO_ROOT / "assets" / "GPT icon.png"
SHOWCASE_FONT_FILES = ("ComicNeue-Regular.ttf", "ComicNeue-Bold.ttf")
SHOWCASE_REGULAR_PROP: font_manager.FontProperties | None = None
SHOWCASE_BOLD_PROP: font_manager.FontProperties | None = None

BLACK = "#202124"
TEXT = "#1f1f1f"
MUTED = "#5f6368"
LIGHT_LINE = "#d9dde3"
RED = "#ff2028"
TEAL = "#0c8f8f"
TEAL_DARK = "#087e7e"
ORANGE = "#ff7600"
ORANGE_DARK = "#e36400"
GREEN = "#0a7a42"
GREEN_DARK = "#056133"
BLUE = "#126de5"
BLUE_DARK = "#0758c8"
PURPLE = "#5522cc"
PURPLE_DARK = "#3b17a8"

TAG_VISUAL = "#0a8b75"
TAG_THINK = "#1464e7"
TAG_ANSWER = "#4f47ce"
TEXT_SCALE = 0.94
MIN_RENDERED_TEXT_SIZE = 12.0
TEXT_Y_NUDGE = 0.7
MIN_TEXT_STROKE = 0.24
MAX_TEXT_STROKE = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets",
        help="Directory for the generated PDF/SVG/PNG.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Directory for reusable SVG icon assets.",
    )
    parser.add_argument("--stem", default="drt_method_diagram", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output dpi.")
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=DEFAULT_FONT_DIR,
        help="Directory containing ComicNeue-Regular.ttf and ComicNeue-Bold.ttf.",
    )
    return parser.parse_args()


def register_showcase_font(font_dir: Path) -> str:
    global SHOWCASE_REGULAR_PROP, SHOWCASE_BOLD_PROP
    regular_path = font_dir / SHOWCASE_FONT_FILES[0]
    bold_path = font_dir / SHOWCASE_FONT_FILES[1]
    for filename in SHOWCASE_FONT_FILES:
        font_path = font_dir / filename
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    if regular_path.exists():
        SHOWCASE_REGULAR_PROP = font_manager.FontProperties(fname=str(regular_path))
    if bold_path.exists():
        SHOWCASE_BOLD_PROP = font_manager.FontProperties(fname=str(bold_path))
    if regular_path.exists():
        return font_manager.FontProperties(fname=str(regular_path)).get_name()
    return "DejaVu Sans"


def configure_matplotlib(font_dir: Path) -> None:
    font_name = register_showcase_font(font_dir)
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name, "DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        }
    )


def add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = "white",
    edge: str = BLACK,
    lw: float = 1.2,
    radius: float = 12,
    linestyle: str | tuple[int, tuple[int, ...]] = "solid",
    alpha: float = 1.0,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12,
    color: str = TEXT,
    weight: str = "bold",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.12,
    zorder: int = 5,
    stroke: float | None = None,
) -> None:
    font_kwargs = {}
    if weight == "bold" and SHOWCASE_BOLD_PROP is not None:
        font_kwargs["fontproperties"] = SHOWCASE_BOLD_PROP
    elif weight != "bold" and SHOWCASE_REGULAR_PROP is not None:
        font_kwargs["fontproperties"] = SHOWCASE_REGULAR_PROP
    else:
        font_kwargs["fontweight"] = weight
    rendered_size = max(size, MIN_RENDERED_TEXT_SIZE)
    if stroke is None:
        stroke = max(MIN_TEXT_STROKE, min(MAX_TEXT_STROKE, rendered_size * 0.032))
    ax.text(
        x,
        y + TEXT_Y_NUDGE,
        text,
        fontsize=rendered_size * TEXT_SCALE,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
        path_effects=[
            path_effects.Stroke(linewidth=stroke, foreground=color),
            path_effects.Normal(),
        ],
        **font_kwargs,
    )


def add_box_text(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    dx: float = 0,
    dy: float = 0,
    size: float = 12,
    color: str = TEXT,
    weight: str = "bold",
    ha: str = "center",
    linespacing: float = 1.08,
    zorder: int = 5,
) -> None:
    add_text(
        ax,
        x + w / 2 + dx,
        y + h / 2 + dy,
        text,
        size=size,
        color=color,
        weight=weight,
        ha=ha,
        va="center",
        linespacing=linespacing,
        zorder=zorder,
    )


def add_bullet(ax: plt.Axes, x: float, y: float, text: str, *, size: float = 13.2) -> None:
    ax.add_patch(Circle((x, y), 2.6, facecolor=BLACK, edgecolor=BLACK, lw=0, zorder=5))
    add_text(ax, x + 16, y, text, size=size, color=TEXT)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLACK,
    lw: float = 1.35,
    mutation: float = 13,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 4,
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
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_stage_badge(ax: plt.Axes, x: float, y: float, text: str, *, color: str = ORANGE, edge: str = ORANGE_DARK) -> None:
    add_round_box(ax, x, y, 78, 31, face=color, edge=edge, lw=0.8, radius=9, zorder=4)
    add_box_text(ax, x, y, 78, 31, text, size=14.2, color="white")


def add_number_circle(ax: plt.Axes, cx: float, cy: float, num: str, *, color: str) -> None:
    ax.add_patch(Circle((cx, cy), 14.5, facecolor=color, edgecolor=color, lw=1.0, zorder=5))
    add_text(ax, cx, cy - 0.5, num, size=14, color="white", weight="bold", ha="center")


def add_tag(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    w: float = 76,
    h: float = 30,
    fill: str | None = None,
    size: float = 11.2,
) -> None:
    fill = fill or "#f8fbff"
    add_round_box(ax, x, y, w, h, face=fill, edge=color, lw=1.0, radius=6, zorder=3)
    add_box_text(ax, x, y, w, h, text, size=size, color=color)


def add_check_mark(ax: plt.Axes, x: float, y: float, size: float, *, color: str = "white", lw: float = 2.0) -> None:
    ax.add_line(Line2D([x, x + size * 0.34, x + size], [y, y + size * 0.38, y - size * 0.44], color=color, lw=lw, solid_capstyle="round", zorder=7))


def _parse_svg_transform(transform: str | None) -> tuple[float, float, float, float, float, float]:
    if not transform:
        return (1, 0, 0, 1, 0, 0)
    matrix_match = re.search(r"matrix\(([^)]+)\)", transform)
    if matrix_match:
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", matrix_match.group(1))]
        if len(values) >= 6:
            return tuple(values[:6])  # type: ignore[return-value]
    translate_match = re.search(r"translate\(([^)]+)\)", transform)
    if translate_match:
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", translate_match.group(1))]
        tx = values[0] if values else 0.0
        ty = values[1] if len(values) > 1 else 0.0
        return (1, 0, 0, 1, tx, ty)
    return (1, 0, 0, 1, 0, 0)


def _apply_svg_matrix(x: float, y: float, matrix: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_svg_path(d: str) -> tuple[list[tuple[float, float]], list[int]]:
    tokens = re.findall(r"[MmLlCcZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    i = 0
    cmd = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)

    def is_command(value: str) -> bool:
        return bool(re.fullmatch(r"[MmLlCcZz]", value))

    def number() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        if is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        if cmd in {"M", "m"}:
            first = True
            while i + 1 < len(tokens) and not is_command(tokens[i]):
                x, y = number(), number()
                if cmd == "m":
                    x += current[0]
                    y += current[1]
                vertices.append((x, y))
                codes.append(MplPath.MOVETO if first else MplPath.LINETO)
                current = (x, y)
                if first:
                    start = current
                first = False
        elif cmd in {"L", "l"}:
            while i + 1 < len(tokens) and not is_command(tokens[i]):
                x, y = number(), number()
                if cmd == "l":
                    x += current[0]
                    y += current[1]
                vertices.append((x, y))
                codes.append(MplPath.LINETO)
                current = (x, y)
        elif cmd in {"C", "c"}:
            while i + 5 < len(tokens) and not is_command(tokens[i]):
                points = [(number(), number()), (number(), number()), (number(), number())]
                if cmd == "c":
                    points = [(current[0] + x, current[1] + y) for x, y in points]
                vertices.extend(points)
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                current = points[-1]
        elif cmd in {"Z", "z"}:
            vertices.append(start)
            codes.append(MplPath.CLOSEPOLY)
            current = start
            cmd = ""
        else:
            break
    return vertices, codes


@functools.lru_cache(maxsize=1)
def _load_qwen_logo_paths() -> tuple[tuple[tuple[list[tuple[float, float]], list[int], tuple[float, float, float]], ...], tuple[float, float, float, float]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(QWEN_LOGO_PATH).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    items = []
    xs: list[float] = []
    ys: list[float] = []
    for group in root.findall("svg:g", ns):
        if group.find("svg:text", ns) is not None:
            continue
        path = group.find("svg:path", ns)
        if path is None:
            continue
        d = path.attrib.get("d", "")
        vertices, codes = _parse_svg_path(d)
        path_matrix = _parse_svg_transform(path.attrib.get("transform"))
        group_matrix = _parse_svg_transform(group.attrib.get("transform"))
        transformed = []
        for x, y in vertices:
            px, py = _apply_svg_matrix(x, y, path_matrix)
            gx, gy = _apply_svg_matrix(px, py, group_matrix)
            transformed.append((gx, gy))
            xs.append(gx)
            ys.append(gy)
        style = path.attrib.get("style", "")
        fill_match = re.search(r"fill:\s*rgb\((\d+),(\d+),(\d+)\)", style)
        fill = tuple(int(v) / 255 for v in fill_match.groups()) if fill_match else (0.1, 0.2, 0.5)
        items.append((transformed, codes, fill))
        if len(items) == 3:
            break
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return tuple(items), bbox


def draw_qwen_logo(ax: plt.Axes, cx: float, cy: float, size: float) -> None:
    paths, (min_x, min_y, max_x, max_y) = _load_qwen_logo_paths()
    width = max_x - min_x
    height = max_y - min_y
    scale = size / max(width, height)
    left = cx - width * scale / 2
    top = cy - height * scale / 2
    for vertices, codes, fill in paths:
        mapped = [(left + (x - min_x) * scale, top + (y - min_y) * scale) for x, y in vertices]
        ax.add_patch(PathPatch(MplPath(mapped, codes), facecolor=fill, edgecolor="none", lw=0, zorder=6))


@functools.lru_cache(maxsize=1)
def _load_qwen_icon_rgba() -> np.ndarray:
    return np.array(Image.open(QWEN_ICON_PATH).convert("RGBA"))


def draw_qwen_icon(ax: plt.Axes, cx: float, cy: float, size: float) -> None:
    arr = _load_qwen_icon_rgba()
    h, w = arr.shape[:2]
    aspect = w / h
    draw_w = size * aspect
    draw_h = size
    ax.imshow(arr, extent=(cx - draw_w / 2, cx + draw_w / 2, cy + draw_h / 2, cy - draw_h / 2), zorder=8)


@functools.lru_cache(maxsize=1)
def _load_gpt_icon_rgba() -> np.ndarray:
    img = Image.open(GPT_ICON_PATH).convert("RGBA")
    arr = np.array(img)
    white = np.all(arr[:, :, :3] > 245, axis=2)
    arr[:, :, 3] = np.where(white, 0, arr[:, :, 3])
    ink = arr[:, :, 3] > 0
    arr[:, :, :3] = np.where(ink[:, :, None], np.array([32, 33, 36], dtype=np.uint8), arr[:, :, :3])
    return arr


def draw_gpt_icon(ax: plt.Axes, cx: float, cy: float, size: float) -> None:
    arr = _load_gpt_icon_rgba()
    h, w = arr.shape[:2]
    aspect = w / h
    draw_w = size * aspect
    draw_h = size
    ax.imshow(arr, extent=(cx - draw_w / 2, cx + draw_w / 2, cy + draw_h / 2, cy - draw_h / 2), zorder=8)


def draw_warning(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = RED) -> None:
    h = size * 0.9
    pts = [(cx, cy - h / 2), (cx - size * 0.48, cy + h / 2), (cx + size * 0.48, cy + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, lw=0, zorder=5))
    add_text(ax, cx, cy + 2, "!", size=size * 0.62, color="white", weight="bold", ha="center")


def draw_atom(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = TEAL) -> None:
    ax.add_patch(Ellipse((cx, cy), size * 0.95, size * 0.33, angle=0, fill=False, edgecolor=color, lw=2.0, zorder=5))
    ax.add_patch(Ellipse((cx, cy), size * 0.95, size * 0.33, angle=60, fill=False, edgecolor=color, lw=2.0, zorder=5))
    ax.add_patch(Ellipse((cx, cy), size * 0.95, size * 0.33, angle=-60, fill=False, edgecolor=color, lw=2.0, zorder=5))
    ax.add_patch(Circle((cx, cy), size * 0.09, facecolor=color, edgecolor=color, lw=0, zorder=6))


def draw_database(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = ORANGE, lw: float = 2.0) -> None:
    w = size * 0.72
    h = size * 0.82
    ax.add_patch(Rectangle((cx - w / 2, cy - h * 0.28), w, h * 0.56, facecolor="white", edgecolor=color, lw=lw, zorder=5))
    ax.add_patch(Ellipse((cx, cy - h * 0.28), w, h * 0.28, facecolor="white", edgecolor=color, lw=lw, zorder=6))
    ax.add_patch(Ellipse((cx, cy), w, h * 0.28, facecolor="white", edgecolor=color, lw=lw, zorder=6))
    ax.add_patch(Ellipse((cx, cy + h * 0.28), w, h * 0.28, facecolor="white", edgecolor=color, lw=lw, zorder=6))


def draw_eye(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = BLACK) -> None:
    ax.add_patch(Ellipse((cx, cy), size * 1.02, size * 0.55, fill=False, edgecolor=color, lw=1.45, zorder=6))
    ax.add_patch(Circle((cx, cy), size * 0.17, facecolor=color, edgecolor=color, lw=0, zorder=7))


def draw_bulb(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = BLACK) -> None:
    ax.add_patch(Circle((cx, cy - size * 0.1), size * 0.28, fill=False, edgecolor=color, lw=1.45, zorder=6))
    ax.add_line(Line2D([cx - size * 0.18, cx + size * 0.18], [cy + size * 0.22, cy + size * 0.22], color=color, lw=1.45, zorder=6))
    ax.add_line(Line2D([cx - size * 0.13, cx + size * 0.13], [cy + size * 0.36, cy + size * 0.36], color=color, lw=1.45, zorder=6))
    ax.add_line(Line2D([cx, cx], [cy + size * 0.06, cy + size * 0.22], color=color, lw=1.45, zorder=6))


def draw_check_circle(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = ORANGE) -> None:
    ax.add_patch(Circle((cx, cy), size * 0.45, facecolor="white", edgecolor=color, lw=1.8, zorder=6))
    add_check_mark(ax, cx - size * 0.18, cy + size * 0.02, size * 0.38, color=color, lw=2.0)


def draw_anchor(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = ORANGE) -> None:
    ax.add_patch(Circle((cx, cy - size * 0.34), size * 0.1, fill=False, edgecolor=color, lw=1.7, zorder=6))
    ax.add_line(Line2D([cx, cx], [cy - size * 0.23, cy + size * 0.36], color=color, lw=1.8, zorder=6))
    ax.add_line(Line2D([cx - size * 0.23, cx + size * 0.23], [cy - size * 0.02, cy - size * 0.02], color=color, lw=1.7, zorder=6))
    ax.add_patch(FancyArrowPatch((cx - size * 0.38, cy + size * 0.1), (cx - size * 0.04, cy + size * 0.36), arrowstyle="-", connectionstyle="arc3,rad=0.45", color=color, lw=1.8, zorder=6))
    ax.add_patch(FancyArrowPatch((cx + size * 0.38, cy + size * 0.1), (cx + size * 0.04, cy + size * 0.36), arrowstyle="-", connectionstyle="arc3,rad=-0.45", color=color, lw=1.8, zorder=6))
    ax.add_line(Line2D([cx - size * 0.38, cx - size * 0.30], [cy + size * 0.1, cy + size * 0.2], color=color, lw=1.8, zorder=6))
    ax.add_line(Line2D([cx + size * 0.38, cx + size * 0.30], [cy + size * 0.1, cy + size * 0.2], color=color, lw=1.8, zorder=6))


def draw_network(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = GREEN) -> None:
    r = size * 0.42
    pts = [
        (cx + r * math.cos(angle), cy + r * math.sin(angle))
        for angle in [-math.pi / 2, -math.pi / 6, math.pi / 6, math.pi / 2, 5 * math.pi / 6, 7 * math.pi / 6]
    ]
    lines = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 2), (1, 4)]
    for i, j in lines:
        ax.add_line(Line2D([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=color, lw=1.7, zorder=5))
    for px, py in pts:
        ax.add_patch(Circle((px, py), size * 0.075, facecolor="white", edgecolor=color, lw=2.0, zorder=6))


def draw_image_icon(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = BLUE) -> None:
    w = size * 0.95
    h = size * 0.72
    add_round_box(ax, cx - w / 2, cy - h / 2, w, h, face="#f7fbff", edge=color, lw=1.9, radius=4, zorder=5)
    ax.add_patch(Circle((cx + w * 0.2, cy - h * 0.18), size * 0.07, facecolor=color, edgecolor=color, lw=0, zorder=6))
    ax.add_patch(Polygon([(cx - w * 0.38, cy + h * 0.28), (cx - w * 0.12, cy - h * 0.02), (cx + w * 0.05, cy + h * 0.18), (cx + w * 0.18, cy + h * 0.02), (cx + w * 0.38, cy + h * 0.28)], closed=True, facecolor=color, edgecolor=color, lw=0, zorder=6))


def draw_question(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = BLUE) -> None:
    ax.add_patch(Circle((cx, cy), size * 0.43, facecolor="#f7fbff", edgecolor=color, lw=1.8, zorder=5))
    ax.text(
        cx,
        cy + size * 0.01,
        "?",
        fontsize=size * 0.58 * TEXT_SCALE,
        color=color,
        ha="center",
        va="center",
        zorder=6,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        path_effects=[
            path_effects.Stroke(linewidth=0.28, foreground=color),
            path_effects.Normal(),
        ],
    )


def draw_refresh(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = PURPLE) -> None:
    r = size * 0.32
    add_arrow(ax, (cx - r, cy - r * 0.35), (cx + r * 0.92, cy - r * 0.12), color=color, lw=2.0, mutation=12, connectionstyle="arc3,rad=-0.6", zorder=6)
    add_arrow(ax, (cx + r, cy + r * 0.35), (cx - r * 0.92, cy + r * 0.12), color=color, lw=2.0, mutation=12, connectionstyle="arc3,rad=-0.6", zorder=6)


def draw_sparkle(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = GREEN) -> None:
    def diamond(x: float, y: float, s: float) -> None:
        ax.add_patch(Polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], closed=True, facecolor=color, edgecolor=color, lw=0, zorder=6))

    diamond(cx, cy, size * 0.15)
    diamond(cx - size * 0.28, cy - size * 0.2, size * 0.08)
    diamond(cx + size * 0.28, cy + size * 0.18, size * 0.07)


def draw_star(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str) -> None:
    points = []
    for i in range(10):
        radius = size if i % 2 == 0 else size * 0.42
        angle = -math.pi / 2 + i * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, lw=0, zorder=7))


def draw_target(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = GREEN) -> None:
    ax.add_patch(Circle((cx, cy), size * 0.34, fill=False, edgecolor=color, lw=1.8, zorder=6))
    ax.add_patch(Circle((cx, cy), size * 0.18, fill=False, edgecolor=color, lw=1.8, zorder=6))
    ax.add_line(Line2D([cx - size * 0.46, cx + size * 0.46], [cy, cy], color=color, lw=1.6, zorder=6))
    ax.add_line(Line2D([cx, cx], [cy - size * 0.46, cy + size * 0.46], color=color, lw=1.6, zorder=6))
    ax.add_patch(Circle((cx, cy), size * 0.055, facecolor=color, edgecolor=color, lw=0, zorder=7))


def draw_nodes(ax: plt.Axes, cx: float, cy: float, size: float, *, color: str = GREEN) -> None:
    pts = [(cx, cy - size * 0.36), (cx - size * 0.34, cy + size * 0.26), (cx + size * 0.34, cy + size * 0.26)]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ax.add_line(Line2D([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=color, lw=1.9, zorder=6))
    for px, py in pts:
        ax.add_patch(Circle((px, py), size * 0.12, facecolor="white", edgecolor=color, lw=2.0, zorder=7))


def draw_header_tag(ax: plt.Axes, x: float, y: float, w: float, text: str, *, edge: str) -> None:
    add_round_box(ax, x, y, w, 29, face="white", edge=edge, lw=1.0, radius=12, zorder=2)
    add_text(ax, x + w / 2, y + 14.5, text, size=10.0, color=BLACK, ha="center")


def draw_icon_box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, icon_fn, icon_color: str, *, edge: str, face: str = "white", label_size: float = 10.5) -> None:
    add_round_box(ax, x, y, w, h, face=face, edge=edge, lw=1.0, radius=9, zorder=2)
    add_text(ax, x + w / 2, y + 36, label, size=label_size, color=BLACK, weight="bold", ha="center", linespacing=1.1)
    icon_fn(ax, x + w / 2, y + h - 38, 41, color=icon_color)


def draw_left_column(ax: plt.Axes) -> None:
    add_round_box(ax, 14, 16, 340, 170, face="white", edge=RED, lw=1.35, radius=18)
    draw_warning(ax, 45, 51, 35, color=RED)
    add_text(ax, 72, 52, "Problem with Long-CoT", size=17, color=RED, weight="bold")
    add_bullet(ax, 42, 92, "Visual Grounding Degradation", size=12.6)
    add_bullet(ax, 42, 126, "Low Information Density Reasoning", size=12.6)
    add_bullet(ax, 42, 160, "Token-Efficiency Collapse", size=12.6)

    add_round_box(ax, 14, 205, 340, 432, face="white", edge=TEAL, lw=1.35, radius=15)
    draw_atom(ax, 46, 241, 36, color=TEAL)
    add_text(ax, 72, 241, "DRT Representation (Ours)", size=16.2, color=TEAL_DARK, weight="bold")
    add_bullet(ax, 50, 273, "Separate perception from deduction", size=11.7)
    add_bullet(ax, 50, 305, "Compress reasoning into dense\ntelegraphic traces", size=11.7)

    add_round_box(ax, 30, 334, 307, 128, face="#fbffff", edge="#8fd4d2", lw=1.1, radius=16, linestyle=(0, (4, 3)))
    add_tag(ax, 47, 346, "<visual>", color=TAG_VISUAL, w=90, h=31, fill="#f4fffb", size=10.5)
    ax.add_line(Line2D([152, 179], [361.5, 361.5], color=BLACK, lw=1.0, zorder=5))
    add_text(ax, 188, 361.5, "grounded\nobservations", size=12.0, linespacing=0.82)
    add_tag(ax, 47, 387, "<think>", color=TAG_THINK, w=90, h=31, fill="#f7fbff", size=10.5)
    ax.add_line(Line2D([152, 179], [402.5, 402.5], color=BLACK, lw=1.0, zorder=5))
    add_text(ax, 186, 402.5, "[Priors] -> dense\ntelegraphic reasoning", size=12.0, linespacing=0.88)
    add_tag(ax, 47, 428, "<answer>", color=TAG_ANSWER, w=90, h=31, fill="#faf8ff", size=10.5)
    ax.add_line(Line2D([152, 179], [443.5, 443.5], color=BLACK, lw=1.0, zorder=5))
    add_text(ax, 190, 443.5, "final answer", size=10.2)

    add_round_box(ax, 30, 479, 307, 132, face="#fbffff", edge="#c8e5e4", lw=1.1, radius=11)
    draw_target(ax, 55, 512, 28, color=TEAL)
    add_text(ax, 75, 512, "Outcome", size=15, color=TEAL_DARK, weight="bold")
    for y, label in [
        (543, "Faithful Visual Grounding"),
        (572, "Information-Dense Traces"),
        (596, "Token & Latency Reduction"),
    ]:
        ax.add_patch(Circle((55, y), 10, facecolor=TEAL, edgecolor=TEAL, lw=0, zorder=5))
        add_check_mark(ax, 50, y + 1, 9, color="white", lw=2.0)
        add_text(ax, 82, y, label, size=11.8, color=TEXT, weight="bold")


def draw_stage1(ax: plt.Axes) -> None:
    add_stage_badge(ax, 390, 15, "Stage 1", color=GREEN_DARK, edge=GREEN_DARK)
    add_text(ax, 479, 31, "SFT: Build DRT-format supervision", size=17.3, color=BLACK, weight="bold")
    draw_header_tag(ax, 390, 47, 656, "Transform raw CoT into DRT supervision for SFT initialization.", edge="#86cbb0")

    add_round_box(ax, 390, 81, 656, 202, face="white", edge=ORANGE, lw=1.4, radius=18)
    add_number_circle(ax, 418, 103, "1", color=ORANGE)
    add_text(ax, 442, 105, "DRT-SFT Data Construction", size=16.2, color=ORANGE_DARK, weight="bold")
    draw_star(ax, 710, 103, 7.2, color=ORANGE_DARK)

    add_round_box(ax, 404, 128, 84, 142, face="white", edge="#ffc48b", lw=1.0, radius=8)
    add_text(ax, 446, 172, "Raw CoT\nData", size=10.4, color=BLACK, weight="bold", ha="center")
    draw_database(ax, 446, 220, 43, color=ORANGE)
    add_arrow(ax, (489, 192), (507, 192), color=BLACK, mutation=12)

    add_round_box(ax, 506, 128, 144, 142, face="#fffdf9", edge="#ffc48b", lw=1.1, radius=12, linestyle=(0, (4, 3)))
    add_text(ax, 578, 144, "Deconstruct", size=10.4, color=ORANGE_DARK, weight="bold", ha="center")
    draw_gpt_icon(ax, 633, 144, 17)
    step_y = [157, 194, 231]
    labels = ["Visual\nObservation", "Logical\nReasoning", "Final\nAnswer"]
    icon_fns = [draw_eye, draw_bulb, draw_check_circle]
    for yy, label, fn in zip(step_y, labels, icon_fns):
        add_round_box(ax, 518, yy, 119, 31, face="white", edge="#ffcfa4", lw=0.9, radius=6)
        fn(ax, 536, yy + 15.5, 22, color=BLACK if fn != draw_check_circle else ORANGE)
        add_text(ax, 596, yy + 15.5, label, size=10.4, color=BLACK, weight="bold", ha="center", linespacing=0.68, stroke=0.20)
    add_arrow(ax, (651, 192), (672, 192), color=BLACK, mutation=12)

    add_round_box(ax, 671, 128, 108, 142, face="#fffdf9", edge="#ffc48b", lw=1.1, radius=12, linestyle=(0, (4, 3)))
    add_text(ax, 720, 144, "Compress", size=10.4, color=ORANGE_DARK, weight="bold", ha="center")
    draw_gpt_icon(ax, 766, 144, 16)
    add_tag(ax, 687, 158, "<visual>", color=TAG_VISUAL, w=76, h=30, fill="#f3fffb", size=10.5)
    add_tag(ax, 687, 197, "<think>", color=TAG_THINK, w=76, h=30, fill="#f7fbff", size=10.5)
    add_tag(ax, 687, 236, "<answer>", color=TAG_ANSWER, w=76, h=30, fill="#faf8ff", size=10.4)
    add_arrow(ax, (780, 192), (796, 192), color=BLACK, mutation=12)

    add_round_box(ax, 795, 128, 118, 142, face="#fffdf9", edge="#ffc48b", lw=1.1, radius=12, linestyle=(0, (4, 3)))
    add_text(ax, 846, 151, "Add explicit\n[Priors] anchor", size=10.4, color=BLACK, weight="bold", ha="center")
    draw_gpt_icon(ax, 901, 144, 16)
    draw_anchor(ax, 854, 212, 51, color=ORANGE)
    add_arrow(ax, (914, 192), (929, 192), color=BLACK, mutation=12)

    add_round_box(ax, 928, 128, 104, 142, face="white", edge="#ffc48b", lw=1.0, radius=8)
    add_text(ax, 980, 151, "Output", size=10.5, color=ORANGE_DARK, weight="bold", ha="center")
    draw_database(ax, 980, 181, 42, color=ORANGE)
    add_text(ax, 980, 224, "DRT-SFT\ndataset", size=10.5, color=BLACK, weight="bold", ha="center")


def draw_sft_initialization(ax: plt.Axes) -> None:
    add_round_box(ax, 1082, 47, 590, 240, face="#fdfffe", edge=GREEN_DARK, lw=1.35, radius=17)
    add_number_circle(ax, 1119, 74, "2", color=GREEN_DARK)
    add_text(ax, 1151, 75, "SFT Initialization", size=17, color=GREEN, weight="bold")
    draw_header_tag(ax, 1146, 89, 427, "Use DRT-style data for SFT to internalize dense reasoning.", edge="#86cbb0")

    draw_icon_box(ax, 1113, 132, 117, 126, "DRT-SFT\ndataset", draw_database, GREEN, edge="#9ed7c0", face="#fbfffd", label_size=10.6)
    add_arrow(ax, (1236, 195), (1277, 195), color=BLACK, mutation=13)
    add_round_box(ax, 1285, 132, 170, 126, face="white", edge="#9ed7c0", lw=1.0, radius=9)
    add_text(ax, 1370, 165, "Base model\nQwen3-VL-8B-\nInstruct", size=10.6, color=BLACK, weight="bold", ha="center")
    draw_qwen_icon(ax, 1370, 224, 54)
    add_arrow(ax, (1462, 195), (1494, 195), color=BLACK, mutation=13)
    add_round_box(ax, 1495, 132, 141, 126, face="white", edge="#9ed7c0", lw=1.0, radius=9)
    add_text(ax, 1565, 154, "Output", size=12.0, color=GREEN, weight="bold", ha="center")
    draw_network(ax, 1565, 196, 36, color=GREEN)
    add_text(ax, 1565, 236, "DRT-SFT\nmodel", size=12.0, color=BLACK, weight="bold", ha="center", linespacing=0.82)


def draw_stage2(ax: plt.Axes) -> None:
    add_stage_badge(ax, 390, 298, "Stage 2", color=PURPLE_DARK, edge=PURPLE_DARK)
    add_text(ax, 479, 314, "RL: Improve reasoning reliability under compact traces", size=16.0, color=BLACK, weight="bold")
    draw_header_tag(ax, 390, 331, 610, "Verified trajectories + process rewards improve compact reasoning.", edge="#c6b7ff")

    add_round_box(ax, 390, 364, 610, 273, face="white", edge=ORANGE, lw=1.4, radius=17)
    add_number_circle(ax, 418, 388, "3", color=ORANGE)
    add_text(ax, 442, 389, "DRT-RL Data Construction", size=16.2, color=ORANGE_DARK, weight="bold")
    draw_star(ax, 695, 387, 7.2, color=ORANGE_DARK)
    add_round_box(ax, 410, 402, 500, 27, face="white", edge="#ffc48b", lw=1.0, radius=8)
    add_text(ax, 660, 416, "Generate & verify process-correct trajectories for RL training.", size=12.0, color=BLACK, weight="bold", ha="center")

    add_round_box(ax, 412, 440, 98, 168, face="white", edge="#ffc48b", lw=1.0, radius=8)
    add_text(ax, 461, 464, "Source\nData", size=12.0, color=ORANGE_DARK, weight="bold", ha="center", linespacing=0.82)
    draw_database(ax, 461, 504, 44, color=ORANGE)
    add_text(ax, 461, 561, "Vision-R1-RL\n+ DAPO data", size=12.0, color=BLACK, weight="bold", ha="center", linespacing=0.82)
    add_arrow(ax, (512, 524), (520, 524), color=BLACK, mutation=12)

    add_round_box(ax, 520, 440, 190, 168, face="#fffdf9", edge="#ffc48b", lw=1.0, radius=8)
    draw_gpt_icon(ax, 692, 457, 22)
    add_text(ax, 611, 457, "Generate", size=10.5, color=ORANGE_DARK, weight="bold", ha="center")
    add_text(ax, 611, 486, "GPT-5.1 samples\nstepwise trajectories.", size=10.4, color=BLACK, weight="bold", ha="center", linespacing=0.86)
    add_round_box(ax, 526, 507, 178, 98, face="white", edge="#ffc48b", lw=0.9, radius=6)
    add_text(ax, 611, 519, "Example trajectory", size=10.4, color=ORANGE_DARK, weight="bold", ha="center")
    add_text(ax, 535, 537, "1. Observe: A has 4 silos;", size=12.0, color=BLACK, stroke=0.18)
    add_text(ax, 552, 551, "B has 1.", size=12.0, color=BLACK, stroke=0.18)
    add_text(ax, 535, 565, "2. Prior: V = pi*r^2*h.", size=12.0, color=BLACK, stroke=0.18)
    add_text(ax, 535, 579, "3. Compute A/B volumes.", size=12.0, color=BLACK, stroke=0.18)
    add_text(ax, 611, 594, "Answer: C", size=10.4, color=BLACK, weight="bold", ha="center", stroke=0.19)
    add_arrow(ax, (711, 524), (722, 524), color=BLACK, mutation=12)

    add_round_box(ax, 722, 440, 176, 168, face="#fffdf9", edge="#ffc48b", lw=1.0, radius=8)
    draw_gpt_icon(ax, 879, 457, 22)
    add_text(ax, 810, 457, "Verify", size=10.5, color=ORANGE_DARK, weight="bold", ha="center")
    add_text(ax, 810, 484, "GPT-5.1 verifier checks:", size=10.4, color=BLACK, weight="bold", ha="center")
    for yy, label in [(510, "Semantic\ncorrectness"), (547, "Visual fidelity"), (584, "Reasoning\nfaithfulness")]:
        add_round_box(ax, 727, yy - 18, 166, 36, face="white", edge="#ffc48b", lw=0.9, radius=6)
        draw_check_circle(ax, 746, yy, 23, color=ORANGE)
        add_text(ax, 765, yy, label, size=12.0, color=BLACK, weight="bold", linespacing=0.88, stroke=0.18)
    add_arrow(ax, (899, 524), (912, 524), color=BLACK, mutation=12)

    add_round_box(ax, 912, 440, 72, 168, face="white", edge="#ffc48b", lw=1.0, radius=8)
    add_text(ax, 948, 464, "Output", size=10.5, color=ORANGE_DARK, weight="bold", ha="center")
    draw_database(ax, 948, 505, 44, color=ORANGE)
    add_text(ax, 948, 563, "DRT-RL\ndataset", size=10.5, color=BLACK, weight="bold", ha="center")


def draw_reward_rl(ax: plt.Axes) -> None:
    add_round_box(ax, 1024, 307, 653, 331, face="#fdfcff", edge=PURPLE, lw=1.35, radius=15)
    add_number_circle(ax, 1049, 329, "4", color=PURPLE_DARK)
    add_text(ax, 1080, 330, "Reward Design + Process RL", size=15.9, color=PURPLE, weight="bold")
    draw_star(ax, 1347, 328, 7.2, color=PURPLE)
    add_round_box(ax, 1035, 348, 493, 31, face="white", edge="#c6b7ff", lw=1.0, radius=15)
    add_text(ax, 1281, 363.5, "Verified trajectories + process-level rewards improve reasoning.", size=12.0, color=BLACK, weight="bold", ha="center")

    add_round_box(ax, 1045, 388, 235, 238, face="white", edge="#c6b7ff", lw=1.0, radius=9)
    add_text(ax, 1162.5, 405, "Judge (Process Signals)", size=12.0, color=PURPLE, weight="bold", ha="center")
    draw_qwen_icon(ax, 1090, 447, 48)
    add_text(ax, 1180, 441, "Qwen3-235B-A22B-\nInstruct", size=12.0, color=BLACK, weight="bold", ha="center", linespacing=0.82)
    for yy, key, body in [
        (497, "N_match", "format match"),
        (534, "N_hall", "hallucination\npenalty"),
        (571, "N_eff", "efficiency score"),
        (608, "N_deep", "exploration depth"),
    ]:
        row_h = 34 if "\n" in body else 28
        add_round_box(ax, 1054, yy - row_h / 2, 218, row_h, face="#fbfaff", edge="#d7ccff", lw=0.9, radius=7)
        add_text(ax, 1070, yy, key, size=12.0, color=PURPLE, weight="bold", ha="left")
        add_text(ax, 1148, yy, body, size=12.0, color=BLACK, ha="left", linespacing=0.88, stroke=0.20)

    add_arrow(ax, (1280, 431), (1290, 431), color=BLACK, mutation=13)

    add_round_box(ax, 1290, 388, 226, 238, face="white", edge="#c6b7ff", lw=1.0, radius=9)
    add_text(ax, 1403, 405, "Reward Function", size=12.0, color=PURPLE, weight="bold", ha="center")
    add_round_box(ax, 1300, 417, 206, 36, face="#fbfaff", edge="#c6b7ff", lw=0.9, radius=7)
    add_text(
        ax,
        1403,
        435,
        r"$R = R_{\mathrm{format}} + R_{\mathrm{main}} + R_{\mathrm{bonus}}$",
        size=12.0,
        color=BLACK,
        weight="bold",
        ha="center",
        stroke=0.12,
    )
    for yy, title, body in [
        (478, r"Format Reward  $R_{\mathrm{format}}$", "stay within DRT format"),
        (532, r"Main Reward  $R_{\mathrm{main}}$", "be correct and faithful"),
        (586, r"Step Bonus  $R_{\mathrm{bonus}}$", "thorough + exploratory"),
    ]:
        add_round_box(ax, 1300, yy - 18, 206, 42, face="white", edge="#d7ccff", lw=0.9, radius=7)
        add_text(ax, 1403, yy - 5, title, size=12.0, color=PURPLE, weight="bold", ha="center")
        add_text(ax, 1403, yy + 12, body, size=12.0, color=BLACK, weight="bold", ha="center", linespacing=0.76, stroke=0.20)

    add_round_box(ax, 1532, 316, 132, 230, face="white", edge="#c6b7ff", lw=1.0, radius=9)
    add_text(ax, 1598, 339, "RL Optimization\n(GRPO)", size=10.6, color=PURPLE, weight="bold", ha="center", linespacing=0.90)
    draw_refresh(ax, 1598, 384, 44, color=PURPLE)
    add_round_box(ax, 1542, 410, 114, 56, face="#fbfffd", edge="#9ed7c0", lw=0.9, radius=7)
    draw_network(ax, 1560, 438, 26, color=GREEN)
    add_text(ax, 1620, 438, "DRT-SFT\nmodel", size=10.4, color=BLACK, weight="bold", ha="center", linespacing=0.88, stroke=0.20)
    add_text(ax, 1598, 475, "+", size=16.5, color=BLACK, weight="bold", ha="center")
    add_round_box(ax, 1542, 483, 114, 56, face="#fffdf9", edge="#ffc48b", lw=0.9, radius=7)
    draw_database(ax, 1560, 511, 26, color=ORANGE)
    add_text(ax, 1624, 511, "DRT-RL\ndataset", size=10.4, color=BLACK, weight="bold", ha="center", linespacing=0.82, stroke=0.20)
    add_arrow(ax, (1598, 546), (1598, 558), color=PURPLE, mutation=11)
    add_round_box(ax, 1532, 558, 132, 68, face="white", edge="#c6b7ff", lw=1.0, radius=8)
    draw_network(ax, 1560, 592, 26, color=PURPLE)
    add_text(ax, 1625, 580, "Output", size=10.4, color=PURPLE, weight="bold", ha="center", stroke=0.20)
    add_text(ax, 1625, 606, "DRT-RL\nmodel", size=10.4, color=BLACK, weight="bold", ha="center", linespacing=0.82, stroke=0.20)

    add_arrow(ax, (1516, 431), (1532, 431), color=BLACK, mutation=13)


def draw_inference(ax: plt.Axes) -> None:
    add_round_box(ax, 1720, 47, 292, 591, face="#f8fbff", edge=BLUE, lw=1.35, radius=13)
    add_round_box(ax, 1720, 47, 292, 55, face=BLUE, edge=BLUE, lw=0, radius=13, zorder=2)
    ax.add_patch(Rectangle((1720, 90), 292, 20, facecolor=BLUE, edgecolor=BLUE, lw=0, zorder=2))
    ax.add_patch(Circle((1754, 76), 15, facecolor=BLUE, edgecolor="white", lw=1.5, zorder=5))
    add_text(ax, 1754, 76, "5", size=15, color="white", weight="bold", ha="center")
    add_text(ax, 1778, 76, "DRT-trained Inference", size=16.0, color="white", weight="bold")

    add_round_box(ax, 1737, 116, 258, 68, face="white", edge="#8ebcff", lw=1.0, radius=8)
    add_text(ax, 1798, 135, "New Question", size=12.0, color=BLUE, weight="bold", ha="center")
    add_text(ax, 1866, 135, "+", size=12.0, color=BLUE, weight="bold", ha="center")
    add_text(ax, 1930, 135, "Image", size=12.0, color=BLUE, weight="bold", ha="center")
    draw_question(ax, 1815, 163, 39, color=BLUE)
    draw_image_icon(ax, 1927, 163, 40, color=BLUE)
    add_arrow(ax, (1866, 184), (1866, 204), color=BLACK, mutation=12)

    add_round_box(ax, 1737, 204, 258, 52, face="white", edge="#8ebcff", lw=1.0, radius=8)
    draw_network(ax, 1777, 229, 40, color=BLUE)
    add_text(ax, 1877, 230, "DRT-RL\nmodel", size=11.0, color=BLUE, weight="bold", ha="center", linespacing=0.86)
    add_arrow(ax, (1866, 256), (1866, 279), color=BLACK, mutation=12)

    add_round_box(ax, 1737, 280, 258, 205, face="white", edge="#8ebcff", lw=1.1, radius=13, linestyle=(0, (4, 3)))
    add_text(ax, 1866, 298, "Structured Output", size=11.0, color=BLUE, weight="bold", ha="center")
    rows = [
        (314, "<visual>", TAG_VISUAL, "#f4fffb", "grounded facts"),
        (376, "<think>", TAG_THINK, "#f7fbff", "dense telegraphic\nreasoning"),
        (438, "<answer>", TAG_ANSWER, "#faf8ff", "final prediction"),
    ]
    for y, tag, color, fill, desc in rows:
        add_round_box(ax, 1750, y, 230, 45, face="white", edge="#b5d0ff", lw=0.9, radius=7)
        add_tag(ax, 1756, y + 8, tag, color=color, w=79, h=29, fill=fill, size=10.5)
        add_text(ax, 1845, y + 22.5, desc, size=10.6, color=BLACK, ha="left")

    for x, label, fn in [
        (1730, "concise", draw_sparkle),
        (1828, "grounded", draw_target),
        (1925, "structured", draw_nodes),
    ]:
        add_round_box(ax, x, 520, 80, 92, face="white", edge=GREEN, lw=1.2, radius=9)
        fn(ax, x + 40, 559, 43, color=GREEN)
        add_text(ax, x + 40, 595, label, size=11.0, color=GREEN, weight="bold", ha="center")


def draw_connectors(ax: plt.Axes) -> None:
    add_arrow(ax, (354, 152), (389, 152), color=BLACK, lw=1.35, mutation=13)
    add_arrow(ax, (1046, 190), (1082, 190), color=BLACK, lw=1.35, mutation=13)
    add_arrow(ax, (1000, 524), (1024, 524), color=BLACK, lw=1.35, mutation=13)
    add_arrow(ax, (1677, 474), (1720, 474), color=BLACK, lw=1.35, mutation=13)
    add_arrow(ax, (1370, 287), (1370, 307), color=BLACK, lw=1.2, mutation=12)


def draw_figure(font_dir: Path = DEFAULT_FONT_DIR) -> plt.Figure:
    configure_matplotlib(font_dir)
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, facecolor="white", edgecolor="none", zorder=0))

    draw_left_column(ax)
    draw_stage1(ax)
    draw_sft_initialization(ax)
    draw_stage2(ax)
    draw_reward_rl(ax)
    draw_inference(ax)
    draw_connectors(ax)
    return fig


def icon_svg(name: str, body: str, color: str) -> str:
    return textwrap.dedent(
        f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="{name}">
          <g fill="none" stroke="{color}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
        {body}
          </g>
        </svg>
        """
    )


def write_icon_assets(assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    icons = {
        "warning.svg": textwrap.dedent(
            f"""\
            <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="warning">
              <path d="M48 10 L88 82 H8 Z" fill="{RED}"/>
              <path d="M48 32 V58" stroke="white" stroke-width="7" stroke-linecap="round"/>
              <circle cx="48" cy="70" r="4" fill="white"/>
            </svg>
            """
        ),
        "atom.svg": icon_svg("atom", '    <ellipse cx="48" cy="48" rx="37" ry="13"/>\n    <ellipse cx="48" cy="48" rx="37" ry="13" transform="rotate(60 48 48)"/>\n    <ellipse cx="48" cy="48" rx="37" ry="13" transform="rotate(-60 48 48)"/>\n    <circle cx="48" cy="48" r="5" fill="' + TEAL + '" stroke="none"/>', TEAL),
        "database.svg": icon_svg("database", '    <ellipse cx="48" cy="24" rx="28" ry="12"/>\n    <path d="M20 24 V72 C20 79 76 79 76 72 V24"/>\n    <path d="M20 48 C20 55 76 55 76 48"/>\n    <path d="M20 72 C20 79 76 79 76 72"/>', ORANGE),
        "eye.svg": icon_svg("eye", '    <path d="M10 48 C22 27 74 27 86 48 C74 69 22 69 10 48 Z"/>\n    <circle cx="48" cy="48" r="10" fill="' + BLACK + '" stroke="none"/>', BLACK),
        "bulb.svg": icon_svg("bulb", '    <path d="M32 41 A16 16 0 1 1 64 41 C64 51 56 55 56 63 H40 C40 55 32 51 32 41 Z"/>\n    <path d="M40 72 H56"/>\n    <path d="M42 82 H54"/>', BLACK),
        "check_circle.svg": icon_svg("check circle", '    <circle cx="48" cy="48" r="33"/>\n    <path d="M31 49 L43 61 L66 35"/>', ORANGE),
        "anchor.svg": icon_svg("anchor", '    <circle cx="48" cy="16" r="8"/>\n    <path d="M48 24 V76"/>\n    <path d="M29 40 H67"/>\n    <path d="M16 56 C22 78 38 82 48 82 C58 82 74 78 80 56"/>\n    <path d="M16 56 L25 62"/>\n    <path d="M80 56 L71 62"/>', ORANGE),
        "network.svg": icon_svg("network", '    <path d="M48 12 L79 30 L79 66 L48 84 L17 66 L17 30 Z"/>\n    <path d="M17 30 L79 66"/>\n    <path d="M79 30 L17 66"/>\n    <circle cx="48" cy="12" r="6" fill="white"/>\n    <circle cx="79" cy="30" r="6" fill="white"/>\n    <circle cx="79" cy="66" r="6" fill="white"/>\n    <circle cx="48" cy="84" r="6" fill="white"/>\n    <circle cx="17" cy="66" r="6" fill="white"/>\n    <circle cx="17" cy="30" r="6" fill="white"/>', GREEN),
        "image.svg": icon_svg("image", '    <rect x="17" y="23" width="62" height="50" rx="5"/>\n    <circle cx="63" cy="36" r="5" fill="' + BLUE + '" stroke="none"/>\n    <path d="M22 68 L39 48 L51 61 L60 51 L75 68 Z" fill="' + BLUE + '" stroke="none"/>', BLUE),
        "question.svg": icon_svg("question", '    <circle cx="48" cy="48" r="30"/>\n    <path d="M39 40 C40 31 57 30 59 42 C60 50 51 52 49 59"/>\n    <path d="M47 72 H48"/>', BLUE),
        "refresh.svg": icon_svg("refresh", '    <path d="M72 32 A28 28 0 0 0 25 29"/>\n    <path d="M25 29 L25 13 L10 28"/>\n    <path d="M24 64 A28 28 0 0 0 71 67"/>\n    <path d="M71 67 L71 83 L86 68"/>', PURPLE),
        "sparkle.svg": textwrap.dedent(
            f"""\
            <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="sparkle">
              <path d="M48 12 L58 38 L84 48 L58 58 L48 84 L38 58 L12 48 L38 38 Z" fill="{GREEN}"/>
              <path d="M76 12 L81 25 L94 30 L81 35 L76 48 L71 35 L58 30 L71 25 Z" fill="{GREEN}"/>
            </svg>
            """
        ),
        "target.svg": icon_svg("target", '    <circle cx="48" cy="48" r="30"/>\n    <circle cx="48" cy="48" r="15"/>\n    <path d="M48 8 V28"/>\n    <path d="M48 68 V88"/>\n    <path d="M8 48 H28"/>\n    <path d="M68 48 H88"/>', GREEN),
        "nodes.svg": icon_svg("nodes", '    <path d="M48 18 L22 72 H74 Z"/>\n    <circle cx="48" cy="18" r="10" fill="white"/>\n    <circle cx="22" cy="72" r="10" fill="white"/>\n    <circle cx="74" cy="72" r="10" fill="white"/>', GREEN),
    }
    for filename, svg in icons.items():
        (assets_dir / filename).write_text(svg, encoding="utf-8")


def save_outputs(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(outputs[0], format="svg")
    fig.savefig(outputs[1], format="pdf")
    fig.savefig(outputs[2], format="png", dpi=dpi)
    return outputs


def main() -> None:
    args = parse_args()
    assets_dir = args.assets_dir or (args.output_dir / "drt_method_diagram_assets")
    write_icon_assets(assets_dir)
    fig = draw_figure(args.font_dir)
    outputs = save_outputs(fig, args.output_dir, args.stem, args.dpi)
    plt.close(fig)
    print("Wrote icon assets to", assets_dir)
    for path in outputs:
        print("Wrote", path)


if __name__ == "__main__":
    main()
