#!/usr/bin/env python3
"""Generate the v3 qualitative showcase and tradeoff plot in one combined pass."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

from plot_drt_token_efficiency_v3 import build_figure
from plot_drt_token_efficiency_v3 import build_points
from plot_drt_token_efficiency_v3 import DEFAULT_FONT_DIR as TRADEOFF_FONT_DIR
from plot_drt_token_efficiency_v3 import plt
from plot_mathverse_showcase_v3 import BLACK
from plot_mathverse_showcase_v3 import DEFAULT_FONT_DIR as SHOWCASE_FONT_DIR
from plot_mathverse_showcase_v3 import Fonts
from plot_mathverse_showcase_v3 import draw_centered
from plot_mathverse_showcase_v3 import draw_showcase


TEASER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = TEASER_ROOT / "example" / "mathverse_3527_surface_area_inferred_x_image.png"
DEFAULT_MAIN_RESULT_TEX = TEASER_ROOT / "main_result.tex"
DEFAULT_OUTPUT_DIR = TEASER_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--main-result-tex", type=Path, default=DEFAULT_MAIN_RESULT_TEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="showcase_tradeoff_combined_v3_integrated")
    parser.add_argument("--font-dir", type=Path, default=SHOWCASE_FONT_DIR)
    parser.add_argument("--tradeoff-font-dir", type=Path, default=TRADEOFF_FONT_DIR)
    parser.add_argument("--title-size", type=int, default=58)
    parser.add_argument("--tradeoff-fig-width", type=float, default=8.6)
    parser.add_argument("--tradeoff-fig-height", type=float, default=7.4)
    parser.add_argument("--right-width", type=int, default=1712)
    parser.add_argument("--gutter", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf-resolution", type=float, default=300.0)
    return parser.parse_args()


def resize_exact(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(size, resampling)


def render_tradeoff(args: argparse.Namespace) -> Image.Image:
    points = build_points(args.main_result_tex)
    fig = build_figure(
        points,
        args.tradeoff_font_dir,
        figsize=(args.tradeoff_fig_width, args.tradeoff_fig_height),
        title_size=1,
        title_text=None,
    )
    fig.subplots_adjust(left=0.115, right=0.955, bottom=0.115, top=0.925)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def draw_vertical_separator(canvas: Image.Image, x: int) -> None:
    draw = ImageDraw.Draw(canvas)
    dash = 24
    gap = 18
    y = 24
    while y < canvas.height - 24:
        draw.line(
            [(x, y), (x, min(y + dash, canvas.height - 24))],
            fill="#9A9A9A",
            width=4,
        )
        y += dash + gap


def write_embedded_svg(png_path: Path, svg_path: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="utf-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                ),
                f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_outputs(canvas: Image.Image, output_base: Path, pdf_resolution: float) -> list[Path]:
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=pdf_resolution)
    write_embedded_svg(png_path, svg_path, canvas.width, canvas.height)
    return [png_path, pdf_path, svg_path]


def compose(args: argparse.Namespace) -> Image.Image:
    showcase = draw_showcase(
        args.image,
        args.font_dir,
        title_size=args.title_size,
        show_title=False,
    )
    tradeoff = render_tradeoff(args)
    tradeoff = resize_exact(tradeoff, (args.right_width, showcase.height))

    canvas = Image.new(
        "RGB",
        (showcase.width + args.gutter + tradeoff.width, showcase.height),
        "white",
    )
    canvas.paste(showcase, (0, 0))
    canvas.paste(tradeoff, (showcase.width + args.gutter, 0))
    draw_vertical_separator(canvas, showcase.width + args.gutter // 2)

    draw = ImageDraw.Draw(canvas)
    title_font = Fonts(args.font_dir).get(args.title_size, bold=True)
    draw_centered(
        draw,
        (showcase.width // 2, 55),
        "(A) Showcasing CoT and DRT Reasoning Paradigms",
        title_font,
        BLACK,
    )
    draw_centered(
        draw,
        (showcase.width + args.gutter + tradeoff.width // 2, 55),
        "(B) Accuracy-Token Tradeoff Across Reasoning Methods",
        title_font,
        BLACK,
    )
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canvas = compose(args)
    for path in save_outputs(canvas, args.output_dir / args.stem, args.pdf_resolution):
        print(path)


if __name__ == "__main__":
    main()
