#!/usr/bin/env python3
"""Draw v3 MathVerse qualitative showcase with DRT-RL as the dominant panel."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_IMAGE = REPO_ROOT / "example" / "mathverse_3527_surface_area_inferred_x_image.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets"
ICON_DIR = REPO_ROOT / "mathverse_asset_pack" / "assets" / "icons"
SEMANTIC_BLOCK_DIR = REPO_ROOT / "mathverse_asset_pack" / "assets" / "semantic_blocks"
CANVAS_SIZE = (2048, 1545)

BLACK = "#111111"
GRAY = "#5C5C5C"
RED = "#C5161D"
RED_DARK = "#A30F15"
RED_LIGHT = "#FFF4F4"
RED_HILITE = "#FFD2D5"
ORANGE = "#EF7F1A"
ORANGE_LIGHT = "#FFF3E8"
ORANGE_HILITE = "#FFE0B7"
BLUE = "#1D43FF"
BLUE_DARK = "#182FC8"
BLUE_LIGHT = "#F7F9FF"
BLUE_SOFT = "#EBF0FF"
THINK_BLUE = "#1687FF"
THINK_LIGHT = "#F3FAFF"
GREEN = "#23833A"
GREEN_LIGHT = "#EFF9F1"

SHOWCASE_TITLE_SIZE = 52
SHOWCASE_TEXT_SIZE = 45
NON_QWEN_SHOWCASE_TEXT_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="MathVerse triangular-prism problem image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save generated files.",
    )
    parser.add_argument(
        "--stem",
        default="mathverse_showcase_v3",
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=DEFAULT_FONT_DIR,
        help="Directory containing ComicNeue font files.",
    )
    parser.add_argument("--title-size", type=int, default=SHOWCASE_TITLE_SIZE)
    parser.add_argument("--hide-title", action="store_true")
    parser.add_argument("--pdf-resolution", type=float, default=300.0)
    return parser.parse_args()


class Fonts:
    def __init__(self, font_dir: Path) -> None:
        self.regular_path = font_dir / "ComicNeue-Regular.ttf"
        self.bold_path = font_dir / "ComicNeue-Bold.ttf"

    def get(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        path = self.bold_path if bold and self.bold_path.exists() else self.regular_path
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x, y = xy
    width, height = text_size(draw, text, font)
    draw.text((x - width // 2, y - height // 2), text, font=font, fill=fill)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    width: int = 2,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_check_mark(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    color: str,
    width: int = 6,
    scale: int = 1,
) -> None:
    points = [
        (x, y + 16 * scale),
        (x + 12 * scale, y + 28 * scale),
        (x + 34 * scale, y),
    ]
    draw.line(points, fill=color, width=width, joint="curve")


def draw_x_mark(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    color: str,
    width: int = 6,
    size: int = 34,
) -> None:
    draw.line([(x, y), (x + size, y + size)], fill=color, width=width)
    draw.line([(x + size, y), (x, y + size)], fill=color, width=width)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y: int,
    x2: int,
    *,
    fill: str,
    width: int = 2,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x = x1
    while x < x2:
        draw.line((x, y, min(x + dash, x2), y), fill=fill, width=width)
        x += dash + gap


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_gap
    return y


def draw_wrapped_around_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    *,
    right_edge: int,
    avoid_box: tuple[int, int, int, int],
    gap: int,
    line_gap: int,
) -> int:
    x, y = xy
    avoid_x1, avoid_y1, avoid_x2, avoid_y2 = avoid_box
    words = text.split()
    current = ""

    while words:
        overlaps_avoid_box = y + font.size > avoid_y1 and y < avoid_y2
        max_width = (avoid_x1 - gap - x) if overlaps_avoid_box else (right_edge - x)
        if max_width < 120:
            y = avoid_y2 + line_gap
            current = ""
            continue

        word = words.pop(0)
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
            continue

        if current:
            draw.text((x, y), current, font=font, fill=fill)
            y += line_gap
            words.insert(0, word)
            current = ""
        else:
            draw.text((x, y), word, font=font, fill=fill)
            y += line_gap

    if current:
        draw.text((x, y), current, font=font, fill=fill)
        y += line_gap
    return y


def draw_rich_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    segments: list[tuple[str, str, str | None]],
    font: ImageFont.ImageFont,
    *,
    pad_x: int = 5,
    pad_y: int = 3,
) -> None:
    cursor = x
    for text, fill, highlight in segments:
        if highlight and text:
            width = int(draw.textlength(text, font=font))
            rounded(
                draw,
                (cursor - pad_x, y - pad_y, cursor + width + pad_x, y + font.size + pad_y),
                fill=highlight,
                outline=highlight,
                width=1,
                radius=7,
            )
        draw.text((cursor, y), text, font=font, fill=fill)
        cursor += int(draw.textlength(text, font=font))


def draw_rich_wrapped(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    segments: list[tuple[str, str, str | None]],
    font: ImageFont.ImageFont,
    *,
    max_width: int,
    line_gap: int,
    pad_x: int = 5,
    pad_y: int = 2,
) -> int:
    cursor_x = x
    cursor_y = y
    for text, fill, highlight in segments:
        parts = text.split(" ")
        tokens = [
            part if index == len(parts) - 1 else f"{part} "
            for index, part in enumerate(parts)
            if part or index != len(parts) - 1
        ]
        for token in tokens:
            token_width = int(draw.textlength(token, font=font))
            if cursor_x > x and cursor_x + token_width > x + max_width:
                cursor_x = x
                cursor_y += line_gap
            if highlight and token.strip():
                rounded(
                    draw,
                    (
                        cursor_x - pad_x,
                        cursor_y - pad_y,
                        cursor_x + token_width + pad_x,
                        cursor_y + font.size + pad_y,
                    ),
                    fill=highlight,
                    outline=highlight,
                    width=1,
                    radius=7,
                )
            draw.text((cursor_x, cursor_y), token, font=font, fill=fill)
            cursor_x += token_width
    return cursor_y + line_gap


def resize_contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    max_w, max_h = box
    scale = min(max_w / image.width, max_h / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(size, resampling)


def paste_v1_icon(
    canvas: Image.Image,
    icon_name: str,
    center: tuple[int, int],
    *,
    height: int,
) -> None:
    icon_sources = {
        "drt_eye_icon.png": ("drt_observe_box.png", (718, 5, 778, 57)),
        "drt_brain_icon.png": ("drt_think_box.png", (724, 8, 776, 68)),
        "drt_green_check_icon.png": ("drt_answer_box.png", (722, 18, 774, 70)),
    }
    source_name, crop_box = icon_sources[icon_name]
    icon_path = SEMANTIC_BLOCK_DIR / source_name
    if not icon_path.exists():
        return
    icon = Image.open(icon_path).convert("RGBA").crop(crop_box)
    pixels = icon.load()
    for y in range(icon.height):
        for x in range(icon.width):
            red, green, blue, alpha = pixels[x, y]
            if red > 242 and green > 242 and blue > 242:
                pixels[x, y] = (red, green, blue, 0)
            elif alpha < 255:
                pixels[x, y] = (red, green, blue, min(255, alpha + 35))
    scale = height / icon.height
    size = (round(icon.width * scale), height)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    icon = icon.resize(size, resampling)
    x = center[0] - icon.width // 2
    y = center[1] - icon.height // 2
    canvas.paste(icon, (x, y), icon)


def draw_example_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    image_path: Path,
) -> None:
    box = (30, 118, 930, 520)
    rounded(draw, box, fill="white", outline=BLACK, width=3, radius=24)

    problem = Image.open(image_path).convert("RGB")
    problem = resize_contain(problem, (470, 270))
    px = 930 - 28 - problem.width
    py = 520 - 8 - problem.height

    canvas.paste(problem, (px, py))

    question_font = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    draw_wrapped(
        draw,
        (62, 138),
        "Find total surface area of the triangular prism shown. The height is 16 cm.",
        question_font,
        BLACK,
        max_width=838,
        line_gap=56,
    )


def draw_long_cot_panel(draw: ImageDraw.ImageDraw, fonts: Fonts) -> None:
    box = (30, 548, 930, 1514)
    rounded(draw, box, fill="white", outline=RED, width=4, radius=24)
    rounded(draw, (30, 548, 930, 626), fill=RED, outline=RED, width=0, radius=24)
    draw_centered(draw, (480, 587), "Qwen3-VL CoT", fonts.get(SHOWCASE_TEXT_SIZE, bold=True), "white")

    prediction_font = fonts.get(SHOWCASE_TEXT_SIZE, bold=True)
    value_font = fonts.get(SHOWCASE_TEXT_SIZE, bold=True)
    token_font = fonts.get(SHOWCASE_TEXT_SIZE, bold=True)
    draw.text((110, 660), "Prediction:", font=prediction_font, fill=BLACK)
    draw.text((370, 660), "960", font=value_font, fill=RED)
    draw_x_mark(draw, 482, 668, color=RED, width=8, size=40)
    draw.text((548, 660), "|  4,937 tokens", font=token_font, fill=BLACK)

    legend_font = fonts.get(SHOWCASE_TEXT_SIZE, bold=True)
    legend_x = 62
    legend_y = 734
    rounded(draw, (legend_x, legend_y, legend_x + 50, legend_y + 50), fill=ORANGE, outline=ORANGE, width=3, radius=9)
    draw.text((legend_x + 68, legend_y - 2), "verbose filler", font=legend_font, fill=GRAY)
    second_x = 500
    rounded(draw, (second_x, legend_y, second_x + 50, legend_y + 50), fill=RED, outline=RED, width=3, radius=9)
    draw.text((second_x + 68, legend_y - 2), "guessed premise", font=legend_font, fill=GRAY)

    rounded(draw, (60, 812, 900, 1494), fill=RED_LIGHT, outline=RED, width=2, radius=16)

    font = fonts.get(SHOWCASE_TEXT_SIZE, bold=True)
    x = 78
    y = 840
    draw_rich_wrapped(
        draw,
        x,
        y,
        [
            ("Got it, let's try to figure out the total surface area... First, I need to recall what a triangular prism is... ", ORANGE, ORANGE_HILITE),
            ("First, let's list out the given information... Let me check the diagram... Let's clarify.... ", ORANGE, ORANGE_HILITE),
            ("[omitted: many repeated trials and reinterpretations] ", GRAY, "#E8E8E8"),
            ("Maybe the height of the triangular base is 16 cm... area of the base is 96, two bases = 192. ", RED_DARK, RED_HILITE),
            ("Lateral surface area = 48*16 = 768. Total = 192 + 768 = 960. ", RED_DARK, RED_HILITE),
            ("[omitted: repeated double-checks that still miss the true answer]", GRAY, "#E8E8E8"),
        ],
        font,
        max_width=812,
        line_gap=50,
    )
    draw.rounded_rectangle((60, 812, 900, 1494), radius=16, outline=RED, width=2)


def draw_summary_pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    fonts: Fonts,
) -> None:
    rounded(draw, box, fill=GREEN_LIGHT, outline=GREEN, width=2, radius=12)
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) // 2
    draw_centered(draw, ((x1 + x2) // 2, center_y), label, fonts.get(28, bold=True), GREEN)


def draw_drt_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, fonts: Fonts) -> None:
    body_font = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    body_bold = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    line_gap = 66
    box = (950, 118, 2018, 1514)
    rounded(draw, box, fill="white", outline=BLUE, width=3, radius=24)
    rounded(draw, (950, 118, 2018, 196), fill=BLUE_DARK, outline=BLUE_DARK, width=0, radius=24)
    draw_centered(draw, (1484, 157), "Dense Reasoning Trace", fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True), "white")

    prediction_font = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    value_font = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    token_font = fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True)
    draw.text((1120, 232), "Prediction:", font=prediction_font, fill=BLACK)
    draw.text((1380, 232), "608", font=value_font, fill=GREEN)
    draw_check_mark(draw, 1490, 244, color=GREEN, width=8, scale=1)
    draw.text((1560, 232), "|  137 tokens", font=token_font, fill=BLACK)

    rounded(draw, (980, 305, 1988, 545), fill=BLUE_LIGHT, outline=BLUE, width=2, radius=15)
    draw.text((1018, 334), "Visual", font=fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True), fill=BLUE)
    visual_lines = [
        "Triangular prism; base = 12 cm;",
        "equal sides = 10 cm; prism length = 16 cm.",
    ]
    y = 404
    for line in visual_lines:
        draw.text((1018, y), line, font=body_font, fill=BLACK)
        y += line_gap
    paste_v1_icon(canvas, "drt_eye_icon.png", (1945, 358), height=72)

    rounded(draw, (980, 575, 1988, 1300), fill=THINK_LIGHT, outline=THINK_BLUE, width=2, radius=15)
    draw.text((1018, 604), "Think", font=fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True), fill=THINK_BLUE)
    y = 680
    draw.text((1018, y), "[Priors] isosceles altitude bisects base;", font=body_font, fill=BLACK)
    y += line_gap
    for line in [
        "      Pythagorean theorem: a^2 + b^2 = c^2",
        "-> half-base = 6",
        "-> x = sqrt(10^2 - 6^2) = 8",
        "-> triangle area = 1/2 * 12 * 8 = 48",
        "-> two bases = 96",
        "-> lateral area = 32 * 16 = 512",
    ]:
        draw.text((1018, y), line, font=body_font, fill=BLACK)
        y += line_gap
    draw.text((1018, y), "-> total surface area = 608.", font=body_font, fill=BLACK)
    paste_v1_icon(canvas, "drt_brain_icon.png", (1940, 1236), height=84)

    rounded(draw, (980, 1330, 1988, 1494), fill=GREEN_LIGHT, outline=GREEN, width=2, radius=15)
    draw.text((1018, 1362), "Answer", font=fonts.get(NON_QWEN_SHOWCASE_TEXT_SIZE, bold=True), fill=GREEN)
    draw.text((1018, 1422), "608", font=body_bold, fill=BLACK)
    paste_v1_icon(canvas, "drt_green_check_icon.png", (1940, 1418), height=80)


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


def draw_showcase(
    image_path: Path,
    font_dir: Path,
    title_size: int = SHOWCASE_TITLE_SIZE,
    show_title: bool = True,
) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    fonts = Fonts(font_dir)

    if show_title:
        draw_centered(
            draw,
            (1024, 55),
            "Showcasing Long-CoT and DRT Reasoning Paradigms",
            fonts.get(title_size, bold=True),
            BLACK,
        )
    draw_example_panel(canvas, draw, fonts, image_path)
    draw_drt_panel(canvas, draw, fonts)
    draw_long_cot_panel(draw, fonts)
    return canvas


def save_outputs(image: Image.Image, output_base: Path, pdf_resolution: float) -> list[Path]:
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=pdf_resolution)
    write_embedded_svg(png_path, svg_path, image.width, image.height)
    return [png_path, pdf_path, svg_path]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = draw_showcase(
        args.image,
        args.font_dir,
        title_size=args.title_size,
        show_title=not args.hide_title,
    )
    for path in save_outputs(image, args.output_dir / args.stem, args.pdf_resolution):
        print(path)


if __name__ == "__main__":
    main()
