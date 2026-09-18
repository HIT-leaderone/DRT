#!/usr/bin/env python3
"""Compare multimodal prompt lengths for VisionR1+DAPO and mmfinereason parquet files."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from qwen_vl_utils.vision_process import IMAGE_MAX_TOKEN_NUM, IMAGE_MIN_TOKEN_NUM, MAX_RATIO, smart_resize


DEFAULT_ORIGINAL = "data/VisionR1_DAPO_ref/train.parquet"
DEFAULT_MMFINEREASON = "data/VisionR1_DAPO_mmfinereason/mmfinereason_train.parquet"
DEFAULT_MODEL_CANDIDATES = (
    "/tmp/qwen35_tok",
    "/tmp/qwen35_27b_tokenizer",
    os.environ.get("HF_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
)


@dataclass
class SampleStats:
    dataset: str
    row: int
    data_source: str
    ability: str
    image_count: int
    width_max: int
    height_max: int
    pixel_max: int
    aspect_max: float
    resized_pixel_max: int
    vision_tokens: int
    text_tokens: int
    exact_prompt_tokens: int
    estimated_prompt_tokens: int
    over_prompt_limit: bool
    over_vision_threshold: bool
    bad: bool
    error: str


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def summarise(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def first_existing_model_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def load_tokenizer_processor(
    model_path: str | None,
    *,
    disable_tokenizer: bool = False,
    disable_processor: bool = False,
) -> tuple[Any | None, Any | None]:
    if not model_path:
        return None, None
    try:
        from transformers import AutoProcessor, AutoTokenizer

        tokenizer = None
        processor = None
        if not disable_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
        if not disable_processor:
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
            except Exception:
                processor = None
        return tokenizer, processor
    except Exception:
        print(f"[warn] failed to load tokenizer/processor from {model_path}", file=sys.stderr)
        traceback.print_exc()
        return None, None


def normalise_prompt(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return copy.deepcopy(list(prompt or []))


def build_messages(prompt: Any, has_multimodal: bool) -> list[dict[str, Any]]:
    messages = normalise_prompt(prompt)
    if not has_multimodal:
        return messages
    for message in messages:
        content = message.get("content", "")
        content_list = []
        segments = [item for item in re.split("(<image>|<video>)", content) if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list
    return messages


def image_bytes(image: Any) -> bytes | None:
    if hasattr(image, "as_py"):
        image = image.as_py()
    if not isinstance(image, dict):
        return None
    raw = image.get("bytes")
    if raw is not None:
        return raw
    nested = image.get("image")
    if isinstance(nested, dict):
        return nested.get("bytes")
    return None


def image_dimensions(image: Any) -> tuple[int, int]:
    raw = image_bytes(image)
    if raw is None:
        raise ValueError("image dict has no bytes field")
    with Image.open(BytesIO(raw)) as img:
        return img.size


def pil_image(image: Any) -> Image.Image:
    raw = image_bytes(image)
    if raw is None:
        raise ValueError("image dict has no bytes field")
    with Image.open(BytesIO(raw)) as img:
        return img.convert("RGB")


def estimate_vision_tokens_for_image(
    image: Any,
    *,
    image_patch_size: int,
    merge_size: int,
) -> tuple[int, int, int, int, float]:
    width, height = image_dimensions(image)
    factor = image_patch_size * merge_size
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width):.2f} > {MAX_RATIO}")
    if isinstance(image, dict):
        min_pixels = image.get("min_pixels", IMAGE_MIN_TOKEN_NUM * factor * factor)
        max_pixels = image.get("max_pixels", IMAGE_MAX_TOKEN_NUM * factor * factor)
        target_h = image.get("resized_height", height)
        target_w = image.get("resized_width", width)
    else:
        min_pixels = IMAGE_MIN_TOKEN_NUM * factor * factor
        max_pixels = IMAGE_MAX_TOKEN_NUM * factor * factor
        target_h = height
        target_w = width
    resized_h, resized_w = smart_resize(target_h, target_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    tokens = (resized_h // factor) * (resized_w // factor)
    return tokens, width, height, resized_w, float(resized_h * resized_w)


def safe_image_list(row: dict[str, Any], image_key: str) -> list[Any]:
    images = row.get(image_key) or []
    if hasattr(images, "tolist"):
        images = images.tolist()
    return list(images)


def count_image_markers(prompt: Any) -> int:
    return sum(str(message.get("content", "")).count("<image>") for message in normalise_prompt(prompt))


def analyse_row(
    dataset_name: str,
    row_idx: int,
    row: dict[str, Any],
    *,
    tokenizer: Any | None,
    processor: Any | None,
    max_prompt_length: int,
    vision_token_threshold: int,
    image_patch_size: int,
    merge_size: int,
    prompt_key: str,
    image_key: str,
) -> SampleStats:
    images = safe_image_list(row, image_key)
    data_source = str(row.get("data_source", ""))
    ability = str(row.get("ability", ""))
    text_tokens = -1
    exact_prompt_tokens = -1
    estimated_prompt_tokens = -1
    vision_tokens = 0
    widths: list[int] = []
    heights: list[int] = []
    pixels: list[int] = []
    aspects: list[float] = []
    resized_pixels: list[int] = []
    error = ""
    bad = False

    try:
        for image in images:
            tokens, width, height, _resized_w, resized_pixel = estimate_vision_tokens_for_image(
                image, image_patch_size=image_patch_size, merge_size=merge_size
            )
            vision_tokens += tokens
            widths.append(width)
            heights.append(height)
            pixels.append(width * height)
            aspects.append(max(width, height) / max(1, min(width, height)))
            resized_pixels.append(int(resized_pixel))

        messages = build_messages(row.get(prompt_key), bool(images))
        raw_prompt = None
        if processor is not None:
            raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            proc_images = [pil_image(image) for image in images] or None
            proc_out = processor(text=[raw_prompt], images=proc_images, return_tensors=None)
            exact_prompt_tokens = len(proc_out["input_ids"][0])
            image_grid_thw = proc_out.get("image_grid_thw")
            if image_grid_thw is not None:
                actual_merge = getattr(getattr(processor, "image_processor", None), "merge_size", merge_size)
                grid_tokens = 0
                for grid in image_grid_thw:
                    t, h, w = [int(x) for x in grid]
                    grid_tokens += t * (h // actual_merge) * (w // actual_merge)
                vision_tokens = grid_tokens
        if tokenizer is not None:
            if raw_prompt is None:
                raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            text_tokens = len(tokenizer.encode(raw_prompt, add_special_tokens=False))
            estimated_prompt_tokens = text_tokens - count_image_markers(row.get(prompt_key)) + vision_tokens
        elif exact_prompt_tokens >= 0:
            estimated_prompt_tokens = exact_prompt_tokens
    except Exception as exc:
        bad = True
        error = f"{type(exc).__name__}: {exc}"

    prompt_len_for_limit = exact_prompt_tokens if exact_prompt_tokens >= 0 else estimated_prompt_tokens
    return SampleStats(
        dataset=dataset_name,
        row=row_idx,
        data_source=data_source,
        ability=ability,
        image_count=len(images),
        width_max=max(widths) if widths else 0,
        height_max=max(heights) if heights else 0,
        pixel_max=max(pixels) if pixels else 0,
        aspect_max=max(aspects) if aspects else 0.0,
        resized_pixel_max=max(resized_pixels) if resized_pixels else 0,
        vision_tokens=vision_tokens,
        text_tokens=text_tokens,
        exact_prompt_tokens=exact_prompt_tokens,
        estimated_prompt_tokens=estimated_prompt_tokens,
        over_prompt_limit=prompt_len_for_limit > max_prompt_length if prompt_len_for_limit >= 0 else False,
        over_vision_threshold=vision_tokens > vision_token_threshold,
        bad=bad,
        error=error,
    )


def iter_parquet_rows(path: str, columns: list[str], batch_size: int):
    parquet = pq.ParquetFile(path)
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield row_offset, row
            row_offset += 1


def analyse_dataset(
    dataset_name: str,
    path: str,
    *,
    tokenizer: Any | None,
    processor: Any | None,
    args: argparse.Namespace,
) -> list[SampleStats]:
    columns = [args.prompt_key, args.image_key, "data_source", "ability"]
    results = []
    for row_idx, row in iter_parquet_rows(path, columns=columns, batch_size=args.batch_size):
        if args.limit >= 0 and row_idx >= args.limit:
            break
        if row_idx and row_idx % args.progress_every == 0:
            print(f"[progress] {dataset_name}: {row_idx} rows", file=sys.stderr)
        results.append(
            analyse_row(
                dataset_name,
                row_idx,
                row,
                tokenizer=tokenizer,
                processor=processor,
                max_prompt_length=args.max_prompt_length,
                vision_token_threshold=args.vision_token_threshold,
                image_patch_size=args.image_patch_size,
                merge_size=args.merge_size,
                prompt_key=args.prompt_key,
                image_key=args.image_key,
            )
        )
    return results


def dataset_summary(rows: list[SampleStats]) -> dict[str, Any]:
    prompt_values = [
        row.exact_prompt_tokens if row.exact_prompt_tokens >= 0 else row.estimated_prompt_tokens
        for row in rows
        if (row.exact_prompt_tokens >= 0 or row.estimated_prompt_tokens >= 0)
    ]
    return {
        "rows": len(rows),
        "bad_rows": sum(row.bad for row in rows),
        "rows_with_images": sum(row.image_count > 0 for row in rows),
        "over_prompt_limit": sum(row.over_prompt_limit for row in rows),
        "over_vision_threshold": sum(row.over_vision_threshold for row in rows),
        "image_count": summarise([row.image_count for row in rows]),
        "vision_tokens": summarise([row.vision_tokens for row in rows]),
        "prompt_tokens": summarise(prompt_values),
        "text_tokens": summarise([row.text_tokens for row in rows if row.text_tokens >= 0]),
        "pixel_max": summarise([row.pixel_max for row in rows]),
        "aspect_max": summarise([row.aspect_max for row in rows]),
        "by_source": by_field_summary(rows, "data_source"),
    }


def by_field_summary(rows: list[SampleStats], field: str) -> dict[str, Any]:
    groups: dict[str, list[SampleStats]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, field)), []).append(row)
    return {
        key: {
            "rows": len(group),
            "rows_with_images": sum(row.image_count > 0 for row in group),
            "over_prompt_limit": sum(row.over_prompt_limit for row in group),
            "vision_tokens": summarise([row.vision_tokens for row in group]),
        }
        for key, group in sorted(groups.items())
    }


def write_top_csv(path: Path, rows: list[SampleStats], top_k: int) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            row.bad,
            row.over_prompt_limit,
            row.vision_tokens,
            row.exact_prompt_tokens if row.exact_prompt_tokens >= 0 else row.estimated_prompt_tokens,
        ),
        reverse=True,
    )[:top_k]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(ranked[0]).keys()) if ranked else list(SampleStats.__annotations__))
        writer.writeheader()
        for row in ranked:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--mmfinereason", default=DEFAULT_MMFINEREASON)
    parser.add_argument("--model-path", default=None, help="Optional local Qwen3-VL tokenizer/processor path.")
    parser.add_argument("--no-model", action="store_true", help="Skip tokenizer and processor loading.")
    parser.add_argument("--disable-processor", action="store_true", help="Use tokenizer text lengths plus estimated vision tokens.")
    parser.add_argument("--disable-tokenizer", action="store_true", help="Skip text tokenization.")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--vision-token-threshold", type=int, default=2048)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=-1, help="Rows per dataset; -1 means full scan.")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/vision_prompt_overlong")
    args = parser.parse_args()

    model_path = None if args.no_model else first_existing_model_path(args.model_path)
    tokenizer, processor = load_tokenizer_processor(
        model_path,
        disable_tokenizer=args.disable_tokenizer,
        disable_processor=args.disable_processor,
    )
    if tokenizer is None and processor is None:
        print("[warn] tokenizer/processor unavailable; prompt token counts will be image-only/text-unavailable estimates.", file=sys.stderr)
    elif processor is None:
        print("[warn] processor unavailable; exact multimodal prompt tokens will be unavailable.", file=sys.stderr)

    all_rows: list[SampleStats] = []
    per_dataset: dict[str, list[SampleStats]] = {}
    for dataset_name, path in (("original", args.original), ("mmfinereason", args.mmfinereason)):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows = analyse_dataset(dataset_name, path, tokenizer=tokenizer, processor=processor, args=args)
        per_dataset[dataset_name] = rows
        all_rows.extend(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "original": args.original,
            "mmfinereason": args.mmfinereason,
            "model_path": model_path,
            "loaded_tokenizer": tokenizer is not None,
            "loaded_processor": processor is not None,
            "max_prompt_length": args.max_prompt_length,
            "vision_token_threshold": args.vision_token_threshold,
            "image_patch_size": args.image_patch_size,
            "merge_size": args.merge_size,
            "limit": args.limit,
        },
        "datasets": {name: dataset_summary(rows) for name, rows in per_dataset.items()},
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_top_csv(output_dir / "top_overlong_or_vision_heavy.csv", all_rows, args.top_k)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[wrote] {output_dir / 'summary.json'}")
    print(f"[wrote] {output_dir / 'top_overlong_or_vision_heavy.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
