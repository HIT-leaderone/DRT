#!/usr/bin/env python3
"""Audit multimodal parquet samples for image/video and prompt-shape outliers."""

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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from qwen_vl_utils.vision_process import IMAGE_MAX_TOKEN_NUM, IMAGE_MIN_TOKEN_NUM, MAX_RATIO, smart_resize


DEFAULT_DATASETS = (
    "original=data/VisionR1_DAPO_ref/train.parquet",
    "mmfinereason=data/VisionR1_DAPO_mmfinereason/mmfinereason_train.parquet",
)
DEFAULT_MODEL_CANDIDATES = (
    "/tmp/qwen35_tok",
    os.environ.get("HF_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
)


@dataclass
class RowAudit:
    dataset: str
    row: int
    data_source: str = ""
    ability: str = ""
    extra_index: str = ""
    extra_num_images: int = -1
    image_count: int = 0
    image_marker_count: int = 0
    video_count: int = 0
    video_marker_count: int = 0
    video_frame_count: int = 0
    width_max: int = 0
    height_max: int = 0
    raw_pixel_max: int = 0
    raw_pixel_sum: int = 0
    aspect_max: float = 0.0
    resized_height_max: int = 0
    resized_width_max: int = 0
    resized_pixel_max: int = 0
    image_grid_h_max: int = 0
    image_grid_w_max: int = 0
    image_llm_tokens_max: int = 0
    image_llm_tokens_sum: int = 0
    text_tokens: int = -1
    estimated_prompt_tokens: int = -1
    exact_prompt_tokens: int = -1
    exact_image_token_count: int = -1
    exact_image_grid_tokens: int = -1
    exact_image_grid_rows: int = -1
    exact_position_shape: str = ""
    exact_position_max: int = -1
    exact_ok: bool = False
    issue_tags: list[str] = field(default_factory=list)
    exact_error: str = ""
    error: str = ""


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


def normalise_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def normalise_prompt(prompt: Any) -> list[dict[str, Any]]:
    return copy.deepcopy(normalise_list(prompt))


def prompt_text(prompt: Any) -> str:
    return "\n".join(str(message.get("content", "")) for message in normalise_prompt(prompt))


def build_messages(prompt: Any) -> list[dict[str, Any]]:
    messages = normalise_prompt(prompt)
    for message in messages:
        content = str(message.get("content", ""))
        pieces = [item for item in re.split("(<image>|<video>)", content) if item != ""]
        content_list: list[dict[str, str]] = []
        for piece in pieces:
            if piece == "<image>":
                content_list.append({"type": "image"})
            elif piece == "<video>":
                content_list.append({"type": "video"})
            else:
                content_list.append({"type": "text", "text": piece})
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


def image_size(image: Any) -> tuple[int, int]:
    raw = image_bytes(image)
    if raw is None:
        raise ValueError("image dict has no bytes field")
    with Image.open(BytesIO(raw)) as img:
        return img.size


def image_token_stats(
    image: Any,
    *,
    image_patch_size: int,
    merge_size: int,
) -> dict[str, int | float]:
    width, height = image_size(image)
    aspect = max(width, height) / max(1, min(width, height))
    factor = image_patch_size * merge_size
    if aspect > MAX_RATIO:
        raise ValueError(f"aspect ratio {aspect:.2f} > {MAX_RATIO}")
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
    grid_h = resized_h // image_patch_size
    grid_w = resized_w // image_patch_size
    llm_grid_h = grid_h // merge_size
    llm_grid_w = grid_w // merge_size
    return {
        "width": width,
        "height": height,
        "raw_pixels": width * height,
        "aspect": aspect,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "resized_pixels": resized_h * resized_w,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "llm_tokens": llm_grid_h * llm_grid_w,
        "resized_mod_factor": (resized_h % factor) + (resized_w % factor),
        "grid_mod_merge": (grid_h % merge_size) + (grid_w % merge_size),
        "max_pixels": max_pixels,
    }


def count_video_frames(video: Any) -> int:
    if hasattr(video, "as_py"):
        video = video.as_py()
    if not isinstance(video, dict):
        return 0
    payload = video.get("video")
    if isinstance(payload, list):
        return len(payload)
    if isinstance(video.get("nframes"), int):
        return int(video["nframes"])
    if isinstance(video.get("max_frames"), int):
        return int(video["max_frames"])
    return 0


def row_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    extra = row.get("extra_info") or {}
    if not isinstance(extra, dict):
        return "", -1, ""
    return str(extra.get("index", "")), int(extra.get("num_images", -1) or -1), str(extra.get("source_index", ""))


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)


def estimate_text_tokens(tokenizer: Any | None, prompt: Any) -> int:
    if tokenizer is None:
        return -1
    messages = build_messages(prompt)
    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tokenizer.encode(raw_prompt, add_special_tokens=False))


def audit_row(
    dataset_name: str,
    row_idx: int,
    row: dict[str, Any],
    *,
    tokenizer: Any | None,
    args: argparse.Namespace,
) -> RowAudit:
    audit = RowAudit(dataset=dataset_name, row=row_idx)
    audit.data_source = str(row.get("data_source", ""))
    audit.ability = str(row.get("ability", ""))
    audit.extra_index, audit.extra_num_images, _source_index = row_identity(row)
    images = normalise_list(row.get(args.image_key))
    videos = normalise_list(row.get(args.video_key))
    text = prompt_text(row.get(args.prompt_key))
    audit.image_count = len(images)
    audit.video_count = len(videos)
    audit.image_marker_count = text.count("<image>")
    audit.video_marker_count = text.count("<video>")
    audit.video_frame_count = sum(count_video_frames(video) for video in videos)

    issue_tags: set[str] = set()
    if audit.image_count > 1:
        issue_tags.add("multi_image")
    if audit.image_count != audit.image_marker_count:
        issue_tags.add("image_marker_mismatch")
    if audit.extra_num_images >= 0 and audit.extra_num_images != audit.image_count:
        issue_tags.add("extra_num_images_mismatch")
    if audit.video_count > 0 or audit.video_marker_count > 0:
        issue_tags.add("has_video")
    if audit.video_count != audit.video_marker_count:
        issue_tags.add("video_marker_mismatch")
    if audit.video_frame_count > args.video_frame_threshold:
        issue_tags.add("many_video_frames")

    per_image_tokens: list[int] = []
    try:
        for image in images:
            stats = image_token_stats(image, image_patch_size=args.image_patch_size, merge_size=args.merge_size)
            width = int(stats["width"])
            height = int(stats["height"])
            raw_pixels = int(stats["raw_pixels"])
            resized_pixels = int(stats["resized_pixels"])
            llm_tokens = int(stats["llm_tokens"])
            audit.width_max = max(audit.width_max, width)
            audit.height_max = max(audit.height_max, height)
            audit.raw_pixel_max = max(audit.raw_pixel_max, raw_pixels)
            audit.raw_pixel_sum += raw_pixels
            audit.aspect_max = max(audit.aspect_max, float(stats["aspect"]))
            audit.resized_height_max = max(audit.resized_height_max, int(stats["resized_h"]))
            audit.resized_width_max = max(audit.resized_width_max, int(stats["resized_w"]))
            audit.resized_pixel_max = max(audit.resized_pixel_max, resized_pixels)
            audit.image_grid_h_max = max(audit.image_grid_h_max, int(stats["grid_h"]))
            audit.image_grid_w_max = max(audit.image_grid_w_max, int(stats["grid_w"]))
            audit.image_llm_tokens_max = max(audit.image_llm_tokens_max, llm_tokens)
            audit.image_llm_tokens_sum += llm_tokens
            per_image_tokens.append(llm_tokens)
            if raw_pixels > int(stats["max_pixels"]):
                issue_tags.add("raw_pixels_gt_processor_max")
            if resized_pixels > int(stats["max_pixels"]):
                issue_tags.add("resized_pixels_gt_processor_max")
            if int(stats["resized_mod_factor"]) != 0:
                issue_tags.add("resize_not_factor_aligned")
            if int(stats["grid_mod_merge"]) != 0:
                issue_tags.add("grid_not_merge_aligned")
            if llm_tokens <= 0:
                issue_tags.add("zero_vision_tokens")
            if llm_tokens > args.per_image_vision_token_threshold:
                issue_tags.add("per_image_vision_tokens_high")
    except Exception as exc:
        audit.error = f"{type(exc).__name__}: {exc}"
        issue_tags.add("image_processing_error")

    try:
        audit.text_tokens = estimate_text_tokens(tokenizer, row.get(args.prompt_key))
        if audit.text_tokens >= 0:
            audit.estimated_prompt_tokens = (
                audit.text_tokens - audit.image_marker_count - audit.video_marker_count + audit.image_llm_tokens_sum
            )
    except Exception as exc:
        audit.error = f"{audit.error}; {type(exc).__name__}: {exc}".strip("; ")
        issue_tags.add("text_tokenization_error")

    if audit.raw_pixel_max > args.raw_pixel_threshold:
        issue_tags.add("raw_pixels_high")
    if audit.width_max > args.raw_edge_threshold or audit.height_max > args.raw_edge_threshold:
        issue_tags.add("raw_edge_high")
    if audit.aspect_max > args.aspect_threshold:
        issue_tags.add("aspect_high")
    if audit.image_llm_tokens_sum > args.vision_token_threshold:
        issue_tags.add("vision_tokens_high")
    if audit.image_llm_tokens_sum > IMAGE_MAX_TOKEN_NUM:
        issue_tags.add("vision_tokens_gt_processor_max")
    if audit.estimated_prompt_tokens > args.max_prompt_length:
        issue_tags.add("prompt_gt_max_prompt_length")
    if audit.estimated_prompt_tokens + args.max_response_length > args.max_total_length:
        issue_tags.add("prompt_plus_response_gt_max_total")

    audit.issue_tags = sorted(issue_tags)
    return audit


def iter_rows(path: str, columns: list[str], batch_size: int):
    pf = pq.ParquetFile(path)
    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield row_offset, row
            row_offset += 1


def load_processor(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)


def postprocess_prompt_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = input_ids.shape[-1]
    if sequence_length > max_length:
        raise NotImplementedError(f"sequence_length={sequence_length} is larger than max_length={max_length}")
    if sequence_length == max_length:
        return input_ids, attention_mask
    pad_len = max_length - sequence_length
    padded_ids = torch.full((input_ids.shape[0], max_length), pad_token_id, dtype=input_ids.dtype)
    padded_mask = torch.zeros((attention_mask.shape[0], max_length), dtype=attention_mask.dtype)
    padded_ids[:, pad_len:] = input_ids
    padded_mask[:, pad_len:] = attention_mask
    return padded_ids, padded_mask


def qwen3_vl_rope_index(
    processor: Any,
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t.item()
            llm_grid_h = h.item() // spatial_merge_size
            llm_grid_w = w.item() // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)
    return position_ids


def exact_check(audit: RowAudit, row: dict[str, Any], processor: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    try:
        from qwen_vl_utils import fetch_image, fetch_video

        def process_image_like_training(image: dict[str, Any]) -> Image.Image:
            image = dict(image)
            if "bytes" in image:
                image["image"] = Image.open(BytesIO(image["bytes"]))
            return fetch_image(image, image_patch_size=args.image_patch_size)

        def process_video_like_training(video: dict[str, Any]):
            return fetch_video(
                dict(video),
                image_patch_size=args.image_patch_size,
                return_video_metadata=True,
            )

        messages = build_messages(row.get(args.prompt_key))
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = normalise_list(copy.deepcopy(row.get(args.image_key)))
        videos = normalise_list(copy.deepcopy(row.get(args.video_key)))
        proc_images = [process_image_like_training(image) for image in images] or None
        videos_kwargs = {}
        proc_videos = None
        if videos:
            proc_videos, video_metadata = zip(
                *[process_video_like_training(video) for video in videos],
                strict=True,
            )
            proc_videos = list(proc_videos)
            videos_kwargs = {"video_metadata": list(video_metadata), "do_sample_frames": False}

        model_inputs = processor(
            text=[raw_prompt],
            images=proc_images,
            videos=proc_videos,
            videos_kwargs=videos_kwargs,
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        audit.exact_prompt_tokens = int(input_ids.shape[-1])
        audit.exact_image_grid_rows = int(model_inputs["image_grid_thw"].shape[0]) if "image_grid_thw" in model_inputs else 0
        merge_size = int(getattr(processor.image_processor, "merge_size", args.merge_size))
        audit.exact_image_grid_tokens = 0
        if "image_grid_thw" in model_inputs:
            grid_tokens = 0
            for t, h, w in model_inputs["image_grid_thw"]:
                grid_tokens += int(t) * (int(h) // merge_size) * (int(w) // merge_size)
            audit.exact_image_grid_tokens = grid_tokens
        audit.exact_image_token_count = int((input_ids == processor.image_token_id).sum().item())

        input_ids, attention_mask = postprocess_prompt_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        vision_position_ids = qwen3_vl_rope_index(
            processor,
            input_ids=input_ids[0],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=attention_mask[0],
        )
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        audit.exact_position_shape = "x".join(str(dim) for dim in position_ids.shape)
        audit.exact_position_max = int(position_ids.max().item())
        audit.exact_ok = (
            tuple(position_ids.shape) == (4, args.max_prompt_length)
            and audit.exact_image_token_count == audit.exact_image_grid_tokens
            and audit.exact_image_grid_rows == audit.image_count
        )
        if not audit.exact_ok:
            tags = set(audit.issue_tags)
            tags.add("exact_shape_or_token_mismatch")
            audit.issue_tags = sorted(tags)
    except Exception as exc:
        audit.exact_error = f"{type(exc).__name__}: {exc}"
        tags = set(audit.issue_tags)
        tags.add("exact_processor_or_rope_error")
        audit.issue_tags = sorted(tags)


def parse_dataset_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = spec
        return Path(path).stem, path
    name, path = spec.split("=", 1)
    return name, path


def pick_exact_candidates(audits: list[RowAudit], args: argparse.Namespace) -> list[RowAudit]:
    tagged = [audit for audit in audits if audit.issue_tags]
    top_prompt = sorted(audits, key=lambda item: item.estimated_prompt_tokens, reverse=True)[: args.exact_top_k]
    top_vision = sorted(audits, key=lambda item: item.image_llm_tokens_sum, reverse=True)[: args.exact_top_k]
    nonissue = [audit for audit in audits if not audit.issue_tags]
    top_nonissue_prompt = sorted(nonissue, key=lambda item: item.estimated_prompt_tokens, reverse=True)[
        : args.exact_top_k
    ]
    top_nonissue_vision = sorted(nonissue, key=lambda item: item.image_llm_tokens_sum, reverse=True)[
        : args.exact_top_k
    ]
    selected: dict[tuple[str, int], RowAudit] = {}
    for audit in tagged + top_prompt + top_vision + top_nonissue_prompt + top_nonissue_vision:
        selected[(audit.dataset, audit.row)] = audit
        if len(selected) >= args.exact_max_rows:
            break
    return list(selected.values())


def by_tag(audits: list[RowAudit]) -> dict[str, int]:
    counts = Counter()
    for audit in audits:
        counts.update(audit.issue_tags)
    return dict(sorted(counts.items()))


def dataset_summary(audits: list[RowAudit]) -> dict[str, Any]:
    return {
        "rows": len(audits),
        "rows_with_issues": sum(bool(item.issue_tags) for item in audits),
        "issue_tags": by_tag(audits),
        "image_count": summarise([item.image_count for item in audits]),
        "video_count": summarise([item.video_count for item in audits]),
        "video_frame_count": summarise([item.video_frame_count for item in audits]),
        "raw_pixel_max": summarise([item.raw_pixel_max for item in audits]),
        "resized_pixel_max": summarise([item.resized_pixel_max for item in audits]),
        "image_llm_tokens_sum": summarise([item.image_llm_tokens_sum for item in audits]),
        "estimated_prompt_tokens": summarise(
            [item.estimated_prompt_tokens for item in audits if item.estimated_prompt_tokens >= 0]
        ),
        "exact_prompt_tokens": summarise([item.exact_prompt_tokens for item in audits if item.exact_prompt_tokens >= 0]),
    }


def write_csv(path: Path, audits: list[RowAudit]) -> None:
    fields = list(RowAudit.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            row = asdict(audit)
            row["issue_tags"] = "|".join(audit.issue_tags)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="name=path parquet input. Can repeat.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--video-key", default="videos")
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=4096)
    parser.add_argument("--vision-token-threshold", type=int, default=2048)
    parser.add_argument("--per-image-vision-token-threshold", type=int, default=2048)
    parser.add_argument("--raw-pixel-threshold", type=int, default=IMAGE_MAX_TOKEN_NUM * 32 * 32)
    parser.add_argument("--raw-edge-threshold", type=int, default=8192)
    parser.add_argument("--aspect-threshold", type=float, default=MAX_RATIO)
    parser.add_argument("--video-frame-threshold", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--exact-top-k", type=int, default=200)
    parser.add_argument("--exact-max-rows", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs/multimodal_sample_audit")
    args = parser.parse_args()

    dataset_specs = args.dataset or list(DEFAULT_DATASETS)
    datasets = [parse_dataset_arg(spec) for spec in dataset_specs]
    model_path = first_existing_model_path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    processor = None if args.no_exact else load_processor(model_path)

    audits: list[RowAudit] = []
    full_rows_for_exact: dict[tuple[str, int], dict[str, Any]] = {}
    columns_by_dataset: dict[str, list[str]] = {}
    for dataset_name, path in datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [args.prompt_key, args.image_key, "data_source", "ability", "extra_info"]
        if args.video_key in schema_names:
            columns.append(args.video_key)
        columns = [column for column in columns if column in schema_names]
        columns_by_dataset[dataset_name] = columns
        for row_idx, row in iter_rows(path, columns, args.batch_size):
            if args.limit >= 0 and row_idx >= args.limit:
                break
            if row_idx and row_idx % args.progress_every == 0:
                print(f"[progress] {dataset_name}: {row_idx} rows", file=sys.stderr)
            audit = audit_row(dataset_name, row_idx, row, tokenizer=tokenizer, args=args)
            audits.append(audit)
            if audit.issue_tags:
                full_rows_for_exact[(dataset_name, row_idx)] = row

    exact_candidates: list[RowAudit] = []
    if processor is not None and tokenizer is not None:
        candidates = pick_exact_candidates(audits, args)
        exact_candidates = candidates
        missing: defaultdict[str, set[int]] = defaultdict(set)
        for audit in candidates:
            if (audit.dataset, audit.row) not in full_rows_for_exact:
                missing[audit.dataset].add(audit.row)
        if missing:
            path_by_name = dict(datasets)
            for dataset_name, row_ids in missing.items():
                path = path_by_name[dataset_name]
                columns = columns_by_dataset[dataset_name]
                for row_idx, row in iter_rows(path, columns, args.batch_size):
                    if row_idx in row_ids:
                        full_rows_for_exact[(dataset_name, row_idx)] = row
                    if len(row_ids) and all((dataset_name, rid) in full_rows_for_exact for rid in row_ids):
                        break

        for idx, audit in enumerate(candidates, 1):
            if idx % 50 == 0:
                print(f"[exact] {idx}/{len(candidates)}", file=sys.stderr)
            row = full_rows_for_exact.get((audit.dataset, audit.row))
            if row is not None:
                exact_check(audit, row, processor, tokenizer, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[RowAudit]] = defaultdict(list)
    for audit in audits:
        by_dataset[audit.dataset].append(audit)
    report = {
        "config": {
            "datasets": dict(datasets),
            "model_path": model_path,
            "loaded_tokenizer": tokenizer is not None,
            "loaded_processor": processor is not None,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
            "vision_token_threshold": args.vision_token_threshold,
            "raw_pixel_threshold": args.raw_pixel_threshold,
            "raw_edge_threshold": args.raw_edge_threshold,
            "image_patch_size": args.image_patch_size,
            "merge_size": args.merge_size,
        },
        "datasets": {name: dataset_summary(rows) for name, rows in by_dataset.items()},
        "all": dataset_summary(audits),
    }
    issue_rows = [audit for audit in audits if audit.issue_tags]
    top_rows = sorted(
        audits,
        key=lambda item: (
            bool(item.issue_tags),
            item.estimated_prompt_tokens,
            item.image_llm_tokens_sum,
            item.raw_pixel_max,
        ),
        reverse=True,
    )[: args.top_k]
    with (output_dir / "summary.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / "issue_rows.csv", issue_rows)
    write_csv(output_dir / "top_rows.csv", top_rows)
    write_csv(output_dir / "exact_rows.csv", exact_candidates)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[wrote] {output_dir / 'summary.json'}")
    print(f"[wrote] {output_dir / 'issue_rows.csv'}")
    print(f"[wrote] {output_dir / 'top_rows.csv'}")
    print(f"[wrote] {output_dir / 'exact_rows.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
