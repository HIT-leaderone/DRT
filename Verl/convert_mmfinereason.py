#!/usr/bin/env python3
"""Convert MFFineReason RL samples into compact verified VLM RL data.

The intended use is a GPT-5.5 rewrite/filter pass over
OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking, focused on the five
target evals used in this workspace:

  MathVista / MathVerse / LogicVista / GSM8K / VideoHolmes

This script deliberately does **not** filter by pass_rate by default.  It only
filters weakly related, empty/open-ended, medical, and trivial OCR/caption QA
samples before asking GPT-5.5 to rewrite + verify.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from convert_cosyn_math import (
    DENSE_PROMPT_TEMPLATE_VL,
    build_image_records,
    build_image_url_content,
    clean_answer_text,
    extract_json,
    init_llm_client,
    normalize_text,
)


DEFAULT_DATASET_NAME = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
DEFAULT_SPLIT = "rl"
DEFAULT_OUTPUT_DIR = "data/mmfinereason_rl_gpt55_filtered_smoke"
DEFAULT_MODEL_NAME = "gpt-5.5-2026-04-24"
DEFAULT_TARGET_BENCHES = ["MathVista", "MathVerse", "LogicVista", "GSM8K", "VideoHolmes"]
MAX_RETRIES = 3


DROP_SOURCE_KEYWORDS = (
    "pathvqa",
    "vqarad",
    "radiology",
    "medical",
    "medqa",
    "med-vqa",
    "slake",
)

KEEP_SOURCE_HINTS = (
    "mmr1",
    "euclid",
    "geo",
    "geom",
    "math",
    "wemath",
    "raven",
    "gameqa",
    "bmmr",
    "virl",
    "walton",
    "visualwebinstruct",
    "mmk12",
    "ai2d",
    "scienceqa",
    "tqa",
)

MATH_VISUAL_KEYWORDS = (
    "angle",
    "triangle",
    "circle",
    "chord",
    "arc",
    "radius",
    "diameter",
    "parallel",
    "perpendicular",
    "area",
    "perimeter",
    "volume",
    "graph",
    "function",
    "equation",
    "coordinate",
    "x-axis",
    "y-axis",
    "slope",
    "probability",
    "ratio",
    "fraction",
    "sequence",
    "pattern",
    "matrix",
    "logic",
    "puzzle",
    "raven",
    "figure",
    "diagram",
    "table",
    "chart",
    "plot",
    "bar chart",
    "line chart",
    "histogram",
    "solve",
    "calculate",
    "compute",
    "find",
    "determine",
    "value",
    "sum",
)

TRIVIAL_OR_CAPTION_PATTERNS = (
    r"\bwhat (is|are) (shown|depicted|displayed|visible|in the image)\b",
    r"\bdescribe the image\b",
    r"\bwhat color\b",
    r"\bwhich color\b",
    r"\bread the text\b",
    r"\bwhat text\b",
    r"\bwhat method is used\b",
    r"\bwhat machine learning models are compared\b",
    r"\bwhat are the main differences\b",
    r"\bsummarize\b",
)

OPEN_ENDED_PATTERNS = (
    r"\bdescribe\b",
    r"\bexplain\b",
    r"\bdiscuss\b",
    r"\bcompare\b",
    r"\bmain differences\b",
    r"\bwhat can you infer\b",
)

STRONG_REASONING_PATTERNS = (
    r"\bsolve\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bfind\b",
    r"\bdetermine\b",
    r"\bprove\b",
    r"\bshow that\b",
    r"\bderive\b",
    r"\bjustify\b",
    r"\bwhat is the value\b",
)


def normalize_source(value: Any) -> str:
    return normalize_text(value).lower()


def answer_is_empty_or_open(answer: str, question: str) -> Tuple[bool, str]:
    ans = clean_answer_text(answer)
    q = normalize_text(question).lower()
    if not ans or ans.lower() in {"none", "null", "nan", "n/a", "unknown", "not given"}:
        return True, "empty_answer"
    words = ans.split()
    is_open_ended = any(re.search(pattern, q) for pattern in OPEN_ENDED_PATTERNS)
    has_strong_reasoning = any(re.search(pattern, q) for pattern in STRONG_REASONING_PATTERNS)
    # Long mathematical proofs / derivations often appear in the dataset's
    # `answer` field.  Do not reject those locally; GPT rewrite+verifier can
    # compress them into compact steps.  Only drop long answers when the task
    # itself is open-ended/descriptive rather than well-posed reasoning.
    if len(words) > 80 and is_open_ended and not has_strong_reasoning:
        return True, "answer_too_long_open_ended"
    if len(words) > 25 and is_open_ended and not has_strong_reasoning:
        return True, "open_ended_descriptive_answer"
    if ans.strip() in {"", "[]", "{}"}:
        return True, "empty_answer"
    return False, "answer_ok"


def has_math_or_bench_signal(text: str, source: str) -> bool:
    t = text.lower()
    if any(hint in source for hint in KEEP_SOURCE_HINTS):
        return True
    if re.search(r"[\d=+\-*/^√π∠°%$]", t):
        return True
    return any(keyword in t for keyword in MATH_VISUAL_KEYWORDS)


def looks_trivial_or_caption_qa(question: str, answer: str, source: str, caption: str) -> Tuple[bool, str]:
    q = normalize_text(question).lower()
    ans = clean_answer_text(answer)
    text = " ".join([q, normalize_text(caption).lower(), source])

    # Keep explicit math/logic/geometry questions even if the answer is short.
    strong_task_signal = any(
        keyword in q
        for keyword in [
            "solve",
            "calculate",
            "compute",
            "find",
            "determine",
            "prove",
            "show that",
            "derive",
            "justify",
            "value",
            "area",
            "perimeter",
            "volume",
            "probability",
            "angle",
            "arc",
            "radius",
            "chord",
            "slope",
            "which figure",
            "logical sequence",
        ]
    )
    if strong_task_signal:
        return False, "nontrivial_task_signal"

    if any(re.search(pattern, q) for pattern in TRIVIAL_OR_CAPTION_PATTERNS):
        return True, "trivial_caption_or_recognition_qa"

    if len(ans.split()) <= 3 and q.startswith(("what ", "which ", "who ", "where ")):
        if not has_math_or_bench_signal(text, source):
            return True, "short_trivial_vqa"

    return False, "not_trivial"


def local_prefilter(row: Dict[str, Any], drop_pass_rate_one: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    source = normalize_source(row.get("source"))
    question = normalize_text(row.get("question"))
    answer = clean_answer_text(row.get("answer"))
    caption = normalize_text(row.get("caption"))
    pass_rate = row.get("pass_rate")
    meta = {
        "source": row.get("source"),
        "pass_rate": pass_rate,
        "is_consistent": row.get("is_consistent"),
    }

    if any(keyword in source for keyword in DROP_SOURCE_KEYWORDS):
        return False, "weak_related_medical_source", meta
    if drop_pass_rate_one:
        try:
            if float(pass_rate) >= 1.0:
                return False, "easy_pass_rate_1", meta
        except Exception:
            pass
    if not question:
        return False, "empty_question", meta
    bad_answer, answer_reason = answer_is_empty_or_open(answer, question)
    if bad_answer:
        return False, answer_reason, meta
    text = " ".join([question, caption, answer])
    if not has_math_or_bench_signal(text, source):
        return False, "low_bench_relevance_local", meta
    is_trivial, trivial_reason = looks_trivial_or_caption_qa(question, answer, source, caption)
    if is_trivial:
        return False, trivial_reason, meta
    return True, "prefilter_pass", meta


def infer_capability_tags(text: str, source: str) -> List[str]:
    t = f"{text} {source}".lower()
    tags: List[str] = []
    checks = [
        ("geometry", ["triangle", "circle", "angle", "arc", "chord", "radius", "perimeter", "area", "parallel"]),
        ("chart_table", ["chart", "graph", "plot", "table", "axis", "bar", "line chart"]),
        ("algebra", ["equation", "function", "quadratic", "slope", "coordinate", "matrix", "variable"]),
        ("counting_probability", ["probability", "count", "ways", "spinner", "fraction", "ratio"]),
        ("logic_puzzle", ["raven", "puzzle", "logical sequence", "solitaire", "game", "pattern"]),
        ("science_diagram", ["physics", "force", "circuit", "cell", "biology", "chemistry"]),
    ]
    for tag, keywords in checks:
        if any(k in t for k in keywords):
            tags.append(tag)
    return tags or ["visual_math"]


def compute_local_bench_relevance(question: str, answer: str, caption: str, source: str) -> Dict[str, int]:
    text = " ".join([question, answer, caption, source]).lower()
    source_l = source.lower()
    score = {bench: 0 for bench in DEFAULT_TARGET_BENCHES}

    if any(k in text for k in ["geometry", "triangle", "circle", "angle", "graph", "chart", "plot", "table", "function", "probability"]):
        score["MathVista"] = 4
        score["MathVerse"] = 4
    elif any(k in text for k in ["math", "calculate", "solve", "value", "equation"]):
        score["MathVista"] = 3
        score["MathVerse"] = 3

    if any(k in text for k in ["raven", "puzzle", "logical sequence", "pattern", "solitaire", "gameqa", "which figure"]):
        score["LogicVista"] = 4
    elif any(k in text for k in ["diagram", "spatial", "infer", "deduce"]):
        score["LogicVista"] = 2

    if any(k in text for k in ["word problem", "arithmetic", "ratio", "percent", "sum", "fraction"]) and "image" not in question.lower()[:20]:
        score["GSM8K"] = 2

    if any(k in text for k in ["video", "temporal", "sequence of frames", "motion"]):
        score["VideoHolmes"] = 3
    elif "gameqa" in source_l:
        score["VideoHolmes"] = 1

    return score


def build_rewrite_messages(row: Dict[str, Any], target_benches: List[str]) -> List[Dict[str, Any]]:
    payload = {
        "id": row.get("id"),
        "source": row.get("source"),
        "pass_rate": row.get("pass_rate"),
        "is_consistent": row.get("is_consistent"),
        "question": row.get("question"),
        "reference_answer": row.get("answer"),
        "caption_weak_reference": row.get("caption"),
        "qwen3vl_235b_thinking_response_weak_reference": row.get("qwen3vl_235b_thinking_response"),
        "target_benches": target_benches,
    }
    text = (
        "You are converting a multimodal STEM reasoning sample into compact, high-quality VLM RL data.\n"
        "The sample includes a Qwen3-VL-235B thinking trace and caption as weak references. They may be verbose "
        "or wrong; verify against the image and problem statement.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)[:16000]}\n\n"
        "Reject if the sample is medical/pathology, pure OCR/caption recognition, trivial non-reasoning VQA, "
        "empty/open-ended, irrelevant to MathVista/MathVerse/LogicVista/GSM8K/VideoHolmes, or not visually grounded.\n\n"
        "If accepted, rewrite into a concise open-form training target. Do not keep the very long teacher trace. "
        "Use 3-10 compact steps. Every visual claim must be supported by visible image evidence or explicit text.\n\n"
        "Return strict JSON only:\n"
        "{\n"
        '  "verdict": "PASS|FAIL",\n'
        '  "reject_reason": "brief reason if FAIL",\n'
        '  "clean_question": "student-facing question, with <image> omitted",\n'
        '  "final_answer": "short final answer",\n'
        '  "solution_steps": ["compact grounded step", "..."],\n'
        '  "visual_evidence": ["visible label/relation used", "..."],\n'
        '  "difficulty": "easy|medium|hard",\n'
        '  "visual_dependency": "weak|medium|strong",\n'
        '  "capability_tags": ["geometry", "chart_table", "logic_puzzle", "..."],\n'
        '  "bench_relevance": {"MathVista": 0, "MathVerse": 0, "LogicVista": 0, "GSM8K": 0, "VideoHolmes": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Bench relevance scores use 0=unrelated, 1=weak, 2=somewhat, 3=good, 4=strong."
    )
    content = build_image_url_content([row.get("image")])
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": "You are a strict multimodal STEM data curator. Output JSON only."},
        {"role": "user", "content": content},
    ]


def normalize_rewrite(parsed: Dict[str, Any], row: Dict[str, Any], local_bench: Dict[str, int]) -> Dict[str, Any]:
    question = normalize_text(parsed.get("clean_question")) or normalize_text(row.get("question"))
    question = re.sub(r"^\s*<image>\s*", "", question, flags=re.IGNORECASE).strip()
    answer = clean_answer_text(parsed.get("final_answer")) or clean_answer_text(row.get("answer"))
    steps = parsed.get("solution_steps")
    if not isinstance(steps, list):
        steps = []
    steps = [normalize_text(step) for step in steps if normalize_text(step)]
    evidence = parsed.get("visual_evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence = [normalize_text(x) for x in evidence if normalize_text(x)]
    tags = parsed.get("capability_tags")
    if not isinstance(tags, list):
        tags = infer_capability_tags(" ".join([question, answer, normalize_text(row.get("caption"))]), normalize_source(row.get("source")))
    tags = [normalize_text(tag).lower() for tag in tags if normalize_text(tag)]
    bench = parsed.get("bench_relevance")
    if not isinstance(bench, dict):
        bench = local_bench
    bench = {bench_name: int(float(bench.get(bench_name, local_bench.get(bench_name, 0)) or 0)) for bench_name in DEFAULT_TARGET_BENCHES}
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    dep = normalize_text(parsed.get("visual_dependency")).lower()
    if dep not in {"weak", "medium", "strong"}:
        dep = "medium"
    difficulty = normalize_text(parsed.get("difficulty")).lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    return {
        "verdict": normalize_text(parsed.get("verdict")).upper() or "FAIL",
        "reject_reason": normalize_text(parsed.get("reject_reason")),
        "clean_question": question,
        "final_answer": answer,
        "solution_steps": steps,
        "visual_evidence": evidence,
        "difficulty": difficulty,
        "visual_dependency": dep,
        "capability_tags": tags,
        "bench_relevance": bench,
        "confidence": confidence,
    }


def build_verifier_messages(row: Dict[str, Any], rewrite: Dict[str, Any], target_benches: List[str]) -> List[Dict[str, Any]]:
    payload = {
        "source": row.get("source"),
        "original_question": row.get("question"),
        "reference_answer": row.get("answer"),
        "caption_weak_reference": row.get("caption"),
        "rewrite": rewrite,
        "target_benches": target_benches,
    }
    text = (
        "Audit this rewritten multimodal RL sample. A correct final answer alone is not sufficient.\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)[:14000]}\n\n"
        "Audit dimensions:\n"
        "1. `visual_fidelity`: every visual claim in the question/steps must be directly visible or a necessary consequence.\n"
        "2. `reasoning_faithfulness`: steps must follow logically; no hidden-answer premise, ad hoc assumptions, or circular reasoning.\n"
        "3. `semantic_answer_correctness`: final_answer must match the visible problem and reference answer when the reference is valid.\n"
        "4. `bench_relevance`: the sample must be useful for at least one of MathVista/MathVerse/LogicVista/GSM8K/VideoHolmes.\n"
        "5. `not_trivial_or_open`: reject pure OCR/caption, trivial recognition, open-ended descriptive QA, or empty-answer cases.\n\n"
        "Decision rule: PASS only if all five dimensions are true. Return strict JSON only:\n"
        "{\n"
        '  "visual_fidelity": true,\n'
        '  "reasoning_faithfulness": true,\n'
        '  "semantic_answer_correctness": true,\n'
        '  "bench_relevance": true,\n'
        '  "not_trivial_or_open": true,\n'
        '  "verdict": "PASS",\n'
        '  "reason": "brief reason"\n'
        "}"
    )
    content = build_image_url_content([row.get("image")])
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": "You are a strict verifier for multimodal reasoning training data. Output JSON only."},
        {"role": "user", "content": content},
    ]


def call_gpt_rewrite_and_verify(row: Dict[str, Any], local_bench: Dict[str, int], llm_client: Any, target_benches: List[str]) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    if llm_client is None:
        return None, "missing_llm_client", {}
    last_audit: Dict[str, Any] = {}
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.send_stable_request(build_rewrite_messages(row, target_benches), temperature=0.0)
            parsed = extract_json(response)
            if not isinstance(parsed, dict):
                continue
            rewrite = normalize_rewrite(parsed, row, local_bench)
            if rewrite["verdict"] != "PASS":
                return None, rewrite.get("reject_reason") or "gpt_rewrite_fail", {"rewrite": rewrite}
            if not rewrite["clean_question"] or not rewrite["final_answer"] or not rewrite["solution_steps"]:
                return None, "gpt_rewrite_missing_fields", {"rewrite": rewrite}
            if max(rewrite["bench_relevance"].values() or [0]) < 2:
                return None, "gpt_low_bench_relevance", {"rewrite": rewrite}

            review = llm_client.send_stable_request(build_verifier_messages(row, rewrite, target_benches), temperature=0.0)
            review_parsed = extract_json(review)
            if isinstance(review_parsed, dict):
                last_audit = review_parsed
                ok = (
                    str(review_parsed.get("verdict", "")).upper() == "PASS"
                    and review_parsed.get("visual_fidelity") is True
                    and review_parsed.get("reasoning_faithfulness") is True
                    and review_parsed.get("semantic_answer_correctness") is True
                    and review_parsed.get("bench_relevance") is True
                    and review_parsed.get("not_trivial_or_open") is True
                )
                if ok:
                    return rewrite, "gpt55_rewrite_verified", last_audit
                return None, normalize_text(review_parsed.get("reason")) or "gpt_verifier_fail", {"rewrite": rewrite, "audit": review_parsed}
        except Exception as exc:
            print(f"[GPT Error] id={row.get('id')} attempt={attempt + 1}: {exc}")
            time.sleep(1)
    return None, "gpt_rewrite_or_verify_failed", last_audit


def deterministic_rewrite(row: Dict[str, Any], local_bench: Dict[str, int]) -> Dict[str, Any]:
    question = re.sub(r"^\s*<image>\s*", "", normalize_text(row.get("question")), flags=re.IGNORECASE).strip()
    answer = clean_answer_text(row.get("answer"))
    teacher = normalize_text(row.get("qwen3vl_235b_thinking_response"))
    if teacher:
        stripped = re.sub(r"</?think>", "", teacher, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", stripped) if s.strip()]
        steps = sentences[:6] or [stripped[:1000]]
    else:
        steps = [f"Use the visual problem statement to solve for the answer {answer}."]
    return {
        "verdict": "PASS",
        "reject_reason": "",
        "clean_question": question,
        "final_answer": answer,
        "solution_steps": steps,
        "visual_evidence": [],
        "difficulty": "medium",
        "visual_dependency": "medium",
        "capability_tags": infer_capability_tags(" ".join([question, answer, normalize_text(row.get("caption"))]), normalize_source(row.get("source"))),
        "bench_relevance": local_bench,
        "confidence": 0.5,
    }


def build_prompt(question: str) -> List[Dict[str, str]]:
    prompt_content = DENSE_PROMPT_TEMPLATE_VL.replace("{Question}", question)
    if "<image>" not in prompt_content:
        prompt_content = "<image>\n" + prompt_content
    return [{"role": "user", "content": prompt_content}]


def build_reject(index: int, row: Dict[str, Any], reason: str, local_meta: Dict[str, Any], audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "original_index": index,
        "reject_reason": reason,
        "source": row.get("source"),
        "pass_rate": row.get("pass_rate"),
        "is_consistent": row.get("is_consistent"),
        "local_meta": local_meta,
        "audit": audit or {},
    }


def process_row(
    index: int,
    row: Dict[str, Any],
    llm_client: Any,
    skip_generation: bool,
    target_benches: List[str],
    drop_pass_rate_one: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    ok, reason, meta = local_prefilter(row, drop_pass_rate_one=drop_pass_rate_one)
    if not ok:
        return None, build_reject(index, row, reason, meta)

    raw_images = [row.get("image")] if row.get("image") is not None else []
    images_field = build_image_records(raw_images)
    if not images_field:
        return None, build_reject(index, row, "missing_or_unencodable_image", meta)

    question = normalize_text(row.get("question"))
    answer = clean_answer_text(row.get("answer"))
    caption = normalize_text(row.get("caption"))
    source = normalize_text(row.get("source"))
    local_bench = compute_local_bench_relevance(question, answer, caption, source)

    if skip_generation:
        rewrite = deterministic_rewrite(row, local_bench)
        rewrite_mode = "skip_deterministic"
        verifier_audit: Dict[str, Any] = {
            "verdict": "SKIP",
            "reason": "skip_generation",
            "visual_fidelity": None,
            "reasoning_faithfulness": None,
            "semantic_answer_correctness": None,
            "bench_relevance": None,
            "not_trivial_or_open": None,
        }
    else:
        rewrite, rewrite_mode, verifier_audit = call_gpt_rewrite_and_verify(row, local_bench, llm_client, target_benches)
        if not rewrite:
            return None, build_reject(index, row, rewrite_mode, meta, verifier_audit)

    extra_info: Dict[str, Any] = {
        "answer": rewrite["final_answer"],
        "problem": rewrite["clean_question"],
        "index": index,
        "original_index": index,
        "id": row.get("id"),
        "dataset": "mmfinereason_rl",
        "hf_dataset": DEFAULT_DATASET_NAME,
        "split": DEFAULT_SPLIT,
        "data_type": "vision",
        "num_images": len(images_field),
        "source": row.get("source"),
        "pass_rate": row.get("pass_rate"),
        "is_consistent": row.get("is_consistent"),
        "source_question": question,
        "source_answer": answer,
        "caption_weak_reference": caption,
        "teacher_response_prefix": normalize_text(row.get("qwen3vl_235b_thinking_response"))[:2000],
        "visual_evidence": rewrite.get("visual_evidence", []),
        "visual_dependency": rewrite.get("visual_dependency"),
        "capability_tags": rewrite.get("capability_tags"),
        "difficulty": rewrite.get("difficulty"),
        "confidence": rewrite.get("confidence"),
        "bench_relevance": rewrite.get("bench_relevance"),
        "local_bench_relevance": local_bench,
        "target_benches": target_benches,
        "rewrite_mode": rewrite_mode,
        "verifier_reason": verifier_audit.get("reason"),
        "verifier_visual_fidelity": verifier_audit.get("visual_fidelity"),
        "verifier_reasoning_faithfulness": verifier_audit.get("reasoning_faithfulness"),
        "verifier_semantic_answer_correctness": verifier_audit.get("semantic_answer_correctness"),
        "verifier_bench_relevance": verifier_audit.get("bench_relevance"),
        "verifier_not_trivial_or_open": verifier_audit.get("not_trivial_or_open"),
        "verifier_verdict": verifier_audit.get("verdict"),
    }
    result = {
        "data_source": "mmfinereason_rl",
        "prompt": build_prompt(rewrite["clean_question"]),
        "ability": "visual_math",
        "reward_model": {
            "ground_truth": json.dumps(
                {"answer": rewrite["final_answer"], "steps": rewrite["solution_steps"]},
                ensure_ascii=False,
            ),
            "style": "rule",
        },
        "extra_info": extra_info,
        "images": images_field,
    }
    return result, None


def iter_rows(dataset: Any, limit: Optional[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield idx, row


def load_split(dataset_name: str, split: str, limit: Optional[int], streaming: bool):
    # The official dataset has 193 SFT shards + 5 RL shards.  Calling
    # load_dataset(name, split="rl[:N]") can still inspect/download SFT shards in
    # some datasets versions, so use the parquet files for the requested split
    # directly.  This keeps smoke tests bounded and full conversion streamable.
    if dataset_name == DEFAULT_DATASET_NAME and split in {"rl", "sft"}:
        pattern = f"hf://datasets/{dataset_name}/data/{split}-*.parquet"
        if streaming:
            return load_dataset("parquet", data_files=pattern, split="train", streaming=True)
        split_expr = f"train[:{limit}]" if limit is not None else "train"
        return load_dataset("parquet", data_files=pattern, split=split_expr)

    if streaming:
        return load_dataset(dataset_name, split=split, streaming=True)
    split_expr = f"{split}[:{limit}]" if limit is not None else split
    return load_dataset(dataset_name, split=split_expr)


def write_outputs(output_dir: str, split: str, accepted: List[Dict[str, Any]], rejects: List[Dict[str, Any]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    accepted_path = os.path.join(output_dir, f"{split}.parquet")
    reject_path = os.path.join(output_dir, f"{split}.rejected.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    if accepted:
        accepted = sorted(accepted, key=lambda x: x.get("extra_info", {}).get("original_index", -1))
        pd.DataFrame(accepted).to_parquet(accepted_path, index=False)
    else:
        # Keep an empty marker file path predictable without forcing an empty
        # nested schema parquet.
        open(accepted_path + ".empty", "w", encoding="utf-8").close()
    with open(reject_path, "w", encoding="utf-8") as f:
        for record in sorted(rejects, key=lambda x: x.get("original_index", -1)):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "accepted": len(accepted),
        "rejected": len(rejects),
        "reject_reasons": Counter(r.get("reject_reason", "unknown") for r in rejects).most_common(),
        "accepted_path": accepted_path if accepted else accepted_path + ".empty",
        "reject_path": reject_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))


def accepted_parts_dir(output_dir: str, split: str) -> str:
    return os.path.join(output_dir, f"{split}.parquet.parts")


def iter_part_files(output_dir: str, split: str) -> List[str]:
    parts_dir = accepted_parts_dir(output_dir, split)
    if not os.path.isdir(parts_dir):
        return []
    return [
        os.path.join(parts_dir, name)
        for name in sorted(os.listdir(parts_dir))
        if name.endswith(".parquet")
    ]


def next_part_path(output_dir: str, split: str) -> str:
    parts_dir = accepted_parts_dir(output_dir, split)
    os.makedirs(parts_dir, exist_ok=True)
    return os.path.join(parts_dir, f"part-{len(iter_part_files(output_dir, split)) + 1:06d}.parquet")


def write_accepted_part(output_dir: str, split: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    part_path = next_part_path(output_dir, split)
    rows = sorted(rows, key=lambda x: x.get("extra_info", {}).get("original_index", -1))
    pd.DataFrame(rows).to_parquet(part_path, index=False)
    return part_path


def append_rejects(output_dir: str, split: str, rejects: List[Dict[str, Any]]) -> Optional[str]:
    if not rejects:
        return None
    os.makedirs(output_dir, exist_ok=True)
    reject_path = os.path.join(output_dir, f"{split}.rejected.jsonl")
    with open(reject_path, "a", encoding="utf-8") as f:
        for record in sorted(rejects, key=lambda x: x.get("original_index", -1)):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return reject_path


def extract_original_index(record: Dict[str, Any]) -> Optional[int]:
    extra = record.get("extra_info")
    if isinstance(extra, dict):
        try:
            return int(extra.get("original_index"))
        except Exception:
            return None
    return None


def read_accepted_indices(output_dir: str, split: str) -> set[int]:
    indices: set[int] = set()
    for path in iter_part_files(output_dir, split):
        try:
            df = pd.read_parquet(path, columns=["extra_info"])
            for row in df.to_dict("records"):
                idx = extract_original_index(row)
                if idx is not None:
                    indices.add(idx)
        except Exception as exc:
            print(f"Warning: failed to read accepted part cache {path}: {exc}")
    legacy_path = os.path.join(output_dir, f"{split}.parquet")
    if os.path.exists(legacy_path):
        try:
            df = pd.read_parquet(legacy_path, columns=["extra_info"])
            for row in df.to_dict("records"):
                idx = extract_original_index(row)
                if idx is not None:
                    indices.add(idx)
        except Exception as exc:
            print(f"Warning: failed to read legacy accepted cache {legacy_path}: {exc}")
    return indices


def read_reject_cache(output_dir: str, split: str) -> Tuple[set[int], Counter]:
    indices: set[int] = set()
    counter: Counter = Counter()
    reject_path = os.path.join(output_dir, f"{split}.rejected.jsonl")
    if not os.path.exists(reject_path):
        return indices, counter
    with open(reject_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            try:
                indices.add(int(record.get("original_index")))
            except Exception:
                pass
            counter[record.get("reject_reason", "unknown")] += 1
    return indices, counter


def write_summary(output_dir: str, split: str, summary: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))


def convert_dataset(
    dataset_name: str,
    split: str,
    output_dir: str,
    limit: Optional[int],
    max_accept: Optional[int],
    max_workers: int,
    skip_generation: bool,
    model_name: str,
    target_benches: List[str],
    drop_pass_rate_one: bool,
    streaming: bool,
    save_interval: int,
    submit_batch_size: int,
    resume: bool,
) -> Dict[str, Any]:
    print(
        f"Loading {dataset_name} split={split}, limit={limit}, streaming={streaming}, "
        f"skip_generation={skip_generation}, max_workers={max_workers}"
    )
    dataset = load_split(dataset_name, split, limit=limit, streaming=streaming)
    try:
        dataset = dataset.cast_column("image", HFImage(decode=True))
    except Exception:
        pass
    llm_client = None if skip_generation else init_llm_client(model_name)

    os.makedirs(output_dir, exist_ok=True)
    reject_path = os.path.join(output_dir, f"{split}.rejected.jsonl")
    accepted_cache: set[int] = set()
    reject_cache: set[int] = set()
    reason_counter: Counter = Counter()
    if resume:
        accepted_cache = read_accepted_indices(output_dir, split)
        reject_cache, reason_counter = read_reject_cache(output_dir, split)
        if accepted_cache or reject_cache:
            print(
                f"Resume cache: accepted={len(accepted_cache)}, rejected={len(reject_cache)}, "
                f"top_reject={reason_counter.most_common(5)}"
            )
    else:
        if os.path.exists(reject_path):
            os.replace(reject_path, reject_path + ".old")
            print(f"Moved old reject file to {reject_path}.old")

    processed = accepted_cache | reject_cache
    accepted_buffer: List[Dict[str, Any]] = []
    rejects_buffer: List[Dict[str, Any]] = []
    accepted_total = len(accepted_cache)
    rejected_total = len(reject_cache)
    completed_this_run = 0
    scanned = 0

    def flush(force: bool = False) -> None:
        nonlocal accepted_buffer, rejects_buffer
        if not force and len(accepted_buffer) < save_interval and len(rejects_buffer) < save_interval:
            return
        wrote = False
        part_path = write_accepted_part(output_dir, split, accepted_buffer)
        if part_path:
            print(f"Checkpoint: wrote {len(accepted_buffer)} accepted -> {part_path}")
            accepted_buffer = []
            wrote = True
        reject_written = append_rejects(output_dir, split, rejects_buffer)
        if reject_written:
            print(f"Checkpoint: appended {len(rejects_buffer)} rejects -> {reject_written}")
            rejects_buffer = []
            wrote = True
        if wrote or force:
            summary = {
                "accepted": accepted_total,
                "rejected": rejected_total,
                "completed_this_run": completed_this_run,
                "scanned": scanned,
                "parts_dir": accepted_parts_dir(output_dir, split),
                "reject_path": reject_path,
                "reject_reasons": reason_counter.most_common(),
                "running": not force,
            }
            write_summary(output_dir, split, summary)

    batch: List[Tuple[int, Dict[str, Any]]] = []

    def run_batch(batch_rows: List[Tuple[int, Dict[str, Any]]]) -> None:
        nonlocal accepted_total, rejected_total, completed_this_run
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_row,
                    idx,
                    row,
                    llm_client,
                    skip_generation,
                    target_benches,
                    drop_pass_rate_one,
                ): idx
                for idx, row in batch_rows
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]
                result, reject = future.result()
                if result:
                    accepted_buffer.append(result)
                    accepted_total += 1
                if reject:
                    rejects_buffer.append(reject)
                    rejected_total += 1
                    reason_counter[reject.get("reject_reason", "unknown")] += 1
                processed.add(idx)
                completed_this_run += 1
                flush(force=False)

    for idx, row in iter_rows(dataset, limit):
        scanned = idx + 1
        if idx in processed:
            continue
        if max_accept is not None and accepted_total >= max_accept:
            break
        batch.append((idx, row))
        if len(batch) >= max(1, submit_batch_size):
            run_batch(batch)
            print(
                f"Progress: scanned≈{idx + 1}, accepted={accepted_total}, rejected={rejected_total}, "
                f"top_reject={reason_counter.most_common(5)}"
            )
            batch = []
    if batch and (max_accept is None or accepted_total < max_accept):
        run_batch(batch)

    flush(force=True)
    summary = {
        "accepted": accepted_total,
        "rejected": rejected_total,
        "completed_this_run": completed_this_run,
        "scanned": scanned,
        "parts_dir": accepted_parts_dir(output_dir, split),
        "reject_path": reject_path,
        "reject_reasons": reason_counter.most_common(),
        "running": False,
    }
    write_summary(output_dir, split, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MFFineReason RL data to verified VLM RL parquet.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Rows scanned from the split; default scans the full split")
    parser.add_argument("--max_accept", type=int, default=None, help="Stop after this many accepted rows")
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--skip_generation", action="store_true", help="Use local deterministic rewrite; no GPT calls")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target_benches", nargs="+", default=DEFAULT_TARGET_BENCHES)
    parser.add_argument("--drop_pass_rate_one", action="store_true", help="Optional; off by default per current experiment")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Use streaming load_dataset")
    parser.add_argument("--save_interval", type=int, default=20, help="Checkpoint after this many buffered accepts/rejects")
    parser.add_argument("--submit_batch_size", type=int, default=64, help="Rows submitted to the executor at once")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip accepted/rejected original_index values already in output_dir")
    parser.add_argument(
        "--safe_exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use os._exit(0) after successful CLI completion to avoid datasets/PIL streaming finalizer crashes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = convert_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
        max_accept=args.max_accept,
        max_workers=args.max_workers,
        skip_generation=args.skip_generation,
        model_name=args.model_name,
        target_benches=args.target_benches,
        drop_pass_rate_one=args.drop_pass_rate_one,
        streaming=args.streaming,
        save_interval=args.save_interval,
        submit_batch_size=args.submit_batch_size,
        resume=args.resume,
    )
    print("Final:", json.dumps(summary, ensure_ascii=False, indent=2))
    if args.safe_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
