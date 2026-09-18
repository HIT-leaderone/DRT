#!/usr/bin/env python3
"""GPT correctness judge for DRT pass@k on VisionR1_DAPO outputs.

This script judges DRT candidate answer slots against the reference answer,
caches each judgement, and recomputes pass@1/2/4/8/16.  By default it uses
slot-level judging: 100 samples x 16 rollouts = 1600 GPT judgements.  A
legacy unique-answer mode is available for cheaper ablations.
It is split-aware: pass --split train or --split test with matching artifacts.

API credentials are read from environment variables or CLI flags.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import math
import mimetypes
import os
import pickle
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-2024-05-13"
JUDGE_VERSION = "meta_v1"
KS = (1, 2, 4, 8, 16)

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
OPTION_RE = re.compile(r"^\s*(?:option\s*)?[\(\[]?([A-Z])[\)\].:：]?(?:\s|$)", re.IGNORECASE)


def resolve_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def resolve_api_base() -> str:
    base = os.environ.get("OPENAI_API_BASE", "").strip()
    if base:
        return base
    return DEFAULT_API_BASE


def extract_answer(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    tag = ANSWER_TAG_RE.findall(text)
    if tag:
        return tag[-1].strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else text


def normalize_answer(value: Any) -> str:
    text = extract_answer(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip().strip("`*_ \t\r\n。.!！?？,，;；:")
    option = OPTION_RE.match(text)
    if option:
        return option.group(1).upper()
    lower = re.sub(r"\s+", " ", text.lower()).strip()
    if lower in {"yes", "y", "true"}:
        return "yes"
    if lower in {"no", "n", "false"}:
        return "no"
    numeric = lower.replace(",", "").strip("$% ")
    if re.fullmatch(r"-?\d+(?:\.0+)?", numeric):
        try:
            value_f = float(numeric)
            if math.isfinite(value_f):
                return str(int(value_f))
        except Exception:
            pass
        return numeric
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)", numeric):
        return numeric.rstrip("0").rstrip(".")
    return lower.strip("。.!！?？,，;；:")


def safe_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def safe_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_image_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    parsed = safe_parse_json(value)
    if isinstance(parsed, list):
        return [str(x) for x in parsed if str(x).strip()]
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan" or text == "[]":
        return []
    return [text]


def parse_reference(ground_truth: Any, answer: Any) -> dict[str, Any]:
    """Extract final answer and reference steps from VisionR1_DAPO metadata."""
    parsed = safe_parse_json(ground_truth)
    if isinstance(parsed, dict) and isinstance(parsed.get("ground_truth"), str):
        nested = safe_parse_json(parsed.get("ground_truth"))
        if isinstance(nested, dict):
            parsed = nested
    if not isinstance(parsed, dict):
        parsed = {}
    steps = parsed.get("steps") or parsed.get("reference_steps") or []
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, list):
        steps = []
    ref_answer = parsed.get("answer", answer)
    return {
        "answer": "" if ref_answer is None else str(ref_answer),
        "steps": [str(x) for x in steps],
        "raw": ground_truth,
    }


def compact_question(question: str, extra_info: dict[str, Any]) -> tuple[str, str]:
    """Return (clean problem, full prompt excerpt)."""
    problem = str(extra_info.get("problem", "") or "").strip()
    q = str(question or "").strip()
    if not problem:
        # Most DAPO prompts append the DCT instruction after a blank line.
        marker = "\n\nAnalyze "
        problem = q.split(marker, 1)[0].strip() if marker in q else q
    return problem, q


def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table: {path}")


def load_rows_from_pkl(pkl_path: Path, data_tsv: Path, limit: int = 0) -> list[dict[str, Any]]:
    data = read_table(data_tsv)
    meta = {str(row["index"]): row.to_dict() for _, row in data.iterrows()}
    obj = pickle.load(open(pkl_path, "rb"))
    rows = []
    for key in sorted(obj, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        pred = obj[key]
        item = meta.get(str(key))
        if item is None or not isinstance(pred, dict):
            continue
        extra_info = safe_parse_json(item.get("extra_info", "")) or {}
        reference = parse_reference(item.get("ground_truth", ""), item.get("answer", ""))
        problem, full_prompt = compact_question(str(item.get("question", "")), extra_info if isinstance(extra_info, dict) else {})
        rows.append(
            {
                "index": str(key),
                "question": full_prompt,
                "problem": problem,
                "answer": str(item.get("answer", "")),
                "reference_answer": reference["answer"],
                "reference_steps": reference["steps"],
                "ground_truth": str(item.get("ground_truth", "")),
                "image_paths": safe_image_paths(item.get("image_path", "")),
                "data_source": str(item.get("data_source", "")),
                "ability": str(item.get("ability", "")),
                "original_index": str(item.get("original_index", "")),
                "extra_info": extra_info if isinstance(extra_info, dict) else {},
                "candidate_answers": safe_json_list(pred.get("drt_candidate_answers")),
                "candidate_responses": safe_json_list(pred.get("drt_rollouts")),
                "candidate_output_tokens": safe_json_list(pred.get("drt_candidate_output_tokens")),
                "selected_index": pred.get("drt_selected_index"),
                "selected_answer": str(pred.get("drt_selected_answer", "")),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def load_rows_from_tsv(pred_tsv: Path, limit: int = 0) -> list[dict[str, Any]]:
    df = read_table(pred_tsv)
    rows = []
    for _, row in df.iterrows():
        extra_info = safe_parse_json(row.get("extra_info", "")) or {}
        reference = parse_reference(row.get("ground_truth", ""), row.get("answer", ""))
        problem, full_prompt = compact_question(str(row.get("question", "")), extra_info if isinstance(extra_info, dict) else {})
        rows.append(
            {
                "index": str(row.get("index", "")),
                "question": full_prompt,
                "problem": problem,
                "answer": str(row.get("answer", "")),
                "reference_answer": reference["answer"],
                "reference_steps": reference["steps"],
                "ground_truth": str(row.get("ground_truth", "")),
                "image_paths": safe_image_paths(row.get("image_path", "")),
                "data_source": str(row.get("data_source", "")),
                "ability": str(row.get("ability", "")),
                "original_index": str(row.get("original_index", "")),
                "extra_info": extra_info if isinstance(extra_info, dict) else {},
                "candidate_answers": safe_json_list(row.get("drt_candidate_answers")),
                "candidate_responses": safe_json_list(row.get("drt_rollouts")),
                "candidate_output_tokens": safe_json_list(row.get("drt_candidate_output_tokens")),
                "selected_index": row.get("drt_selected_index", None),
                "selected_answer": str(row.get("drt_selected_answer", "")),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def judgement_key(
    split: str,
    index: str,
    answer: str,
    candidate: str,
    judge_version: str = JUDGE_VERSION,
    slot_idx: int | str | None = None,
) -> str:
    raw = json.dumps(
        {
            "judge_version": judge_version,
            "split": split,
            "index": str(index),
            "answer": str(answer),
            "candidate": str(candidate),
            "slot_idx": None if slot_idx is None else str(slot_idx),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def image_to_data_url(path: str, max_image_mb: float) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    if p.stat().st_size > max_image_mb * 1024 * 1024:
        return ""
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def candidate_response_for_prompt(row: dict[str, Any], candidate: str, slot_idx: int | str | None = None) -> tuple[str, str]:
    """Prefer the raw rollout DRT response when the inference artifact stored it."""
    responses = row.get("candidate_responses") or []
    try:
        idx = int(slot_idx) if slot_idx is not None else -1
    except Exception:
        idx = -1
    if 0 <= idx < len(responses) and str(responses[idx]).strip():
        return str(responses[idx]).strip(), "raw_drt_rollout"
    # Older runs only persisted drt_candidate_answers, not all raw rollout
    # strings. Keep the slot-level cache/evaluation shape, but mark this
    # clearly so these results are not mistaken for full-response judging.
    return "" if candidate is None else str(candidate).strip(), "saved_candidate_answer_only"


def build_prompt(row: dict[str, Any], candidate: str, slot_idx: int | str | None = None) -> str:
    problem = row.get("problem", "") or row.get("question", "")
    question = row.get("question", "")
    gt = row.get("reference_answer") or row.get("answer", "")
    gt_steps = row.get("reference_steps", []) or []
    candidate_text, candidate_source = candidate_response_for_prompt(row, candidate, slot_idx)
    extra = row.get("extra_info", {}) if isinstance(row.get("extra_info"), dict) else {}
    metadata = {
        "split_index": row.get("index", ""),
        "original_index": row.get("original_index", ""),
        "data_source": row.get("data_source", ""),
        "ability": row.get("ability", ""),
        "dataset": extra.get("dataset", ""),
        "hf_dataset": extra.get("hf_dataset", ""),
        "data_type": extra.get("data_type", ""),
        "step_generation_mode": extra.get("step_generation_mode", ""),
        "image_paths": row.get("image_paths", []),
        "candidate_slot_idx": slot_idx,
        "candidate_source": candidate_source,
    }
    steps_text = "\n".join(f"- {truncate_text(x, 1200)}" for x in gt_steps[:8])
    candidate_label = (
        "Candidate rollout response"
        if candidate_source == "raw_drt_rollout"
        else "Candidate answer string extracted from one or more rollout outputs"
    )
    return f"""You are a strict answer-equivalence judge for a visual reasoning/math benchmark.
Use the problem text, optional image(s), reference final answer, and reference solution steps/metadata to decide whether the candidate is correct.
Do NOT reward a candidate merely for plausible reasoning. Judge only whether its final answer is equivalent to the reference answer for this exact question.
Mark correct if the candidate is mathematically or semantically equivalent, including equivalent numbers, fractions, units, or the same multiple-choice option.
Mark incorrect if the candidate is different, ambiguous, empty, references the wrong object/quantity, or only contains reasoning without a final equivalent answer.
Return JSON only, exactly one object: {{"correct": true}} or {{"correct": false}}.

Problem:
{truncate_text(problem, 3000)}

Full benchmark prompt excerpt:
{truncate_text(question, 3000)}

Reference answer:
{gt}

Reference solution steps:
{truncate_text(steps_text, 6000) if steps_text else "(none provided)"}

Sample metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

{candidate_label}:
{truncate_text(candidate_text, 4000)}
"""


def build_user_content(
    args: argparse.Namespace,
    row: dict[str, Any],
    candidate: str,
    *,
    slot_idx: int | str | None = None,
    allow_images: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    prompt = build_prompt(row, candidate, slot_idx=slot_idx)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_count = 0
    if allow_images and args.image_mode == "auto":
        for path in (row.get("image_paths") or [])[: args.max_images]:
            url = image_to_data_url(str(path), args.max_image_mb)
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
            image_count += 1
    return content, image_count


def parse_correct(text: str) -> bool | None:
    if text is None:
        return None
    s = str(text).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
            return obj["correct"]
    except Exception:
        pass
    low = s.lower()
    if '"correct": true' in low or "correct: true" in low:
        return True
    if '"correct": false' in low or "correct: false" in low:
        return False
    if re.fullmatch(r"(?:true|correct|yes)\.?", low):
        return True
    if re.fullmatch(r"(?:false|incorrect|no)\.?", low):
        return False
    return None


def call_gpt(
    args: argparse.Namespace,
    api_key: str,
    row: dict[str, Any],
    candidate: str,
    key: str,
    slot_idx: int | str | None,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-TT-LOGID": f"dapo_gpt_judge_{args.split}_{row['index']}_{key[:10]}_{int(time.time())}",
    }
    last: dict[str, Any] = {}
    force_text_only = args.image_mode != "auto"
    for attempt in range(args.retry):
        content, image_count = build_user_content(args, row, candidate, slot_idx=slot_idx, allow_images=not force_text_only)
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "n": 1,
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        started = time.time()
        try:
            resp = requests.post(args.api_base, headers=headers, data=json.dumps(payload), timeout=args.timeout)
            elapsed = round(time.time() - started, 3)
            body = resp.json() if resp.text else {}
            content = ""
            choices = body.get("choices") or []
            if choices:
                content = ((choices[0].get("message") or {}).get("content") or "").strip()
            correct = parse_correct(content)
            usage = body.get("usage") or {}
            last = {
                "status_code": resp.status_code,
                "elapsed_sec": elapsed,
                "response_model": body.get("model"),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "content": content[:300],
                "correct": correct,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "image_count": image_count,
                "image_mode": "auto" if image_count else "text",
                "error": body.get("error"),
                "attempt": attempt + 1,
            }
            if resp.status_code == 200 and correct is not None:
                return last
            # Some internal OpenAI-compatible gateways may reject multimodal
            # content for a routed model. Fall back to the same metadata prompt
            # without image bytes instead of failing the judgement.
            if image_count and resp.status_code in {400, 404, 415, 422}:
                force_text_only = True
        except Exception as exc:
            last = {"exception": type(exc).__name__, "error": str(exc)[:300], "attempt": attempt + 1}
        time.sleep(min(2 ** attempt, 8))
    if last.get("correct") is None:
        last["correct"] = False
        last["failed_parse_or_call"] = True
    return last


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as rf:
        for line in rf:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("judgement_key")
            if key:
                cache[key] = obj
    return cache


def iter_tasks(args: argparse.Namespace, rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]]):
    if args.judge_unit == "slot":
        for row in rows:
            for slot_idx, cand in enumerate(row["candidate_answers"][:16]):
                key = judgement_key(args.split, row["index"], row["answer"], cand, args.judge_version, slot_idx)
                if key in cache:
                    continue
                yield key, row, cand, slot_idx
        return

    emitted = set()
    for row in rows:
        unique = []
        seen = set()
        for cand in row["candidate_answers"][:16]:
            raw = extract_answer(cand)
            if raw not in seen:
                seen.add(raw)
                unique.append(raw)
        # also judge selected answer if it is not among candidates for selected_acc robustness
            selected = extract_answer(row.get("selected_answer", ""))
        if selected and selected not in seen:
            unique.append(selected)
        for cand in unique:
            key = judgement_key(args.split, row["index"], row["answer"], cand, args.judge_version, None)
            if key in emitted or key in cache:
                continue
            emitted.add(key)
            yield key, row, cand, None


def write_record(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as wf:
        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        wf.flush()


def prelabel_exact(args: argparse.Namespace, rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]], out_jsonl: Path) -> int:
    if args.no_exact_shortcut:
        return 0
    added = 0
    for key, row, raw, slot_idx in iter_tasks(args, rows, {}):
        gt_norm = normalize_answer(row["answer"])
        if key in cache:
            continue
        cand_norm = normalize_answer(raw)
        if cand_norm and cand_norm == gt_norm:
            rec = {
                "judgement_key": key,
                "split": args.split,
                "index": row["index"],
                "answer": row["answer"],
                "candidate": raw,
                "slot_idx": slot_idx,
                "correct": True,
                "source": "exact_shortcut",
                "judge_version": args.judge_version,
                "judge_unit": args.judge_unit,
            }
            cache[key] = rec
            write_record(out_jsonl, rec)
            added += 1
    return added


def run_judging(args: argparse.Namespace, rows: list[dict[str, Any]], out_jsonl: Path) -> dict[str, dict[str, Any]]:
    api_key = resolve_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required (env or submit-job template).")
    args.api_base = args.api_base or resolve_api_base()
    cache = load_cache(out_jsonl)
    exact_added = prelabel_exact(args, rows, cache, out_jsonl)
    tasks = list(iter_tasks(args, rows, cache))
    if args.max_api_calls and len(tasks) > args.max_api_calls:
        tasks = tasks[: args.max_api_calls]
    print(
        json.dumps(
            {
                "split": args.split,
                "rows": len(rows),
                "cached_before_api": len(cache),
                "exact_shortcut_added": exact_added,
                "api_tasks": len(tasks),
                "api_base": args.api_base,
                "model": args.model,
                "judge_version": args.judge_version,
                "image_mode": args.image_mode,
                "out_jsonl": str(out_jsonl),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.nproc) as ex:
        fut_map = {
            ex.submit(call_gpt, args, api_key, row, cand, key, slot_idx): (key, row, cand, slot_idx)
            for key, row, cand, slot_idx in tasks
        }
        for fut in futures.as_completed(fut_map):
            key, row, cand, slot_idx = fut_map[fut]
            result = fut.result()
            rec = {
                "judgement_key": key,
                "split": args.split,
                "index": row["index"],
                "answer": row["answer"],
                "candidate": cand,
                "slot_idx": slot_idx,
                "judge_version": args.judge_version,
                "judge_unit": args.judge_unit,
                **result,
            }
            cache[key] = rec
            write_record(out_jsonl, rec)
            done += 1
            if done % args.progress_every == 0:
                ok = sum(1 for item in cache.values() if item.get("correct"))
                fail = sum(1 for item in cache.values() if item.get("failed_parse_or_call"))
                print(json.dumps({"done_api": done, "cache": len(cache), "correct_cached": ok, "failed": fail}), flush=True)
    return load_cache(out_jsonl)


def compute_summary(args: argparse.Namespace, rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def slot_key(row: dict[str, Any], cand: str, slot_idx: int | str | None) -> str:
        key_candidate = cand if args.judge_unit == "slot" else extract_answer(cand)
        return judgement_key(
            args.split,
            row["index"],
            row["answer"],
            key_candidate,
            args.judge_version,
            slot_idx if args.judge_unit == "slot" else None,
        )

    def judge(row: dict[str, Any], cand: str, slot_idx: int | str | None = None) -> bool:
        key = slot_key(row, cand, slot_idx)
        item = cache.get(key)
        return bool(item and item.get("correct"))

    passk = {}
    coverage = 0
    for row in rows:
        if args.judge_unit == "slot":
            needed = [slot_key(row, c, i) for i, c in enumerate(row["candidate_answers"][:16])]
        else:
            needed_values = {extract_answer(c) for c in row["candidate_answers"][:16]}
            if extract_answer(row.get("selected_answer", "")):
                needed_values.add(extract_answer(row.get("selected_answer", "")))
            needed = [judgement_key(args.split, row["index"], row["answer"], c, args.judge_version, None) for c in needed_values]
        if all(k in cache for k in needed):
            coverage += 1
    evaluable = []
    for row in rows:
        if args.judge_unit == "slot":
            needed = [slot_key(row, c, i) for i, c in enumerate(row["candidate_answers"][:16])]
        else:
            needed_values = {extract_answer(c) for c in row["candidate_answers"][:16]}
            needed = [judgement_key(args.split, row["index"], row["answer"], c, args.judge_version, None) for c in needed_values]
        if all(k in cache for k in needed):
            evaluable.append(row)
    for k in KS:
        passk[f"pass@{k}"] = (
            sum(any(judge(row, cand, i) for i, cand in enumerate(row["candidate_answers"][:k])) for row in evaluable) / len(evaluable)
            if evaluable else None
        )
    def selected_correct(row: dict[str, Any]) -> bool:
        try:
            idx = int(row.get("selected_index"))
        except Exception:
            idx = -1
        candidates = row["candidate_answers"][:16]
        if 0 <= idx < len(candidates):
            return judge(row, candidates[idx], idx)
        if args.judge_unit == "unique" and row.get("selected_answer", ""):
            return judge(row, row.get("selected_answer", ""), None)
        return False

    selected_acc = (
        sum(selected_correct(row) for row in evaluable) / len(evaluable)
        if evaluable else None
    )
    candidate_acc = (
        sum(
            sum(judge(row, cand, i) for i, cand in enumerate(row["candidate_answers"][:16])) / max(1, len(row["candidate_answers"][:16]))
            for row in evaluable
        ) / len(evaluable)
        if evaluable else None
    )
    avg_correct = (
        sum(sum(judge(row, cand, i) for i, cand in enumerate(row["candidate_answers"][:16])) for row in evaluable) / len(evaluable)
        if evaluable else None
    )
    failures = sum(1 for item in cache.values() if item.get("failed_parse_or_call"))
    sources = Counter(item.get("source", "api") for item in cache.values())
    response_models = Counter(item.get("response_model") for item in cache.values() if item.get("response_model"))
    return {
        "split": args.split,
        "rows_total": len(rows),
        "rows_evaluable": len(evaluable),
        "rows_full_judgement_coverage": coverage,
        "judgement_cache_size": len(cache),
        "failed_parse_or_call": failures,
        "source_counts": dict(sources),
        "response_model_counts": dict(response_models),
        "judge_version": args.judge_version,
        "judge_unit": args.judge_unit,
        "image_mode": args.image_mode,
        **passk,
        "selected_acc": selected_acc,
        "candidate_acc@16": candidate_acc,
        "avg_correct@16": avg_correct,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--input-pkl", help="DRT inference pkl for train partial/full")
    p.add_argument("--input-tsv", help="Prediction TSV for test/full")
    p.add_argument("--data-tsv", help="LMUData TSV with question/answer metadata")
    p.add_argument("--output-dir", default="submit-job/analysis_logs/dapo_gpt_judge_passk")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--retry", type=int, default=4)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--judge-version", default=JUDGE_VERSION)
    p.add_argument("--judge-unit", choices=["slot", "unique"], default="slot")
    p.add_argument("--image-mode", choices=["auto", "path-only", "off"], default="auto")
    p.add_argument("--max-images", type=int, default=4)
    p.add_argument("--max-image-mb", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-api-calls", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--no-exact-shortcut", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.input_pkl:
        rows = load_rows_from_pkl(Path(args.input_pkl), Path(args.data_tsv), args.limit)
        stem = Path(args.input_pkl).parent.name
    elif args.input_tsv:
        rows = load_rows_from_tsv(Path(args.input_tsv), args.limit)
        stem = Path(args.input_tsv).stem
    else:
        raise SystemExit("--input-pkl or --input-tsv required")
    suffix = f"_{args.limit}" if args.limit else ""
    shortcut_suffix = "noexact" if args.no_exact_shortcut else "exact"
    out_jsonl = out_dir / f"{args.split}_{stem}{suffix}_{args.model}_{args.judge_version}_{args.judge_unit}_{args.image_mode}_{shortcut_suffix}_judgements.jsonl"
    if args.summary_only:
        cache = load_cache(out_jsonl)
    else:
        cache = run_judging(args, rows, out_jsonl)
    summary = compute_summary(args, rows, cache)
    summary.update({"judgement_jsonl": str(out_jsonl), "model": args.model, "judge_version": args.judge_version})
    summary_path = out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
