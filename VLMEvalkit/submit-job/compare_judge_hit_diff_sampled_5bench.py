"""Sampled hit-consistency check across the five paper benchmarks.

This script is intentionally read-only with respect to existing VLMEvalKit
result directories.  It loads old prediction/evaluation artifacts, re-runs only
the GPT-judge-dependent parsing/scoring parts with a new requested judge model,
and writes all comparison artifacts under ``submit-job/analysis_logs``.

Typical use:

    export OPENAI_API_KEY=...
    python3 submit-job/compare_judge_hit_diff_sampled_5bench.py \\
      --combo label=/path/to/T20260421_Gxxxx \\
      --sample-size 35 --nproc 8

The default new judge request is ``gpt-4o-2024-05-13`` on the i18n endpoint;
on 2026-05-08 that endpoint routes it to ``gpt-4.1-mini-2025-04-14``.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from vlmeval.smp import load
from vlmeval.dataset.utils.logicvista import build_prompt_logicvista
from vlmeval.dataset.utils.mathverse import (
    build_mathverse_gpt4_extract_prompt,
    build_mathverse_gpt4_score_prompt,
    parse_mathverse_judgement,
    post_check_score as mathverse_post_check_score,
)
from vlmeval.dataset.utils.mathvista import build_mathvista_gpt4_prompt, post_check as mathvista_post_check
from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item
from vlmeval.dataset.utils.videoholmes import extract_option as videoholmes_extract_option


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_NEW_JUDGE = "gpt-4o-2024-05-13"
DEFAULT_OUTPUT_DIR = Path("submit-job/analysis_logs/judge_hit_diff_sample35_5bench")
TASKS = ["MathVista_MINI", "MathVerse_MINI", "LogicVista", "GSM8K", "Video_Holmes"]
OLD_JUDGE = "gpt-4.1-2025-04-14"
FAIL_MSG = "Failed to obtain answer via API."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sampled old hits with re-judged hits across MathVista/MathVerse/LogicVista/GSM8K/Video_Holmes."
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="A combo in label=/path/to/T*_G* form. Can be repeated.",
    )
    parser.add_argument(
        "--combo-file",
        default=None,
        help="Optional text file with one label=/path/to/T*_G* per line; blank lines and # comments are ignored.",
    )
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--new-judge-model", default=DEFAULT_NEW_JUDGE)
    parser.add_argument("--sample-size", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def norm_index(value: Any) -> Any:
    try:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\d+", value):
            return int(value)
    except Exception:
        pass
    return value


def get_by_index(mapping: dict[Any, Any], index: Any, default: Any = None) -> Any:
    keys = [index, norm_index(index), str(index)]
    try:
        keys.append(float(index))
    except Exception:
        pass
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


class JudgeClient:
    """Tiny chat-completions client compatible with VLMEvalKit MCQ helpers."""

    def __init__(self, args: argparse.Namespace):
        self.api_base = args.api_base
        self.api_key = args.api_key
        self.model = args.new_judge_model
        self.max_tokens = args.max_tokens
        self.timeout = args.timeout

    def chat(self, prompt: str, *, tag: str, temperature: float = 0.0) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-TT-LOGID": f"vlmevalkit_diff_{int(time.time())}_{tag}_{temperature}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "n": 1,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        started = time.time()
        result: dict[str, Any] = {
            "requested_model": self.model,
            "temperature": temperature,
        }
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            result["status_code"] = response.status_code
            result["elapsed_sec"] = round(time.time() - started, 3)
            try:
                body = response.json()
            except Exception:
                result["raw_body_prefix"] = response.text[:500]
                return result
            if not isinstance(body, dict):
                result["raw_json_type"] = type(body).__name__
                result["raw_json_prefix"] = repr(body)[:500]
                return result
        except Exception as exc:
            result["exception"] = repr(exc)
            result["elapsed_sec"] = round(time.time() - started, 3)
            return result

        result["response_model"] = body.get("model")
        result["error"] = body.get("error")
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        result["prompt_tokens"] = usage.get("prompt_tokens")
        result["completion_tokens"] = usage.get("completion_tokens")
        result["reasoning_tokens"] = details.get("reasoning_tokens")
        choices = body.get("choices") or []
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            result["finish_reason"] = choice.get("finish_reason")
            result["content"] = message.get("content") or ""
        return result

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Compatibility shim for ``extract_answer_from_item``.
        attempt = self.chat(prompt, tag="mcq", temperature=temperature)
        content = as_str(attempt.get("content", ""))
        return content if content else FAIL_MSG


def generate_with_retry(
    judge: JudgeClient,
    prompt: str,
    *,
    tag: str,
    retry: int,
    validator,
) -> tuple[str, str, list[dict[str, Any]]]:
    attempts = []
    logs = []
    for i in range(retry):
        temperature = i * 0.5
        attempt = judge.chat(prompt, tag=tag, temperature=temperature)
        attempts.append(attempt)
        content = as_str(attempt.get("content", ""))
        ok, normalized = validator(content)
        if attempt.get("status_code") == 200 and ok:
            logs.append("Succeed")
            return normalized, "\n".join(logs), attempts
        logs.append(
            "Try {i}: status={status} finish={finish} content={content!r} reasoning_tokens={reasoning}".format(
                i=i,
                status=attempt.get("status_code"),
                finish=attempt.get("finish_reason"),
                content=content,
                reasoning=attempt.get("reasoning_tokens"),
            )
        )
    logs.append(f"All {retry} retries failed.")
    return "", "\n".join(logs), attempts


def valid_nonempty(content: str) -> tuple[bool, str]:
    return bool(content and FAIL_MSG not in content), content


def valid_logicvista(content: str) -> tuple[bool, str]:
    content = as_str(content)
    return bool(content and content.isupper() and content.isalpha()), content


def valid_mathverse_score(content: str) -> tuple[bool, str]:
    parsed = parse_mathverse_judgement(content)
    return parsed is not None, "" if parsed is None else str(parsed)


def logicvista_hit(line: dict[str, Any], res: Any) -> bool:
    answer = [x.lower() for x in as_str(line.get("answer", "")).split(", ") if x]
    answer.sort()
    extracted = [alpha.lower() for alpha in as_str(res)]
    extracted.sort()
    return "".join(extracted) == "".join(answer)


def find_one(tdir: Path, patterns: Iterable[str], required: bool = True) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(tdir.glob(pattern)))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find any of {list(patterns)} under {tdir}")
    return None


def load_task_context(tdir: Path, task: str) -> dict[str, Any]:
    pred = find_one(tdir, [f"*_{task}.tsv", f"*_{task}.xlsx"])
    ctx: dict[str, Any] = {"prediction_file": str(pred), "data": load(str(pred))}
    if task == "MathVista_MINI":
        pkl = find_one(tdir, [f"*_{task}_{OLD_JUDGE}.pkl"])
        ctx["old_pkl"] = str(pkl)
        ctx["old"] = load(str(pkl))
    elif task == "MathVerse_MINI":
        extract_pkl = find_one(tdir, [f"*_{task}_{OLD_JUDGE}_extract.pkl"])
        score_pkl = find_one(tdir, [f"*_{task}_{OLD_JUDGE}_score.pkl"])
        ctx["old_extract_pkl"] = str(extract_pkl)
        ctx["old_score_pkl"] = str(score_pkl)
        ctx["old_extract"] = load(str(extract_pkl))
        ctx["old_score"] = load(str(score_pkl))
    elif task == "LogicVista":
        # VLMEvalKit names this artifact gpt4.1 rather than the full model id.
        pkl = find_one(tdir, [f"*_{task}_gpt4.1.pkl", f"*_{task}_{OLD_JUDGE}.pkl"])
        ctx["old_pkl"] = str(pkl)
        ctx["old"] = load(str(pkl))
    elif task == "Video_Holmes":
        score_file = find_one(tdir, [f"*_{task}_score.tsv", f"*_{task}_score.xlsx"])
        ctx["old_score_file"] = str(score_file)
        ctx["old_score"] = load(str(score_file))
    elif task == "GSM8K":
        pass
    else:
        raise ValueError(f"Unsupported task: {task}")
    return ctx


def available_indices(ctx: dict[str, Any], task: str) -> list[Any]:
    data = ctx["data"]
    if task == "MathVista_MINI":
        old = ctx["old"]
        return [row["index"] for _, row in data.iterrows() if get_by_index(old, row["index"]) is not None]
    if task == "MathVerse_MINI":
        old_score = ctx["old_score"]
        return [row["index"] for _, row in data.iterrows() if get_by_index(old_score, row["index"]) is not None]
    if task == "LogicVista":
        old = ctx["old"]
        return [row["index"] for _, row in data.iterrows() if get_by_index(old, row["index"]) is not None]
    if task == "Video_Holmes":
        return list(ctx["data"]["index"])
    if task == "GSM8K":
        return list(ctx["data"]["index"])
    raise ValueError(task)


def sample_records(ctx: dict[str, Any], task: str, label: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    indices = available_indices(ctx, task)
    rng = random.Random(seed + stable_int(f"{label}:{task}"))
    if sample_size and len(indices) > sample_size:
        selected = set(rng.sample(list(indices), sample_size))
    else:
        selected = set(indices)
    records = [row.to_dict() for _, row in ctx["data"].iterrows() if row["index"] in selected]
    records.sort(key=lambda row: str(row["index"]))
    return records


def eval_mathvista(args: argparse.Namespace, judge: JudgeClient, line: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_item = get_by_index(ctx["old"], index, {})
    old_res = old_item.get("res", "")
    old_hit = bool(mathvista_post_check({**line, "res": old_res}, prefetch=False))

    prefetch_res = mathvista_post_check(line, prefetch=True)
    attempts: list[dict[str, Any]] = []
    if prefetch_res:
        new_res = prefetch_res
        new_log = "Prefetch succeed"
        used_api = False
    else:
        new_res, new_log, attempts = generate_with_retry(
            judge,
            build_mathvista_gpt4_prompt(line),
            tag=f"mathvista_{index}",
            retry=args.retry,
            validator=valid_nonempty,
        )
        used_api = True
    new_hit = bool(mathvista_post_check({**line, "res": new_res}, prefetch=False))
    return base_row("MathVista_MINI", index, line, old_res, old_hit, new_res, new_hit, used_api, new_log, attempts)


def eval_mathverse(args: argparse.Namespace, judge: JudgeClient, line: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_extract_item = get_by_index(ctx["old_extract"], index, {})
    old_score_item = get_by_index(ctx["old_score"], index, {})
    old_res = old_extract_item.get("extract", "")
    old_hit = bool(old_score_item.get("score", False))

    extract, extract_log, extract_attempts = generate_with_retry(
        judge,
        build_mathverse_gpt4_extract_prompt(line),
        tag=f"mathverse_extract_{index}",
        retry=args.retry,
        validator=valid_nonempty,
    )
    line_with_extract = {**line, "extract": extract}
    prefetch_score = mathverse_post_check_score(line_with_extract, prefetch=True)
    attempts = list(extract_attempts)
    if prefetch_score:
        new_hit = True
        score_res = "1"
        score_log = "Prefetch succeed"
    else:
        score_res, score_log, score_attempts = generate_with_retry(
            judge,
            build_mathverse_gpt4_score_prompt(line_with_extract),
            tag=f"mathverse_score_{index}",
            retry=args.retry,
            validator=valid_mathverse_score,
        )
        attempts.extend(score_attempts)
        new_hit = score_res == "1"
    new_res = f"extract={extract}; score={score_res}"
    new_log = f"extract: {extract_log}\nscore: {score_log}"
    return base_row("MathVerse_MINI", index, line, old_res, old_hit, new_res, new_hit, True, new_log, attempts)


def eval_logicvista(args: argparse.Namespace, judge: JudgeClient, line: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_item = get_by_index(ctx["old"], index, {})
    old_res = old_item.get("res", "")
    old_hit = bool(old_item.get("hit", False))

    new_res, new_log, attempts = generate_with_retry(
        judge,
        build_prompt_logicvista(line),
        tag=f"logicvista_{index}",
        retry=args.retry,
        validator=valid_logicvista,
    )
    new_hit = logicvista_hit(line, new_res)
    return base_row("LogicVista", index, line, old_res, old_hit, new_res, new_hit, True, new_log, attempts)


def parse_video_candidates(line: dict[str, Any]) -> dict[str, str]:
    choices: dict[str, str] = {}
    try:
        candidates = ast.literal_eval(as_str(line.get("candidates", "")))
    except Exception:
        candidates = []
    for idx, option in enumerate(candidates):
        label = chr(ord("A") + idx)
        text = as_str(option)
        prefix = f"{label}."
        if prefix in text:
            text = text[text.find(prefix) + len(prefix) :].strip(". \n")
        choices[label] = text
    return choices


def eval_videoholmes(args: argparse.Namespace, judge: JudgeClient, line: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_score_df = ctx["old_score"]
    old_rows = old_score_df[old_score_df["index"] == index]
    old_hit = boolish(old_rows.iloc[0]["score"]) if len(old_rows) else False

    prediction = as_str(line.get("prediction", ""))
    local_res = videoholmes_extract_option(prediction, line)
    attempts: list[dict[str, Any]] = []
    used_api = False
    new_log = "Local extraction succeed"
    if local_res == "WRONG":
        # Mirror Video_Holmes.evaluate: fall back to GPT-based MCQ matching.
        item = dict(line)
        item.update(parse_video_candidates(line))
        before = time.time()
        new_res = extract_answer_from_item(judge, item, "Video-Holmes")["opt"]
        # ``extract_answer_from_item`` calls JudgeClient.generate; for compactness
        # we do not retain per-attempt metadata here.  It is still API-backed.
        new_log = f"GPT MCQ extraction used; elapsed_sec={round(time.time() - before, 3)}"
        used_api = True
    else:
        new_res = local_res
    new_hit = as_str(new_res) == as_str(line.get("answer", ""))
    return base_row("Video_Holmes", index, line, "", old_hit, new_res, new_hit, used_api, new_log, attempts)


def eval_gsm8k(_: argparse.Namespace, __: JudgeClient, line: dict[str, Any], ___: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_hit = boolish(line.get("hit", False))
    # GSM8K is locally/exactly scored in these artifacts, so there is no new GPT judge to run.
    return base_row("GSM8K", index, line, "", old_hit, "", old_hit, False, "No GPT judge for GSM8K; reused existing exact hit.", [])


def base_row(
    task: str,
    index: Any,
    line: dict[str, Any],
    old_res: Any,
    old_hit: bool,
    new_res: Any,
    new_hit: bool,
    used_api: bool,
    new_log: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": task,
        "index": norm_index(index),
        "question": as_str(line.get("question", line.get("question_for_eval", "")))[:300],
        "prediction": as_str(line.get("prediction", ""))[:500],
        "answer": as_str(line.get("answer", "")),
        "old_res": as_str(old_res)[:500],
        "old_hit": bool(old_hit),
        "new_res": as_str(new_res)[:500],
        "new_hit": bool(new_hit),
        "hit_changed": bool(old_hit) != bool(new_hit),
        "used_api": bool(used_api),
        "new_log": new_log,
        "api_attempts": attempts,
    }


EVAL_FN = {
    "MathVista_MINI": eval_mathvista,
    "MathVerse_MINI": eval_mathverse,
    "LogicVista": eval_logicvista,
    "GSM8K": eval_gsm8k,
    "Video_Holmes": eval_videoholmes,
}


def parse_combos(args: argparse.Namespace) -> list[tuple[str, Path]]:
    entries = list(args.combo)
    if args.combo_file:
        for line in Path(args.combo_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    combos = []
    for entry in entries:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            path = entry
            label = Path(path).parent.name
        combos.append((label.strip(), Path(path).expanduser()))
    return combos


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [attempt for row in rows for attempt in row.get("api_attempts", [])]
    reasoning = [int(x["reasoning_tokens"]) for x in attempts if x.get("reasoning_tokens") is not None]
    return {
        "sample_size": len(rows),
        "old_hit_count": int(sum(row["old_hit"] for row in rows)),
        "new_hit_count": int(sum(row["new_hit"] for row in rows)),
        "old_acc": round(sum(row["old_hit"] for row in rows) / len(rows), 6) if rows else None,
        "new_acc": round(sum(row["new_hit"] for row in rows) / len(rows), 6) if rows else None,
        "hit_changed_count": int(sum(row["hit_changed"] for row in rows)),
        "used_api_count": int(sum(row["used_api"] for row in rows)),
        "api_attempt_count": len(attempts),
        "response_model_counts": dict(Counter(a.get("response_model") for a in attempts if a.get("response_model"))),
        "finish_reason_counts": dict(Counter(a.get("finish_reason") for a in attempts if a.get("finish_reason"))),
        "reasoning_tokens_min": min(reasoning) if reasoning else None,
        "reasoning_tokens_max": max(reasoning) if reasoning else None,
        "reasoning_tokens_mean": round(statistics.mean(reasoning), 3) if reasoning else None,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required. Set it in the environment or pass --api-key.")
    combos = parse_combos(args)
    if not combos:
        raise SystemExit("No --combo or --combo-file was provided.")

    judge = JudgeClient(args)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []

    for combo_idx, (label, tdir) in enumerate(combos, start=1):
        if not tdir.is_dir():
            raise FileNotFoundError(tdir)
        print(f"\n[{combo_idx}/{len(combos)}] {label}: {tdir}")
        for task in TASKS:
            print(f"  - {task}: loading/sample/eval", flush=True)
            try:
                ctx = load_task_context(tdir, task)
                records = sample_records(ctx, task, label, args.sample_size, args.seed)
                with futures.ThreadPoolExecutor(max_workers=args.nproc) as executor:
                    futs = [
                        executor.submit(EVAL_FN[task], args, judge, record, ctx)
                        for record in records
                    ]
                    rows = [fut.result() for fut in futures.as_completed(futs)]
                rows.sort(key=lambda row: str(row["index"]))
                for row in rows:
                    row["combo_label"] = label
                    row["combo_dir"] = str(tdir)
                summary = summarize_rows(rows)
                summary.update(
                    {
                        "combo_label": label,
                        "combo_dir": str(tdir),
                        "task": task,
                        "prediction_file": ctx.get("prediction_file"),
                        "api_base": args.api_base,
                        "new_requested_judge_model": args.new_judge_model,
                    }
                )
                all_summaries.append(summary)
                changed_rows.extend(row for row in rows if row["hit_changed"])
                task_jsonl = out_dir / f"{safe_name(label)}__{task}.jsonl"
                with task_jsonl.open("w") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    "    sample={sample_size} old={old_hit_count} new={new_hit_count} diff={hit_changed_count} api={used_api_count}".format(
                        **summary
                    ),
                    flush=True,
                )
            except Exception as exc:
                error_summary = {
                    "combo_label": label,
                    "combo_dir": str(tdir),
                    "task": task,
                    "error": repr(exc),
                    "sample_size": 0,
                    "hit_changed_count": None,
                }
                all_summaries.append(error_summary)
                print(f"    ERROR: {exc!r}", flush=True)

    summary_path = out_dir / "summary.json"
    changed_path = out_dir / "hit_changed_examples.jsonl"
    csv_path = out_dir / "summary.csv"
    summary_doc = {
        "created_utc": stamp,
        "api_base": args.api_base,
        "new_requested_judge_model": args.new_judge_model,
        "sample_size_per_combo_task": args.sample_size,
        "seed": args.seed,
        "combos": [{"label": label, "dir": str(path)} for label, path in combos],
        "summaries": all_summaries,
        "total_hit_changed_count": int(
            sum(x.get("hit_changed_count") or 0 for x in all_summaries if isinstance(x.get("hit_changed_count"), int))
        ),
    }
    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2))
    with changed_path.open("w") as f:
        for row in changed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(all_summaries).to_csv(csv_path, index=False)

    print("\nDONE")
    print(f"Summary: {summary_path}")
    print(f"CSV: {csv_path}")
    print(f"Hit-changed examples: {changed_path}")
    print(
        json.dumps(
            {
                "combo_task_count": len(all_summaries),
                "errored_combo_task_count": sum(1 for x in all_summaries if x.get("error")),
                "total_hit_changed_count": summary_doc["total_hit_changed_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:160] or "combo"


if __name__ == "__main__":
    main()
