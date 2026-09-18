"""Compare MathVista judge extraction before/after changing the judge model.

This script is read-only with respect to the original VLMEvalKit result
directory.  It loads an existing MathVista prediction file and its existing
judge pkl, re-runs only MathVista answer extraction on a sampled subset with a
new requested judge model, then writes comparison artifacts to a separate
output directory.

It is useful for validating endpoint-side model routing changes, for example:

    export OPENAI_API_KEY=...
    python3 submit-job/compare_mathvista_judge_diff.py \
      --prediction-file /path/to/*_MathVista_MINI.xlsx \
      --old-judge-pkl /path/to/*_MathVista_MINI_gpt-4.1-2025-04-14.pkl \
      --sample-size 1000
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from vlmeval.smp import load
from vlmeval.dataset.utils.mathvista import build_mathvista_gpt4_prompt, post_check


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_NEW_JUDGE = "gpt-4o-2024-05-13"
DEFAULT_OUTPUT_DIR = Path("submit-job/analysis_logs/mathvista_judge_diff_gpt4o0513")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MathVista old judge extraction with a new judge model.")
    parser.add_argument("--prediction-file", required=True, help="Existing MathVista prediction xlsx/tsv file.")
    parser.add_argument("--old-judge-pkl", required=True, help="Existing MathVista judge pkl to compare against.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--new-judge-model", default=DEFAULT_NEW_JUDGE)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def check_hit(line: dict[str, Any], res: Any) -> bool:
    obj = dict(line)
    obj["res"] = res
    return bool(post_check(obj, prefetch=False))


def call_chat_completion(args: argparse.Namespace, prompt: str, index: Any, temperature: float) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
        "X-TT-LOGID": f"vlmevalkit_mathvista_diff_{int(time.time())}_{index}_{temperature}",
    }
    payload = {
        "model": args.new_judge_model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "n": 1,
        "temperature": temperature,
        "max_tokens": args.max_tokens,
    }
    started = time.time()
    response = requests.post(args.api_base, headers=headers, data=json.dumps(payload), timeout=args.timeout)
    result = {
        "status_code": response.status_code,
        "elapsed_sec": round(time.time() - started, 3),
        "temperature": temperature,
    }
    try:
        body = response.json()
    except Exception:
        result["raw_body_prefix"] = response.text[:500]
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


def eval_one(args: argparse.Namespace, line: dict[str, Any], old_item: dict[str, Any]) -> dict[str, Any]:
    index = line["index"]
    old_res = old_item.get("res", "")
    old_log = old_item.get("log", "")
    old_hit = check_hit(line, old_res)

    prefetch_res = post_check(line, prefetch=True)
    api_attempts: list[dict[str, Any]] = []
    if prefetch_res:
        new_res = prefetch_res
        new_log = "Prefetch succeed"
    else:
        prompt = build_mathvista_gpt4_prompt(line)
        new_res = ""
        new_log_parts = []
        for i in range(args.retry):
            temperature = i * 0.5
            attempt = call_chat_completion(args, prompt, index, temperature)
            api_attempts.append(attempt)
            content = attempt.get("content", "")
            if attempt.get("status_code") == 200 and content:
                new_res = content
                new_log_parts.append("Succeed")
                break
            new_log_parts.append(
                "Try {i}: finish={finish} content={content!r} reasoning_tokens={reasoning}".format(
                    i=i,
                    finish=attempt.get("finish_reason"),
                    content=content,
                    reasoning=attempt.get("reasoning_tokens"),
                )
            )
        if not new_res:
            new_log_parts.append("All retries failed.")
        new_log = "\n".join(new_log_parts)

    new_hit = check_hit(line, new_res)
    return {
        "index": int(index) if str(index).isdigit() else index,
        "question": as_str(line.get("question", ""))[:300],
        "prediction": as_str(line.get("prediction", ""))[:500],
        "answer": as_str(line.get("answer", "")),
        "answer_type": as_str(line.get("answer_type", "")),
        "question_type": as_str(line.get("question_type", "")),
        "old_res": old_res,
        "old_log": old_log,
        "old_hit": old_hit,
        "new_res": new_res,
        "new_log": new_log,
        "new_hit": new_hit,
        "res_changed": as_str(old_res) != as_str(new_res),
        "hit_changed": old_hit != new_hit,
        "used_api": not bool(prefetch_res),
        "api_attempts": api_attempts,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    attempt_rows = [attempt for row in rows for attempt in row["api_attempts"]]
    reasoning = [
        int(attempt["reasoning_tokens"])
        for attempt in attempt_rows
        if attempt.get("reasoning_tokens") is not None
    ]
    response_models = Counter(attempt.get("response_model") for attempt in attempt_rows if attempt.get("response_model"))
    finish_reasons = Counter(attempt.get("finish_reason") for attempt in attempt_rows if attempt.get("finish_reason"))
    empty_length_128 = sum(
        1
        for attempt in attempt_rows
        if attempt.get("finish_reason") == "length"
        and not attempt.get("content")
        and attempt.get("reasoning_tokens") == args.max_tokens
    )
    return {
        "prediction_file": args.prediction_file,
        "old_judge_pkl": args.old_judge_pkl,
        "api_base": args.api_base,
        "new_requested_judge_model": args.new_judge_model,
        "sample_size": len(rows),
        "used_api_count": sum(row["used_api"] for row in rows),
        "prefetch_count": sum(not row["used_api"] for row in rows),
        "old_failed_count": sum(as_str(row["old_res"]) == "" for row in rows),
        "new_failed_count": sum(as_str(row["new_res"]) == "" for row in rows),
        "res_changed_count": sum(row["res_changed"] for row in rows),
        "hit_changed_count": sum(row["hit_changed"] for row in rows),
        "old_acc": sum(row["old_hit"] for row in rows) / len(rows) if rows else None,
        "new_acc": sum(row["new_hit"] for row in rows) / len(rows) if rows else None,
        "api_attempt_count": len(attempt_rows),
        "response_model_counts": dict(response_models),
        "finish_reason_counts": dict(finish_reasons),
        "empty_length_all_reasoning_count": empty_length_128,
        "reasoning_tokens_min": min(reasoning) if reasoning else None,
        "reasoning_tokens_max": max(reasoning) if reasoning else None,
        "reasoning_tokens_mean": statistics.mean(reasoning) if reasoning else None,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required. Set it in the environment or pass --api-key.")

    data = load(args.prediction_file)
    old = load(args.old_judge_pkl)
    records = [row.to_dict() for _, row in data.iterrows() if row["index"] in old]
    rng = random.Random(args.seed)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)
    records = sorted(records, key=lambda item: item["index"])

    with futures.ThreadPoolExecutor(max_workers=args.nproc) as executor:
        futs = [executor.submit(eval_one, args, line, old[line["index"]]) for line in records]
        rows = [fut.result() for fut in futures.as_completed(futs)]
    rows = sorted(rows, key=lambda item: item["index"])
    summary = summarize(rows, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jsonl_path = out_dir / f"mathvista_judge_diff_{stamp}.jsonl"
    summary_path = out_dir / f"mathvista_judge_diff_{stamp}_summary.json"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSONL: {jsonl_path}")
    print(f"Summary: {summary_path}")
    print("\nExamples where hit changed:")
    for row in [row for row in rows if row["hit_changed"]][:10]:
        print(
            json.dumps(
                {
                    "index": row["index"],
                    "prediction": row["prediction"][:160],
                    "answer": row["answer"],
                    "old_res": row["old_res"],
                    "old_hit": row["old_hit"],
                    "new_res": row["new_res"],
                    "new_hit": row["new_hit"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
