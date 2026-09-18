#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOW_BENCH_DIR = REPO_ROOT / "show-result" / "benchmark-summary"
if str(SHOW_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(SHOW_BENCH_DIR))

import show_mathbench_result as show_bench

from vlmeval.dataset import build_dataset
from vlmeval.smp.file import get_intermediate_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate all Video-Holmes result files discovered by show_mathbench_result.py and monitor score diffs."
    )
    parser.add_argument("--judge-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--api-nproc", type=int, default=4)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of files to process.")
    parser.add_argument(
        "--report-path",
        default="video_holmes_reeval_diff_report.json",
        help="Where to save the diff report JSON.",
    )
    return parser.parse_args()


def find_video_holmes_benchmark():
    for benchmark in show_bench.BENCHMARKS:
        if benchmark["name"] == "VideoHolmes":
            return benchmark
    raise RuntimeError("VideoHolmes benchmark definition not found in show_mathbench_result.py")


def detect_dataset_name(raw_path: str, dataset_names: list[str]) -> str:
    filename = Path(raw_path).name
    for dataset_name in sorted(dataset_names, key=len, reverse=True):
        if f"_{dataset_name}." in filename:
            return dataset_name
    return dataset_names[0]


def choose_existing_score_path(raw_path: str) -> str | None:
    def _resolve_artifact(path: str) -> str | None:
        artifact = Path(path)
        if artifact.exists():
            return str(artifact)
        basename = artifact.name
        parent = artifact.parent
        candidates = sorted(
            path for path in parent.glob(f"T*/{basename}")
            if path.exists()
        )
        if candidates:
            return str(candidates[-1])
        return None

    rating_path = _resolve_artifact(get_intermediate_file_path(raw_path, "_rating", "json"))
    score_path = _resolve_artifact(get_intermediate_file_path(raw_path, "_score"))
    if rating_path is not None:
        return rating_path
    if score_path is not None:
        return score_path
    return None


def ensure_openai_env(args: argparse.Namespace) -> None:
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base


def find_fallback_raw_path(folder_path: str, raw_path: str) -> str | None:
    basename = Path(raw_path).name
    parent = Path(folder_path)
    candidates = sorted(parent.glob(f"T*/{basename}"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    existing.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return str(existing[-1])


def ensure_raw_path_available(job: dict) -> dict:
    raw_path = Path(job["raw_path"])
    if raw_path.exists():
        job["raw_path_relinked_from"] = None
        return job

    fallback = find_fallback_raw_path(job["folder_path"], job["raw_path"])
    if fallback is None:
        raise FileNotFoundError(job["raw_path"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.is_symlink() or raw_path.exists():
        raw_path.unlink()
    raw_path.symlink_to(fallback)
    job["raw_path_relinked_from"] = fallback
    return job


def backup_and_remove(path: str, backup_suffix: str) -> str | None:
    src = Path(path)
    if not src.exists() and not src.is_symlink():
        return None

    if src.is_symlink() and not src.exists():
        backup_path = src.with_name(src.name + backup_suffix + ".broken_symlink")
        backup_path.write_text(os.readlink(src), encoding="utf-8")
        src.unlink()
        return str(backup_path)

    backup_path = src.with_name(src.name + backup_suffix)
    shutil.copy2(src, backup_path)
    src.unlink()
    return str(backup_path)


def collect_jobs(limit: int | None) -> list[dict]:
    benchmark = find_video_holmes_benchmark()
    jobs = []
    for folder_path, files in show_bench.collect_candidate_folders():
        model_name = Path(folder_path).name
        raw_path = show_bench._find_raw_prediction_file(  # noqa: SLF001
            folder_path,
            files,
            model_name,
            benchmark["dataset_names"],
        )
        if raw_path is None:
            continue
        dataset_name = detect_dataset_name(raw_path, benchmark["dataset_names"])
        jobs.append(
            {
                "folder_path": folder_path,
                "model_name": model_name,
                "raw_path": raw_path,
                "dataset_name": dataset_name,
            }
        )

    jobs = sorted(jobs, key=lambda item: (item["folder_path"], item["raw_path"]))
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def reevaluate_job(job: dict, judge_model: str, api_nproc: int, backup_suffix: str) -> dict:
    job = ensure_raw_path_available(dict(job))
    raw_path = job["raw_path"]
    dataset_name = job["dataset_name"]
    old_score_path = choose_existing_score_path(raw_path)
    old_score = show_bench.parse_video_holmes_score(old_score_path) if old_score_path else "-"

    rating_path = get_intermediate_file_path(raw_path, "_rating", "json")
    score_path = get_intermediate_file_path(raw_path, "_score")

    backups = {
        "rating": backup_and_remove(rating_path, backup_suffix),
        "score": backup_and_remove(score_path, backup_suffix),
    }

    start = time.time()
    dataset = build_dataset(dataset_name)
    rating = dataset.evaluate(raw_path, model=judge_model, nproc=api_nproc)
    elapsed_sec = time.time() - start

    new_score_path = choose_existing_score_path(raw_path)
    new_score = show_bench.parse_video_holmes_score(new_score_path) if new_score_path else "-"

    rating_total = None
    if isinstance(rating, dict):
        total = rating.get("total")
        if isinstance(total, dict):
            rating_total = total

    return {
        **job,
        "old_score_path": old_score_path,
        "new_score_path": new_score_path,
        "old_score": old_score,
        "new_score": new_score,
        "elapsed_sec": round(elapsed_sec, 3),
        "backups": backups,
        "rating_total": rating_total,
    }


def save_report(report: dict, report_path: Path) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    ensure_openai_env(args)
    jobs = collect_jobs(args.limit)
    total = len(jobs)
    print(f"Discovered {total} Video-Holmes result files from show_mathbench_result.py scope.")
    if total == 0:
        return

    backup_suffix = f".bak_videoholmes_reeval_{time.strftime('%Y%m%d_%H%M%S')}"
    report = {
        "judge_model": args.judge_model,
        "api_nproc": args.api_nproc,
        "backup_suffix": backup_suffix,
        "jobs": [],
    }
    report_path = Path(args.report_path)

    changed = 0
    unchanged = 0
    failed = 0

    for idx, job in enumerate(jobs, 1):
        label = f"[{idx}/{total}] {job['model_name']} :: {Path(job['raw_path']).name}"
        try:
            result = reevaluate_job(job, args.judge_model, args.api_nproc, backup_suffix)
            old_score = result["old_score"]
            new_score = result["new_score"]

            delta = None
            if old_score != "-" and new_score != "-":
                delta = round(float(new_score) - float(old_score), 4)

            if delta is None or abs(delta) < 1e-9:
                unchanged += 1
                status = "UNCHANGED"
            else:
                changed += 1
                status = "CHANGED"

            result["delta"] = delta
            result["status"] = status
            report["jobs"].append(result)

            extra = ""
            if result["rating_total"] is not None:
                total_info = result["rating_total"]
                extra = (
                    f" total={total_info.get('correct')}/{total_info.get('total')}"
                    f" acc={round(float(total_info.get('acc', 0)) * 100, 4)}"
                )
            print(
                f"{label} -> {status}: old={old_score} new={new_score}"
                + (f" delta={delta:+.4f}" if delta is not None else "")
                + f" elapsed={result['elapsed_sec']:.3f}s{extra}"
            , flush=True)
            save_report(report, report_path)
        except Exception as exc:
            failed += 1
            report["jobs"].append(
                {
                    **job,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )
            print(f"{label} -> FAILED: {exc}", flush=True)
            save_report(report, report_path)

    report["summary"] = {
        "total_jobs": total,
        "changed": changed,
        "unchanged": unchanged,
        "failed": failed,
    }

    save_report(report, report_path)
    print(f"Saved diff report to {report_path}")
    print(
        f"Summary: total={total}, changed={changed}, unchanged={unchanged}, failed={failed}"
    )


if __name__ == "__main__":
    main()
