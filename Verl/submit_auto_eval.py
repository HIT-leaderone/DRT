#!/usr/bin/env python3
"""Monitor HDFS checkpoints and submit VLMEvalKit auto-eval jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_NAME = "Vision-R1-800it-lr5e-6"
HDFS_ROOT = os.environ.get("AUTO_EVAL_HDFS_ROOT", "")
PROMPT_TYPE = "Short-COT-Image"
POOL = "public"
DEFAULT_TASKS = ("MathVista_MINI", "MathVerse_MINI", "LogicVista", "GSM8K", "MATH-500")
DEFAULT_RETRY = 10
DEFAULT_POLL_INTERVAL = 300
HADOOP_CONF_DIR_OVERRIDE_ENV = "AUTO_EVAL_HADOOP_CONF_DIR"
DEFAULT_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_OPENAI_AUTH_SCHEME = "bearer"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-05-13"

REQUIRED_HF_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "tokenizer.json",
)


@dataclass(frozen=True)
class Experiment:
    exp_name: str
    model_type: str

    def hdfs_dir(self, hdfs_root: str, project_name: str) -> str:
        return f"{hdfs_root.rstrip('/')}/{project_name}/{self.exp_name}"


EXPERIMENTS = (
    Experiment(
        exp_name="qwen3-235B-reward-ray-train-autoeval-step-bonus-mmfinereason-h100-official",
        model_type="Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-mmfinereason-official",
    ),
    Experiment(
        exp_name="qwen3-235B-reward-ray-train-autoeval-step-bonus-visionr1-mmfinereason-visiononly",
        model_type="Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-mmfinereason",
    ),
    Experiment(
        exp_name="qwen3-235B-reward-ray-train-autoeval-step-bonus-mmfinereason-h100",
        model_type="Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-mmfinereason",
    ),
)


def parse_tasks(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]


def parse_steps(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    steps = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        steps.add(int(item))
    return steps


def run_hdfs(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    hadoop_conf_dir = env.get(HADOOP_CONF_DIR_OVERRIDE_ENV)
    if hadoop_conf_dir:
        env["HADOOP_CONF_DIR"] = hadoop_conf_dir
    return subprocess.run(["hdfs", "dfs", *args], check=False, capture_output=True, text=True, env=env)


def list_global_steps(hdfs_dir: str) -> list[int]:
    result = run_hdfs(["-ls", hdfs_dir])
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        print(f"[warn] hdfs ls failed: {hdfs_dir}\n{stderr}", file=sys.stderr)
        return []

    steps = []
    for line in result.stdout.splitlines():
        match = re.search(r"/global_step_(\d+)(?:\s*$|/)", line)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def checkpoint_hf_path(exp: Experiment, step: int, args: argparse.Namespace) -> str:
    return f"{exp.hdfs_dir(args.hdfs_root, args.project_name)}/global_step_{step}/actor/huggingface"


def is_checkpoint_ready(hf_path: str, required_files: tuple[str, ...]) -> bool:
    result = run_hdfs(["-ls", hf_path])
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        print(f"[skip] checkpoint not ready: {hf_path}\n{stderr}")
        return False
    files = {line.rsplit("/", 1)[-1] for line in result.stdout.splitlines() if "/" in line}
    missing = [name for name in required_files if name not in files]
    if missing:
        print(f"[skip] checkpoint not ready: {hf_path}; missing: {', '.join(missing)}")
        return False
    return True


def default_state_path() -> Path:
    raw = os.environ.get(
        "AUTO_EVAL_MONITOR_STATE",
        "~/.cache/verl_qwen3vl_auto_eval_monitor/submitted.json",
    )
    return Path(raw).expanduser()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 2, "submitted": [], "records": {}}
    with path.open("r") as f:
        payload = json.load(f)
    submitted = payload.get("submitted", [])
    if not isinstance(submitted, list):
        raise ValueError(f"Invalid state file, 'submitted' must be a list: {path}")
    records = payload.get("records", {})
    if not isinstance(records, dict):
        raise ValueError(f"Invalid state file, 'records' must be an object: {path}")
    for key in submitted:
        records.setdefault(key, {"legacy": True, "submission_key": key})
    return {"version": 2, "submitted": sorted(set(submitted) | set(records)), "records": records}


def submitted_keys(state: dict[str, Any]) -> set[str]:
    return set(state.get("submitted", [])) | set(state.get("records", {}))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    records = state.setdefault("records", {})
    state["version"] = 2
    state["submitted"] = sorted(set(state.get("submitted", [])) | set(records))
    with tmp_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def legacy_submission_key(exp: Experiment, step: int, task: str) -> str:
    return f"{exp.exp_name}|{step}|{task}"


def eval_config_signature(args: argparse.Namespace) -> str:
    payload = {
        "judge_model": args.judge_model,
        "openai_api_base": args.openai_api_base,
        "openai_auth_scheme": args.openai_auth_scheme,
        "pool": args.pool,
        "retry": args.retry,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def submission_key(exp: Experiment, step: int, task: str, args: argparse.Namespace) -> str:
    return f"{legacy_submission_key(exp, step, task)}|eval={eval_config_signature(args)}"


def make_record(
    exp: Experiment,
    step: int,
    task: str,
    hf_path: str,
    args: argparse.Namespace,
    launch_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "submission_key": submission_key(exp, step, task, args),
        "legacy_key": legacy_submission_key(exp, step, task),
        "submitted_at": utc_now(),
        "exp_name": exp.exp_name,
        "model_type": exp.model_type,
        "prompt_type": PROMPT_TYPE,
        "step": step,
        "task": task,
        "hdfs_path": hf_path,
        "pool": args.pool,
        "retry": args.retry,
        "judge_model": args.judge_model,
        "openai_api_base": args.openai_api_base,
        "openai_auth_scheme": args.openai_auth_scheme,
        "job_run_id": launch_result.get("job_run_id"),
        "launch": json_safe(launch_result),
    }


@lru_cache(maxsize=1)
def load_launch_eval_tasks():
    module_path = Path(__file__).resolve().parent / "verl" / "trainer" / "ppo" / "launch_eval_task.py"
    spec = importlib.util.spec_from_file_location("verl_qwen3vl_launch_eval_task", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load launch_eval_task module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.launch_eval_tasks


def submit_one(exp: Experiment, step: int, task: str, hf_path: str, args: argparse.Namespace) -> dict[str, Any]:
    launch_eval_tasks = load_launch_eval_tasks()

    last_error = None
    for attempt in range(1, args.launch_retries + 1):
        try:
            result = launch_eval_tasks(
                PROMPT_TYPE,
                exp.model_type,
                task,
                hdfs_path=hf_path,
                pool=args.pool,
                iter=str(step),
                retry=args.retry,
                judge_model=args.judge_model,
                openai_api_base=args.openai_api_base,
                openai_auth_scheme=args.openai_auth_scheme,
            )
            if args.submit_interval > 0:
                time.sleep(args.submit_interval)
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= args.launch_retries:
                break
            print(
                "[retry] "
                f"launch failed attempt={attempt}/{args.launch_retries} "
                f"exp={exp.exp_name} step={step} task={task}: {exc}",
                file=sys.stderr,
            )
            time.sleep(args.launch_retry_sleep)
    raise RuntimeError(
        f"launch failed after {args.launch_retries} attempts: "
        f"exp={exp.exp_name} step={step} task={task}"
    ) from last_error


def scan_once(args: argparse.Namespace, state: dict[str, Any]) -> int:
    tasks = parse_tasks(args.tasks)
    only_steps = parse_steps(args.steps)
    required_files = tuple(args.required_file)
    submitted = submitted_keys(state)
    submitted_count = 0

    for exp in EXPERIMENTS:
        if args.experiment and exp.exp_name not in args.experiment:
            continue

        steps = list_global_steps(exp.hdfs_dir(args.hdfs_root, args.project_name))
        if args.min_step is not None:
            steps = [step for step in steps if step >= args.min_step]
        if args.max_step is not None:
            steps = [step for step in steps if step <= args.max_step]
        if only_steps is not None:
            steps = [step for step in steps if step in only_steps]

        if not steps:
            print(f"[scan] no candidate checkpoints: {exp.exp_name}")
            continue

        for step in steps:
            hf_path = checkpoint_hf_path(exp, step, args)
            if not is_checkpoint_ready(hf_path, required_files):
                continue

            for task in tasks:
                key = submission_key(exp, step, task, args)
                if key in submitted:
                    continue

                print(
                    "[submit] "
                    f"exp={exp.exp_name} step={step} task={task} "
                    f"model_type={exp.model_type} hdfs_path={hf_path}"
                )
                if args.dry_run:
                    continue
                launch_result = submit_one(exp, step, task, hf_path, args)
                if launch_result.get("status") == "skipped":
                    print(f"[skip] auto-eval launch disabled; state not updated for key={key}")
                    continue
                state.setdefault("records", {})[key] = make_record(exp, step, task, hf_path, args, launch_result)
                submitted.add(key)
                save_state(args.state_file, state)
                submitted_count += 1

    return submitted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor HDFS checkpoint directories and submit optional auto-eval jobs.",
    )
    parser.add_argument("--once", action="store_true", help="Scan once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned submissions without launching jobs.")
    parser.add_argument("--project-name", default=os.environ.get("AUTO_EVAL_PROJECT_NAME", PROJECT_NAME))
    parser.add_argument(
        "--hdfs-root",
        default=HDFS_ROOT,
        help="HDFS root containing <project-name>/<experiment>/global_step_* checkpoints.",
    )
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--state-file", type=Path, default=default_state_path())
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--retry", type=int, default=DEFAULT_RETRY)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--launch-retry-sleep", type=int, default=30)
    parser.add_argument("--submit-interval", type=int, default=5)
    parser.add_argument("--openai-api-base", default=os.environ.get("AUTO_EVAL_OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE))
    parser.add_argument("--openai-auth-scheme", default=os.environ.get("AUTO_EVAL_OPENAI_AUTH_SCHEME", DEFAULT_OPENAI_AUTH_SCHEME))
    parser.add_argument("--judge-model", default=os.environ.get("AUTO_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--steps", help="Comma-separated checkpoint steps to include, for example 10,20,30.")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=[exp.exp_name for exp in EXPERIMENTS],
        help="Restrict to one experiment. Can be passed multiple times.",
    )
    parser.add_argument(
        "--required-file",
        action="append",
        default=list(REQUIRED_HF_FILES),
        help="HDFS file under actor/huggingface required before submission. Repeatable.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.state_file = args.state_file.expanduser()
    if not args.hdfs_root:
        parser.error("--hdfs-root or AUTO_EVAL_HDFS_ROOT is required")

    state = load_state(args.state_file)
    print(f"[state] {args.state_file} loaded; submitted={len(submitted_keys(state))}")
    print(f"[tasks] {parse_tasks(args.tasks)}")
    hadoop_conf_dir = os.environ.get(HADOOP_CONF_DIR_OVERRIDE_ENV) or os.environ.get("HADOOP_CONF_DIR", "<unset>")
    print(f"[hdfs] HADOOP_CONF_DIR={hadoop_conf_dir}")
    print(f"[openai] base={args.openai_api_base} auth_scheme={args.openai_auth_scheme} judge={args.judge_model}")

    while True:
        count = scan_once(args, state)
        if args.dry_run:
            print("[dry-run] state file not updated")
        elif count == 0:
            save_state(args.state_file, state)

        if args.once:
            break

        print(f"[sleep] {args.poll_interval}s")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
