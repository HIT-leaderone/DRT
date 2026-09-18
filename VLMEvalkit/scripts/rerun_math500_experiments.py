#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(os.environ.get("VLMEVALKIT_WORK_DIR", "./results"))
DEFAULT_DATASET = "MATH-500"
RESULT_SUFFIXES = (
    ".xlsx",
    "_score.json",
    ".pkl",
    ".tsv",
    ".json",
    ".csv",
)


@dataclass(frozen=True)
class ExperimentSpec:
    work_dir: Path
    model: str
    prompt_type: str


def infer_prompt_type(path: Path) -> str:
    parts = set(path.parts)
    if "Short-COT-Image" in parts:
        return "Short-COT-Image"
    if "Short-COT-Video" in parts:
        return "Short-COT-Video"
    if "Directly-Answer" in parts:
        return "Directly-Answer"
    return "Custom-Prompt"


def infer_model_name(path: Path, dataset: str) -> str | None:
    name = path.name
    suffixes = [
        f"_{dataset}.xlsx",
        f"_{dataset}.tsv",
        f"_{dataset}.json",
        f"_{dataset}.csv",
        f"_{dataset}.pkl",
        f"_{dataset}_score.json",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return None


def infer_model_dir(path: Path, model_name: str) -> Path | None:
    current = path.parent
    while current != current.parent:
        if current.name == model_name:
            return current
        current = current.parent
    return None


def discover_specs(base_dir: Path, dataset: str) -> tuple[list[ExperimentSpec], list[Path]]:
    specs = set()
    matched_files: list[Path] = []
    for dirpath, _, filenames in os.walk(base_dir):
        root = Path(dirpath)
        for filename in filenames:
            if dataset not in filename:
                continue
            if not any(filename.endswith(suffix) for suffix in RESULT_SUFFIXES):
                continue
            path = root / filename
            model_name = infer_model_name(path, dataset)
            if model_name is None:
                continue
            model_dir = infer_model_dir(path, model_name)
            if model_dir is None:
                continue
            matched_files.append(path)
            specs.add(
                ExperimentSpec(
                    work_dir=model_dir.parent,
                    model=model_name,
                    prompt_type=infer_prompt_type(path),
                ))
    return sorted(specs, key=lambda x: (str(x.work_dir), x.model)), sorted(set(matched_files))


def build_command(spec: ExperimentSpec, dataset: str, judge: str, api_nproc: int, use_vllm: bool) -> list[str]:
    cmd = [
        sys.executable,
        "run.py",
        "--data",
        dataset,
        "--model",
        spec.model,
        "--work-dir",
        str(spec.work_dir),
        "--judge",
        judge,
        "--api-nproc",
        str(api_nproc),
        "--verbose",
    ]
    if use_vllm:
        cmd.append("--use-vllm")
    return cmd


def collect_spec_files(spec: ExperimentSpec, dataset: str) -> list[Path]:
    root = spec.work_dir / spec.model
    matches: list[Path] = []
    if not root.exists():
        return matches
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if dataset in filename:
                matches.append(Path(dirpath) / filename)
    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Discover experiments with existing MATH-500 results under a VLMEvalKit "
            "workspace, optionally delete old MATH-500 files, and rerun them."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Root directory to scan.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset keyword to match.")
    parser.add_argument(
        "--judge",
        default="gpt-4.1-2025-04-14",
        help="Judge model passed to run.py.",
    )
    parser.add_argument(
        "--api-nproc",
        type=int,
        default=4,
        help="Value passed to run.py --api-nproc.",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Do not pass --use-vllm to run.py.",
    )
    parser.add_argument(
        "--cleanup-existing",
        action="store_true",
        help="Delete matched dataset-related files under each model directory before rerun.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute cleanup and rerun. Without this flag, only print actions.",
    )
    args = parser.parse_args()

    specs, matched_files = discover_specs(args.base_dir, args.dataset)

    print(f"[mode] {'execute' if args.execute else 'dry-run'}")
    print(f"[base-dir] {args.base_dir}")
    print(f"[dataset] {args.dataset}")
    print(f"[matched-files] {len(matched_files)}")
    for path in matched_files:
        print(path)

    print(f"[experiments] {len(specs)}")
    for spec in specs:
        print(f"{spec.work_dir} | model={spec.model} | prompt_type={spec.prompt_type}")

    print("[commands]")
    for spec in specs:
        cmd = build_command(
            spec,
            dataset=args.dataset,
            judge=args.judge,
            api_nproc=args.api_nproc,
            use_vllm=not args.no_vllm,
        )
        print(f"PROMPT_TYPE={spec.prompt_type} " + " ".join(cmd))

    if not args.execute:
        return

    env = os.environ.copy()
    for spec in specs:
        if args.cleanup_existing:
            existing_files = collect_spec_files(spec, args.dataset)
            for path in existing_files:
                try:
                    path.unlink()
                    print(f"[deleted] {path}")
                except FileNotFoundError:
                    pass

        env["PROMPT_TYPE"] = spec.prompt_type
        cmd = build_command(
            spec,
            dataset=args.dataset,
            judge=args.judge,
            api_nproc=args.api_nproc,
            use_vllm=not args.no_vllm,
        )
        print(f"[running] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=env, check=True)


if __name__ == "__main__":
    main()
