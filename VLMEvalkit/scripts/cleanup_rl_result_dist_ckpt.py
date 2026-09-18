#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_BASE_URI = os.environ.get("RL_RESULT_BASE_URI", "hdfs://namenode/path/to/RL_result")
DEFAULT_KEEP_TSV = "/tmp/rl_result_audit/keep_paths.tsv"


def normalize_hdfs_path(raw_path: str) -> str:
    path = raw_path.strip()
    if not path:
        raise ValueError("empty path")

    if path.startswith("hdfs://"):
        parts = urlsplit(path)
        path = parts.path

    if not path.startswith("/"):
        raise ValueError(f"unsupported HDFS path: {raw_path}")

    return path.rstrip("/")


def load_keep_paths(tsv_path: Path) -> tuple[set[str], list[str]]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"keep TSV not found: {tsv_path}")

    keep_paths: list[str] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"keep TSV is empty: {tsv_path}")
        handle.seek(0)

        header = first_line.rstrip("\n").split("\t")
        if "path" in header:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                raw_path = (row.get("path") or "").strip()
                if raw_path:
                    keep_paths.append(raw_path)
        else:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                keep_paths.append(line.split("\t")[-1].strip())

    normalized = {normalize_hdfs_path(path) for path in keep_paths}
    return normalized, keep_paths


def list_candidate_paths(hdfs_bin: str, base_uri: str, path_pattern: re.Pattern[str]) -> list[str]:
    command = [hdfs_bin, "dfs", "-ls", "-R", base_uri]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to list HDFS paths with {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    candidates: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line or line.startswith("Found "):
            continue
        parts = line.split(None, 7)
        if len(parts) != 8:
            continue
        path = parts[7].strip()
        normalized = normalize_hdfs_path(path)
        if path_pattern.match(normalized):
            candidates[normalized] = path

    return [candidates[key] for key in sorted(candidates)]


def write_manifest(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def delete_one_path(hdfs_bin: str, path: str, skip_trash: bool) -> tuple[str, bool, str]:
    command = [hdfs_bin, "dfs", "-rm", "-r"]
    if skip_trash:
        command.append("-skipTrash")
    command.append(path)

    print(f"[deleting] {path}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    return path, result.returncode == 0, output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete all HDFS dist_ckpt directories under RL_result except the paths "
            "explicitly listed in the keep TSV. Dry-run by default."
        )
    )
    parser.add_argument(
        "--keep-tsv",
        default=DEFAULT_KEEP_TSV,
        help="Path to the edited keep TSV. If it has multiple columns, the last 'path' column is used.",
    )
    parser.add_argument(
        "--base-uri",
        default=DEFAULT_BASE_URI,
        help=(
            "HDFS root to scan. Defaults to RL_RESULT_BASE_URI or a placeholder; "
            "pass a real URI for your storage environment."
        ),
    )
    parser.add_argument(
        "--hdfs-bin",
        default="hdfs",
        help="HDFS client binary to invoke.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel delete workers used only with --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete non-whitelisted dist_ckpt directories.",
    )
    parser.add_argument(
        "--skip-trash",
        action="store_true",
        help="Pass -skipTrash to hdfs dfs -rm -r.",
    )
    parser.add_argument(
        "--print-delete-paths",
        action="store_true",
        help="Print every delete target after the summary.",
    )
    parser.add_argument(
        "--plan-dir",
        default="",
        help="Optional local directory used to write keep/delete/missing manifests.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    keep_tsv = Path(args.keep_tsv).expanduser().resolve()
    keep_set, keep_rows = load_keep_paths(keep_tsv)
    if not keep_set:
        raise ValueError("keep set is empty; refusing to continue")

    base_inode = normalize_hdfs_path(args.base_uri)
    candidate_pattern = re.compile(
        rf"^{re.escape(base_inode)}/[^/]+/[^/]+/global_step_\d+/actor/dist_ckpt$"
    )
    candidates = list_candidate_paths(args.hdfs_bin, args.base_uri, candidate_pattern)

    if not candidates:
        raise RuntimeError(f"no dist_ckpt candidates found under {args.base_uri}")

    candidate_by_inode = {normalize_hdfs_path(path): path for path in candidates}
    matched_keep = sorted(
        candidate_by_inode[inode] for inode in keep_set if inode in candidate_by_inode
    )
    delete_targets = sorted(
        path for inode, path in candidate_by_inode.items() if inode not in keep_set
    )
    missing_keep = sorted(inode for inode in keep_set if inode not in candidate_by_inode)

    print(f"[mode] {'execute' if args.execute else 'dry-run'}")
    print(f"[keep-tsv] {keep_tsv}")
    print(f"[base-uri] {args.base_uri}")
    print(f"[keep-rows] {len(keep_rows)}")
    print(f"[keep-paths-unique] {len(keep_set)}")
    print(f"[candidate-count] {len(candidates)}")
    print(f"[matched-keep-count] {len(matched_keep)}")
    print(f"[missing-keep-count] {len(missing_keep)}")
    print(f"[delete-count] {len(delete_targets)}")
    if missing_keep:
        print("[warning] some keep paths were not found in the scanned candidate set", file=sys.stderr)

    if args.plan_dir:
        plan_dir = Path(args.plan_dir).expanduser().resolve()
        write_manifest(plan_dir / "keep.txt", matched_keep)
        write_manifest(plan_dir / "delete.txt", delete_targets)
        write_manifest(plan_dir / "missing_keep.txt", missing_keep)
        print(f"[plan-dir] {plan_dir}")

    if args.print_delete_paths:
        for path in delete_targets:
            print(f"[delete-target] {path}")

    if not args.execute:
        return 0

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as executor:
        futures = {
            executor.submit(delete_one_path, args.hdfs_bin, path, args.skip_trash): path
            for path in delete_targets
        }
        for future in as_completed(futures):
            path, ok, output = future.result()
            if ok:
                print(f"[deleted] {path}")
            else:
                failures.append((path, output))
                print(f"[delete-failed] {path}", file=sys.stderr)
                if output:
                    print(output, file=sys.stderr)

    print(f"[deleted-count] {len(delete_targets) - len(failures)}")
    print(f"[failed-count] {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
