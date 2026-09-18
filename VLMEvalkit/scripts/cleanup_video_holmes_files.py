#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


BASE_DIR = Path(os.environ.get("VLMEVALKIT_SHORT_COT_IMAGE_DIR", "./Short-COT-Image"))
AUTOEVAL_MODEL_DIR = "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Ray-Training-RL-autoeval"
THINKING_MODEL_DIR = "Qwen3-VL-8B-Thinking-VisionR1-SFT-2000it"
DEFAULT_KEYWORD = "Video_Holmes"


def build_target_roots() -> list[Path]:
    roots = [BASE_DIR / f"{i}0it" / AUTOEVAL_MODEL_DIR for i in range(1, 8)]
    roots.append(BASE_DIR / THINKING_MODEL_DIR)
    return roots


def collect_matching_files(roots: list[Path], keyword: str) -> list[Path]:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if keyword in filename:
                    matches.append(Path(dirpath) / filename)
    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "List or delete Video_Holmes-related result files under the specified "
            "Short-COT-Image experiment directories. Dry-run by default."
        )
    )
    parser.add_argument(
        "--keyword",
        default=DEFAULT_KEYWORD,
        help="Only files whose basename contains this keyword will be selected.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched files. Without this flag, only print them.",
    )
    parser.add_argument(
        "--show-missing-roots",
        action="store_true",
        help="Also print roots that do not exist.",
    )
    args = parser.parse_args()

    roots = build_target_roots()
    if args.show_missing_roots:
        for root in roots:
            if not root.exists():
                print(f"[missing-root] {root}")

    matches = collect_matching_files(roots, args.keyword)

    print(f"[mode] {'execute' if args.execute else 'dry-run'}")
    print(f"[keyword] {args.keyword}")
    print(f"[matched-files] {len(matches)}")
    for path in matches:
        print(path)

    if not args.execute:
        return

    for path in matches:
        path.unlink()
        print(f"[deleted] {path}")


if __name__ == "__main__":
    main()
