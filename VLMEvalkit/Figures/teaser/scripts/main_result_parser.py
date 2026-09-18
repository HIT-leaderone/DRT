#!/usr/bin/env python3
"""Utilities for reading the compact LaTeX main-result table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


BENCHMARKS = ("MathVista", "MathVerse", "LogicVista", "GSM8K", "Video-Holmes", "AVG.")


@dataclass(frozen=True)
class BenchmarkResult:
    acc: float
    tokens: float


@dataclass(frozen=True)
class MethodResult:
    method: str
    metrics: dict[str, BenchmarkResult]


def clean_method_name(text: str) -> str:
    text = text.strip().rstrip("\\").strip()
    text = re.sub(r"~?\\cite\{[^}]+\}", "", text)
    text = text.replace("~", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_main_result_table(tex_path: Path) -> dict[str, MethodResult]:
    lines = tex_path.read_text(encoding="utf-8").splitlines()
    results: dict[str, MethodResult] = {}
    row_count = len(BENCHMARKS)

    index = 0
    while index < len(lines):
        raw_line = lines[index].strip()
        method = clean_method_name(raw_line)

        if not method or method.startswith("\\") or method.startswith("&"):
            index += 1
            continue

        values: list[float] = []
        cursor = index + 1
        while cursor < len(lines):
            metric_line = lines[cursor].strip()
            if not metric_line:
                cursor += 1
                continue
            if not metric_line.startswith("&"):
                break

            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", metric_line)
            if len(numbers) >= 2:
                values.extend([float(numbers[0]), float(numbers[1])])
            cursor += 1

        if len(values) >= row_count * 2:
            metrics = {
                benchmark: BenchmarkResult(
                    acc=values[position * 2],
                    tokens=values[position * 2 + 1],
                )
                for position, benchmark in enumerate(BENCHMARKS)
            }
            results[method] = MethodResult(method=method, metrics=metrics)
            index = cursor
            continue

        index += 1

    if not results:
        raise ValueError(f"No method rows parsed from {tex_path}")

    return results
