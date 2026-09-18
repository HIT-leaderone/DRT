#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reproduce" / "visionthink" / "scripts"))

from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.smp import dump  # noqa: E402

DEFAULT_DATASETS = [
    "MathVista_MINI",
    "MathVerse_MINI",
    "LogicVista",
    "Video_Holmes",
    "GSM8K",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionThink time-eval runner for VLMEvalKit datasets.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--setting", choices=["sequential", "parallel"], default="parallel")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=32,
        help="Number of concurrent client workers submitting requests to the local VisionThink scheduler.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum dynamic batch size inside the local VisionThink scheduler. Defaults to parallel-workers.",
    )
    parser.add_argument(
        "--batch-collect-timeout-ms",
        type=float,
        default=20.0,
        help="Maximum wait time for collecting a fuller dynamic batch before flushing.",
    )
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-generation-round", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-downsample-image", action="store_true")
    return parser.parse_args()


def _mean(values: list[Any]) -> float:
    valid = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or math.isnan(value):
            continue
        weight = int(row.get("num_samples", 0))
        if weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def resolve_batch_size(args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return args.batch_size
    if args.setting == "parallel":
        return args.parallel_workers
    return 1


@dataclass
class ScheduledRequest:
    sample: Any
    future: Future
    submit_ts: float


class LocalDynamicBatchScheduler:
    def __init__(
        self,
        runner: VisionThinkRunner,
        max_batch_size: int,
        batch_collect_timeout_ms: float,
    ) -> None:
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.batch_collect_timeout_sec = batch_collect_timeout_ms / 1000.0
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_token = object()
        self._thread: threading.Thread | None = None
        self._closed = False

    def __enter__(self) -> "LocalDynamicBatchScheduler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._serve_loop,
            name="visionthink-dynamic-batcher",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            if self._thread is not None:
                self._thread.join()
            return
        self._closed = True
        if self._thread is not None:
            self._queue.put(self._stop_token)
            self._thread.join()

    def submit(self, sample: Any) -> Future:
        if self._closed:
            raise RuntimeError("LocalDynamicBatchScheduler is already closed.")
        self.start()
        future: Future = Future()
        request = ScheduledRequest(
            sample=sample,
            future=future,
            submit_ts=time.perf_counter(),
        )
        self._queue.put(request)
        return future

    def _collect_batch(self, first_request: ScheduledRequest) -> tuple[list[ScheduledRequest], bool]:
        batch = [first_request]
        stop_after_batch = False
        deadline = time.perf_counter() + self.batch_collect_timeout_sec

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is self._stop_token:
                stop_after_batch = True
                break
            batch.append(item)

        return batch, stop_after_batch

    def _serve_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop_token:
                break

            batch, stop_after_batch = self._collect_batch(item)
            batch_start = time.perf_counter()

            try:
                raw_texts, token_lengths = self.runner.generate_batch([request.sample for request in batch])
                if len(raw_texts) != len(batch) or len(token_lengths) != len(batch):
                    raise RuntimeError(
                        f"Batch output shape mismatch: got {len(raw_texts)} texts and {len(token_lengths)} lengths for {len(batch)} requests."
                    )

                batch_end = time.perf_counter()
                batch_latency = batch_end - batch_start
                batch_size_used = len(batch)

                for request, raw_text, token_length in zip(batch, raw_texts, token_lengths):
                    if request.future.done():
                        continue
                    request.future.set_result(
                        {
                            "prediction": self.runner._extract_answer(raw_text),
                            "raw_prediction": raw_text,
                            "output_length_tokens": token_length,
                            "item_latency_sec": batch_end - request.submit_ts,
                            "queue_wait_sec": batch_start - request.submit_ts,
                            "model_exec_sec": batch_latency,
                            "batch_latency_sec": batch_latency,
                            "batch_size_used": batch_size_used,
                        }
                    )
            except Exception as exc:
                error = RuntimeError(f"VisionThink batch inference failed: {exc}")
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(error)

            if stop_after_batch:
                break


class VisionThinkDatasetEvaluator:
    def __init__(
        self,
        runner: VisionThinkRunner,
        output_dir: Path,
        setting: str,
        parallel_workers: int,
        max_batch_size: int,
        batch_collect_timeout_ms: float,
    ) -> None:
        self.runner = runner
        self.output_dir = output_dir
        self.setting = setting
        self.parallel_workers = parallel_workers
        self.max_batch_size = max_batch_size
        self.batch_collect_timeout_ms = batch_collect_timeout_ms

    def _failure_result(self, row, error: str) -> dict[str, Any]:
        return {
            "index": row.get("index", -1),
            "prediction": "",
            "raw_prediction": "",
            "answer": row.get("answer", ""),
            "item_latency_sec": None,
            "queue_wait_sec": None,
            "model_exec_sec": None,
            "batch_latency_sec": None,
            "batch_size_used": None,
            "prepare_input_sec": None,
            "output_length_tokens": 0,
            "success": False,
            "error": error,
        }

    def _evaluate_sequential(self, dataset_name: str, dataset, data: pd.DataFrame) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for idx in range(len(data)):
            row = data.iloc[idx]
            try:
                prepare_start = time.perf_counter()
                sample = self.runner.build_sample_input(dataset_name, dataset, row)
                prepare_input_sec = time.perf_counter() - prepare_start

                batch_start = time.perf_counter()
                raw_texts, token_lengths = self.runner.generate_batch([sample])
                batch_end = time.perf_counter()
                batch_latency = batch_end - batch_start

                results.append(
                    {
                        "index": sample.index,
                        "prediction": self.runner._extract_answer(raw_texts[0]),
                        "raw_prediction": raw_texts[0],
                        "answer": sample.answer,
                        "item_latency_sec": batch_latency,
                        "queue_wait_sec": 0.0,
                        "model_exec_sec": batch_latency,
                        "batch_latency_sec": batch_latency,
                        "batch_size_used": 1,
                        "prepare_input_sec": prepare_input_sec,
                        "output_length_tokens": token_lengths[0],
                        "success": True,
                    }
                )
            except Exception as exc:
                print(f"处理第 {idx} 个样本时出错: {exc}")
                results.append(self._failure_result(row, str(exc)))

        return results

    def _process_parallel_sample(
        self,
        dataset_name: str,
        dataset,
        row,
        scheduler: LocalDynamicBatchScheduler,
    ) -> dict[str, Any]:
        prepare_start = time.perf_counter()
        sample = self.runner.build_sample_input(dataset_name, dataset, row)
        prepare_input_sec = time.perf_counter() - prepare_start

        scheduled = scheduler.submit(sample).result()
        scheduled.update(
            {
                "index": sample.index,
                "answer": sample.answer,
                "prepare_input_sec": prepare_input_sec,
                "success": True,
            }
        )
        return scheduled

    def _evaluate_parallel(self, dataset_name: str, dataset, data: pd.DataFrame) -> list[dict[str, Any]]:
        num_samples = len(data)
        print(f"并行评测模式: 使用 {self.parallel_workers} 个客户端并发线程")
        print(f"动态 batching 最大 batch size: {self.max_batch_size}")
        print(f"批收集超时: {self.batch_collect_timeout_ms:.1f} ms")
        print(f"总样本数: {num_samples}")

        results: list[dict[str, Any] | None] = [None] * num_samples

        with LocalDynamicBatchScheduler(
            runner=self.runner,
            max_batch_size=self.max_batch_size,
            batch_collect_timeout_ms=self.batch_collect_timeout_ms,
        ) as scheduler:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_to_idx = {}
                for idx in range(num_samples):
                    row = data.iloc[idx].copy()
                    future = executor.submit(
                        self._process_parallel_sample,
                        dataset_name,
                        dataset,
                        row,
                        scheduler,
                    )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    row = data.iloc[idx]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        print(f"处理第 {idx} 个样本时出错: {exc}")
                        results[idx] = self._failure_result(row, str(exc))

        return [result if result is not None else self._failure_result(data.iloc[idx], "Missing result.") for idx, result in enumerate(results)]

    def evaluate_dataset(self, dataset_name: str, limit: int | None) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"评测 {dataset_name}")
        print(f"评测模式: {self.setting}")
        if self.setting == "parallel":
            print(f"客户端并发线程数: {self.parallel_workers}")
            print(f"动态 batching 最大 batch size: {self.max_batch_size}")
        print("=" * 60)

        dataset = build_dataset(dataset_name)
        data = dataset.data.copy()
        if limit is not None:
            data = data.iloc[:limit].copy()
        num_samples = len(data)
        print(f"样本数量: {num_samples}")

        start_time = time.time()
        if self.setting == "sequential":
            predictions = self._evaluate_sequential(dataset_name, dataset, data)
        elif self.setting == "parallel":
            predictions = self._evaluate_parallel(dataset_name, dataset, data)
        else:
            raise ValueError(f"Unknown setting: {self.setting}")
        total_time = time.time() - start_time

        records = []
        for idx in range(num_samples):
            row_dict = data.iloc[idx].to_dict()
            pred_item = predictions[idx]
            row_dict["prediction"] = pred_item["prediction"]
            row_dict["raw_prediction"] = pred_item["raw_prediction"]
            row_dict["item_latency_sec"] = pred_item["item_latency_sec"]
            row_dict["queue_wait_sec"] = pred_item["queue_wait_sec"]
            row_dict["model_exec_sec"] = pred_item["model_exec_sec"]
            row_dict["batch_latency_sec"] = pred_item["batch_latency_sec"]
            row_dict["batch_size_used"] = pred_item["batch_size_used"]
            row_dict["prepare_input_sec"] = pred_item["prepare_input_sec"]
            row_dict["output_length_tokens"] = pred_item["output_length_tokens"]
            records.append(row_dict)

        pred_df = pd.DataFrame(records)
        pred_path = self.output_dir / f"{dataset_name}_visionthink_{self.setting}_predictions.xlsx"
        dump(pred_df, str(pred_path))

        avg_output_tokens = _mean([item["output_length_tokens"] for item in predictions])
        avg_item_latency = _mean([item["item_latency_sec"] for item in predictions])
        avg_queue_wait = _mean([item["queue_wait_sec"] for item in predictions])
        avg_model_exec = _mean([item["model_exec_sec"] for item in predictions])
        avg_prepare_input = _mean([item["prepare_input_sec"] for item in predictions])
        avg_batch_size_used = _mean([item["batch_size_used"] for item in predictions])

        return {
            "dataset": dataset_name,
            "num_samples": num_samples,
            "total_time_sec_wall": total_time,
            "avg_time_per_sample_sec_wall": (total_time / num_samples) if num_samples else 0.0,
            "avg_item_latency_sec": avg_item_latency,
            "avg_queue_wait_sec": avg_queue_wait,
            "avg_model_exec_sec": avg_model_exec,
            "avg_prepare_input_sec": avg_prepare_input,
            "avg_batch_size_used": avg_batch_size_used,
            "avg_output_length_tokens": avg_output_tokens,
            "accuracy": 0.0,
            "predictions_file": str(pred_path),
            "model": Path(self.runner.model_path).name.replace("/", "_"),
        }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch_size = resolve_batch_size(args)

    from run_vlmeval_bridge import VisionThinkRunner

    runner = VisionThinkRunner(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=effective_batch_size,
        max_images=args.max_images,
        max_generation_round=args.max_generation_round,
        downsample_image=not args.no_downsample_image,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    evaluator = VisionThinkDatasetEvaluator(
        runner=runner,
        output_dir=output_dir,
        setting=args.setting,
        parallel_workers=args.parallel_workers,
        max_batch_size=effective_batch_size,
        batch_collect_timeout_ms=args.batch_collect_timeout_ms,
    )

    rows = []
    for dataset_name in args.datasets:
        rows.append(evaluator.evaluate_dataset(dataset_name, args.limit))

    total_samples = sum(row["num_samples"] for row in rows)
    total_time = sum(row["total_time_sec_wall"] for row in rows)

    summary_rows = [
        {
            "dataset": row["dataset"],
            "num_samples": row["num_samples"],
            "total_time_sec_wall": row["total_time_sec_wall"],
            "avg_time_per_sample_sec_wall": row["avg_time_per_sample_sec_wall"],
            "avg_item_latency_sec": row["avg_item_latency_sec"],
            "avg_queue_wait_sec": row["avg_queue_wait_sec"],
            "avg_model_exec_sec": row["avg_model_exec_sec"],
            "avg_prepare_input_sec": row["avg_prepare_input_sec"],
            "avg_batch_size_used": row["avg_batch_size_used"],
            "avg_output_length_tokens": row["avg_output_length_tokens"],
            "accuracy": row["accuracy"],
        }
        for row in rows
    ]
    summary_rows.append(
        {
            "dataset": "TOTAL",
            "num_samples": total_samples,
            "total_time_sec_wall": total_time,
            "avg_time_per_sample_sec_wall": (total_time / total_samples) if total_samples > 0 else 0.0,
            "avg_item_latency_sec": _weighted_mean(rows, "avg_item_latency_sec"),
            "avg_queue_wait_sec": _weighted_mean(rows, "avg_queue_wait_sec"),
            "avg_model_exec_sec": _weighted_mean(rows, "avg_model_exec_sec"),
            "avg_prepare_input_sec": _weighted_mean(rows, "avg_prepare_input_sec"),
            "avg_batch_size_used": _weighted_mean(rows, "avg_batch_size_used"),
            "avg_output_length_tokens": _weighted_mean(rows, "avg_output_length_tokens"),
            "accuracy": None,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"summary_visionthink_{args.setting}.xlsx"
    dump(summary_df, str(summary_path))

    meta_path = output_dir / "visionthink_timeeval_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "datasets": args.datasets,
                "setting": args.setting,
                "parallel_workers": args.parallel_workers,
                "max_batch_size": effective_batch_size,
                "batch_size_override": args.batch_size,
                "batch_collect_timeout_ms": args.batch_collect_timeout_ms,
                "max_images": args.max_images,
                "max_generation_round": args.max_generation_round,
                "downsample_image": not args.no_downsample_image,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
