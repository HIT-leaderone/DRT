#!/usr/bin/env python3
"""Monitor SM Activity and Tensor Core Activity on the current node.

Primary backend:
- DCGM `dcgmi dmon`
  - 1002 = DCGM_FI_PROF_SM_ACTIVE
  - 1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE

Fallback backend:
- NVML (`pynvml`)
  - exposes only approximate GPU busy utilization
  - does NOT provide Tensor Core Activity

References:
- NVIDIA DCGM field IDs:
  https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
- NVIDIA DCGM profiling overview:
  https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html
"""

from __future__ import annotations

import argparse
import csv
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pynvml

DCGM_SM_ACTIVE = 1002
DCGM_TENSOR_ACTIVE = 1004
FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class GpuMeta:
    index: int
    name: str
    uuid: str


@dataclass
class SampleRow:
    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    sm_active: Optional[float]
    tensor_active: Optional[float]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["auto", "dcgm", "nvml"],
        default="auto",
        help="Monitoring backend. `dcgm` is the only accurate backend for Tensor Core Activity.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="Sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of refresh iterations. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for persisting samples.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect one refresh and exit.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:6.2f}%"


def get_gpu_metadata() -> list[GpuMeta]:
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    metas: list[GpuMeta] = []
    for index in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8", errors="replace")
        metas.append(GpuMeta(index=index, name=str(name), uuid=str(uuid)))
    return metas


def close_nvml() -> None:
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass


def choose_backend(requested: str) -> str:
    has_dcgmi = shutil.which("dcgmi") is not None
    if requested == "auto":
        return "dcgm" if has_dcgmi else "nvml"
    return requested


def safe_float(token: str) -> Optional[float]:
    if token.upper() == "N/A":
        return None
    if not FLOAT_RE.match(token):
        return None
    return float(token)


def parse_dcgm_dmon_line(
    line: str, metric_names: list[str]
) -> Optional[tuple[int, str, dict[str, Optional[float]]]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens = stripped.split()
    if len(tokens) < len(metric_names) + 1:
        return None

    value_tokens = tokens[-len(metric_names) :]
    prefix = tokens[: -len(metric_names)]
    if not prefix:
        return None

    gpu_index: Optional[int] = None
    entity = " ".join(prefix)

    if len(prefix) == 1 and prefix[0].isdigit():
        gpu_index = int(prefix[0])
        entity = f"GPU {gpu_index}"
    elif prefix[-1].isdigit():
        gpu_index = int(prefix[-1])
        entity = " ".join(prefix[:-1]) or f"GPU {gpu_index}"

    if gpu_index is None:
        return None

    values = {metric: safe_float(token) for metric, token in zip(metric_names, value_tokens)}
    return gpu_index, entity, values


class DcgmStream:
    def __init__(self, interval_ms: int):
        self.interval_ms = interval_ms
        self.field_ids = [DCGM_SM_ACTIVE, DCGM_TENSOR_ACTIVE]
        self.metric_names = ["sm_active", "tensor_active"]
        self.process: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.error_queue: queue.Queue[str] = queue.Queue()
        self.latest: dict[int, dict[str, Optional[float]]] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def start(self) -> None:
        command = [
            "dcgmi",
            "dmon",
            "-e",
            ",".join(str(field_id) for field_id in self.field_ids),
            "-d",
            str(self.interval_ms),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None

        stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        stderr_thread.start()

        for line in self.process.stdout:
            parsed = parse_dcgm_dmon_line(line, self.metric_names)
            if parsed is None:
                continue
            gpu_index, _entity, values = parsed
            with self.lock:
                self.latest[gpu_index] = values
            if self.stop_event.is_set():
                break

        stderr_thread.join(timeout=1.0)

    def _stderr_loop(self) -> None:
        assert self.process is not None
        assert self.process.stderr is not None
        for line in self.process.stderr:
            stripped = line.strip()
            if stripped:
                self.error_queue.put(stripped)
            if self.stop_event.is_set():
                break

    def snapshot(self) -> dict[int, dict[str, Optional[float]]]:
        with self.lock:
            return {gpu_index: dict(values) for gpu_index, values in self.latest.items()}

    def check_errors(self) -> None:
        if self.process is not None:
            retcode = self.process.poll()
            if retcode not in (None, 0):
                errors: list[str] = []
                while not self.error_queue.empty():
                    errors.append(self.error_queue.get_nowait())
                suffix = f": {' | '.join(errors)}" if errors else ""
                raise RuntimeError(
                    "`dcgmi dmon` exited unexpectedly while collecting "
                    f"fields {self.field_ids}{suffix}"
                )

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class DcgmMonitor:
    source = "dcgm"

    def __init__(self, metas: list[GpuMeta], interval_ms: int):
        self.metas = metas
        self.interval_ms = interval_ms
        self.stream = DcgmStream(interval_ms)

    def start(self) -> None:
        if shutil.which("dcgmi") is None:
            raise RuntimeError(
                "DCGM backend requested, but `dcgmi` is not installed. "
                "Install NVIDIA DCGM or switch to `--backend nvml`."
            )
        self.stream.start()
        time.sleep(max(self.interval_ms / 1000.0, 0.2))
        self.stream.check_errors()

    def sample(self) -> list[SampleRow]:
        self.stream.check_errors()
        latest = self.stream.snapshot()
        rows: list[SampleRow] = []
        for meta in self.metas:
            metrics = latest.get(meta.index, {})
            rows.append(
                SampleRow(
                    gpu_index=meta.index,
                    gpu_name=meta.name,
                    gpu_uuid=meta.uuid,
                    sm_active=metrics.get("sm_active"),
                    tensor_active=metrics.get("tensor_active"),
                    source=self.source,
                )
            )
        return rows

    def stop(self) -> None:
        self.stream.stop()


class NvmlMonitor:
    source = "nvml-approx"

    def __init__(self, metas: list[GpuMeta]):
        self.metas = metas
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(meta.index) for meta in metas]

    def start(self) -> None:
        return None

    def sample(self) -> list[SampleRow]:
        rows: list[SampleRow] = []
        for meta, handle in zip(self.metas, self.handles):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            rows.append(
                SampleRow(
                    gpu_index=meta.index,
                    gpu_name=meta.name,
                    gpu_uuid=meta.uuid,
                    sm_active=float(util.gpu) / 100.0,
                    tensor_active=None,
                    source=self.source,
                )
            )
        return rows

    def stop(self) -> None:
        return None


def make_monitor(backend: str, metas: list[GpuMeta], interval_ms: int):
    if backend == "dcgm":
        return DcgmMonitor(metas, interval_ms)
    if backend == "nvml":
        return NvmlMonitor(metas)
    raise ValueError(f"Unsupported backend: {backend}")


def render_table(rows: list[SampleRow], timestamp: datetime) -> str:
    header = (
        f"[{format_ts(timestamp)}] source={rows[0].source if rows else 'unknown'}\n"
        "GPU  SM Activity    Tensor Core Activity  Name"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.gpu_index:>3}  {format_pct(row.sm_active):>11}  "
            f"{format_pct(row.tensor_active):>20}  {row.gpu_name}"
        )

    sm_values = [row.sm_active for row in rows if row.sm_active is not None]
    tensor_values = [row.tensor_active for row in rows if row.tensor_active is not None]
    sm_avg = sum(sm_values) / len(sm_values) if sm_values else None
    tensor_avg = sum(tensor_values) / len(tensor_values) if tensor_values else None
    lines.append(
        f"AVG  {format_pct(sm_avg):>11}  {format_pct(tensor_avg):>20}  node-average"
    )

    if rows and rows[0].source == "nvml-approx":
        lines.append(
            "note: NVML backend only exposes approximate GPU busy utilization; "
            "Tensor Core Activity needs DCGM profiling fields."
        )

    return "\n".join(lines)


def open_csv_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(
        [
            "timestamp_utc",
            "gpu_index",
            "gpu_name",
            "gpu_uuid",
            "source",
            "sm_active_ratio",
            "tensor_core_activity_ratio",
        ]
    )
    return handle, writer


def write_csv_rows(writer: csv.writer, timestamp: datetime, rows: list[SampleRow]) -> None:
    ts = timestamp.isoformat()
    for row in rows:
        writer.writerow(
            [
                ts,
                row.gpu_index,
                row.gpu_name,
                row.gpu_uuid,
                row.source,
                "" if row.sm_active is None else f"{row.sm_active:.6f}",
                "" if row.tensor_active is None else f"{row.tensor_active:.6f}",
            ]
        )


def main() -> int:
    args = parse_args()
    if args.once:
        args.iterations = 1

    selected_backend = choose_backend(args.backend)

    csv_handle = None
    monitor = None
    try:
        metas = get_gpu_metadata()
        monitor = make_monitor(selected_backend, metas, args.interval_ms)
        monitor.start()

        writer = None
        if args.csv is not None:
            csv_handle, writer = open_csv_writer(args.csv)

        stop = False

        def _handle_signal(_signum, _frame) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        iteration = 0
        while not stop and (args.iterations <= 0 or iteration < args.iterations):
            rows = monitor.sample()
            now = utc_now()
            print(render_table(rows, now), flush=True)
            if writer is not None:
                write_csv_rows(writer, now, rows)
                csv_handle.flush()
            iteration += 1
            if args.iterations > 0 and iteration >= args.iterations:
                break
            time.sleep(args.interval_ms / 1000.0)

    except pynvml.NVMLError as exc:
        print(f"Failed to initialize NVML: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if monitor is not None:
            monitor.stop()
        if csv_handle is not None:
            csv_handle.close()
        close_nvml()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
