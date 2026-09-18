import os
import threading
import time

import pynvml
import torch


# ================= 配置区域 =================
SCRIPT_VERSION = "2026-03-20-r4"
CHECK_INTERVAL = 0.2          # 主循环轮询间隔
IDLE_HOLD_SECONDS = 5.0       # 连续空闲多久后，才启动一轮负载
LOAD_PHASE_SECONDS = 60.0     # 单轮持续施加负载的时长
CHECK_PHASE_SECONDS = 10.0    # 单轮结束后，暂停检查竞争进程的时长

IDLE_UTIL_THRESHOLD = 4       # GPU 利用率低于该值，才认为可能空闲
PROCESS_SAMPLE_UTIL_THRESHOLD = 2   # 记录进程活跃样本的最低阈值
PROCESS_ACTIVE_WINDOW = 1.5         # 进程活跃样本保留时间窗口

DTYPE = torch.float16
# ===========================================


SELF_PID = os.getpid()

gpu_running_status = {}
gpu_idle_since = {}
gpu_phase = {}
gpu_phase_deadline = {}
gpu_profiles = {}
gpu_process_sample_ts = {}
gpu_recent_process_activity = {}
gpu_check_seen_pids = {}


def get_gpu_name(handle):
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def build_gpu_profile(gpu_name):
    if "H100" in gpu_name:
        return {
            "idle_power_threshold": 120.0,
            "matrix_size": 4096,
            "burst_iterations": 4,
            "burst_pause": 0.0001,
        }
    if "A100" in gpu_name:
        return {
            "idle_power_threshold": 120.0,
            "matrix_size": 4096,
            "burst_iterations": 4,
            "burst_pause": 0.0001,
        }
    return {
        "idle_power_threshold": 70.0,
        "matrix_size": 3072,
        "burst_iterations": 3,
        "burst_pause": 0.0001,
    }


def get_compute_pids(handle):
    query_fns = [
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
    ]

    for query_fn in query_fns:
        if query_fn is None:
            continue
        try:
            return {proc.pid for proc in query_fn(handle)}
        except pynvml.NVMLError_NotSupported:
            return set()
        except pynvml.NVMLError_FunctionNotFound:
            continue
        except pynvml.NVMLError:
            return set()

    return set()


def get_active_foreign_processes(handle, gpu_index, now):
    """
    返回最近确实有 GPU 活动样本的外部进程。
    若驱动/库不支持进程级利用率，则退化为只看 compute context。
    """
    last_seen_ts = gpu_process_sample_ts[gpu_index]
    activity = gpu_recent_process_activity[gpu_index]

    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, last_seen_ts)
    except pynvml.NVMLError_NotFound:
        samples = []
    except pynvml.NVMLError_NotSupported:
        samples = None
    except pynvml.NVMLError:
        samples = []

    if samples is None:
        pids = sorted(pid for pid in get_compute_pids(handle) if pid != SELF_PID)
        return {
            "pids": pids,
            "source": "context",
        }

    next_ts = last_seen_ts
    for sample in samples:
        next_ts = max(next_ts, int(sample.timeStamp) + 1)
        sample_util = max(int(sample.smUtil), int(sample.memUtil))
        if sample_util >= PROCESS_SAMPLE_UTIL_THRESHOLD:
            activity[int(sample.pid)] = now

    gpu_process_sample_ts[gpu_index] = next_ts

    stale_pids = [pid for pid, seen_at in activity.items() if now - seen_at > PROCESS_ACTIVE_WINDOW]
    for pid in stale_pids:
        del activity[pid]

    pids = sorted(pid for pid in activity if pid != SELF_PID)
    return {
        "pids": pids,
        "source": "util",
    }


def stress_gpu_task(gpu_index, profile):
    """
    单线程连续跑固定时长，不在运行中途让路。
    """
    a = None
    b = None
    c = None

    try:
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")

        matrix_size = profile["matrix_size"]
        burst_iterations = profile["burst_iterations"]
        burst_pause = profile["burst_pause"]
        run_until = time.time() + LOAD_PHASE_SECONDS
        burst_count = 0

        with torch.inference_mode():
            a = torch.randn(matrix_size, matrix_size, device=device, dtype=DTYPE)
            b = torch.randn(matrix_size, matrix_size, device=device, dtype=DTYPE)
            c = torch.empty(matrix_size, matrix_size, device=device, dtype=DTYPE)

            while time.time() < run_until:
                for _ in range(burst_iterations):
                    torch.mm(a, b, out=c)

                torch.cuda.synchronize(device)
                burst_count += 1

                if burst_pause > 0:
                    time.sleep(burst_pause)

        print(
            f"[负载结束] GPU {gpu_index}: 已连续运行 {LOAD_PHASE_SECONDS:.0f}s, "
            f"bursts={burst_count}"
        )

    except Exception as e:
        print(f"[Error] GPU {gpu_index} 负载任务出错: {e}")
    finally:
        try:
            del a
            del b
            del c
        except Exception:
            pass

        try:
            with torch.cuda.device(gpu_index):
                torch.cuda.empty_cache()
        except Exception:
            pass

        gpu_running_status[gpu_index] = False


def main():
    if not torch.cuda.is_available():
        print("未检测到可用 CUDA 设备。")
        return

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"无法初始化 NVML (NVIDIA 驱动管理库): {e}")
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        print(
            f"use_gpu.py version={SCRIPT_VERSION}, pid={SELF_PID}\n"
            f"检测到 {device_count} 张 NVIDIA 显卡。开始监控..."
        )

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = get_gpu_name(handle)
            profile = build_gpu_profile(gpu_name)

            gpu_profiles[i] = profile
            gpu_running_status[i] = False
            gpu_idle_since[i] = None
            gpu_phase[i] = "idle"
            gpu_phase_deadline[i] = 0.0
            gpu_process_sample_ts[i] = 0
            gpu_recent_process_activity[i] = {}
            gpu_check_seen_pids[i] = set()

            print(
                f"GPU {i}: {gpu_name}, "
                f"idle_power<{profile['idle_power_threshold']}W, "
                f"matrix={profile['matrix_size']}, "
                f"iters={profile['burst_iterations']}, "
                f"pause={profile['burst_pause']}, "
                f"load={LOAD_PHASE_SECONDS:.0f}s, check={CHECK_PHASE_SECONDS:.0f}s"
            )

        while True:
            now = time.time()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                profile = gpu_profiles[i]
                phase = gpu_phase[i]

                power_usage_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu

                is_idle_now = (
                    gpu_util <= IDLE_UTIL_THRESHOLD
                    and power_usage_w <= profile["idle_power_threshold"]
                )

                if phase == "running":
                    if not gpu_running_status[i]:
                        gpu_phase[i] = "checking"
                        gpu_phase_deadline[i] = now + CHECK_PHASE_SECONDS
                        gpu_check_seen_pids[i] = set()
                        gpu_idle_since[i] = None
                        print(
                            f"[进入检查] GPU {i}: 负载阶段结束，暂停 {CHECK_PHASE_SECONDS:.0f}s 检查竞争进程"
                        )
                    continue

                active_foreign = get_active_foreign_processes(handle, i, now)
                active_foreign_pids = active_foreign["pids"]

                if phase == "checking":
                    if active_foreign_pids:
                        gpu_check_seen_pids[i].update(active_foreign_pids)

                    if now < gpu_phase_deadline[i]:
                        continue

                    seen_pids = sorted(gpu_check_seen_pids[i])
                    gpu_phase[i] = "idle"

                    if seen_pids:
                        gpu_idle_since[i] = None
                        print(
                            f"[检查结果] GPU {i}: 在 {CHECK_PHASE_SECONDS:.0f}s 窗口内发现竞争进程 "
                            f"{seen_pids} -> 暂不启动下一轮"
                        )
                    else:
                        gpu_idle_since[i] = now - IDLE_HOLD_SECONDS
                        print(
                            f"[检查结果] GPU {i}: {CHECK_PHASE_SECONDS:.0f}s 内未发现竞争进程 -> 允许下一轮负载"
                        )

                    continue

                if active_foreign_pids:
                    gpu_idle_since[i] = None
                    continue

                if is_idle_now:
                    if gpu_idle_since[i] is None:
                        gpu_idle_since[i] = now
                else:
                    gpu_idle_since[i] = None

                idle_duration = 0.0
                if gpu_idle_since[i] is not None:
                    idle_duration = now - gpu_idle_since[i]

                can_launch = (
                    phase == "idle"
                    and not gpu_running_status[i]
                    and idle_duration >= IDLE_HOLD_SECONDS
                )

                if can_launch:
                    gpu_phase[i] = "running"
                    gpu_running_status[i] = True
                    gpu_idle_since[i] = None

                    print(
                        f"[启动负载] GPU {i}: 连续空闲 {idle_duration:.1f}s, "
                        f"util={gpu_util}%, power={power_usage_w:.1f}W, "
                        f"matrix={profile['matrix_size']}, "
                        f"iters={profile['burst_iterations']}, "
                        f"pause={profile['burst_pause']}, "
                        f"duration={LOAD_PHASE_SECONDS:.0f}s"
                    )

                    t = threading.Thread(
                        target=stress_gpu_task,
                        args=(i, profile),
                        daemon=True,
                    )
                    try:
                        t.start()
                    except Exception:
                        gpu_running_status[i] = False
                        gpu_phase[i] = "idle"
                        raise

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
