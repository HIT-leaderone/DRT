import pynvml
import torch
import time
import threading

# ================= 配置区域 =================
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name:
        POWER_THRESHOLD = 300
    elif "A100" in gpu_name:
        POWER_THRESHOLD = 200
    else:
        POWER_THRESHOLD = 200
else:
    POWER_THRESHOLD = 200

print(f"Detected GPU: {gpu_name if torch.cuda.is_available() else 'None'}")
print(f"POWER_THRESHOLD = {POWER_THRESHOLD}")

MATRIX_SIZE = 4096          # 矩阵大小 (4096 * 4096)
CHECK_INTERVAL = 0.1        # 检查间隔 (秒)
STRESS_ITERATIONS = 20      # 每次触发负载时执行的矩阵乘法次数
# ===========================================

# 用于记录每个 GPU 是否正在运行负载任务，防止线程堆积
gpu_stressing_status = {}

def stress_gpu_task(gpu_index):
    """
    在指定 GPU 上执行矩阵运算以拉升负载
    """
    try:
        # 标记该 GPU 正在进行压力测试
        gpu_stressing_status[gpu_index] = True
        
        # 指定设备
        device = torch.device(f"cuda:{gpu_index}")
        
        # 创建 4096 * 4096 的随机矩阵
        # 使用 float32 (默认) 可以产生显著的计算负载
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        
        # 执行多次矩阵乘法运算
        for _ in range(STRESS_ITERATIONS):
            # 矩阵乘法是计算密集型操作，能有效拉升 SM 利用率和功耗
            c = torch.mm(a, b)
        
        # 强制同步，确保计算真正完成（因为 CUDA 是异步的）
        torch.cuda.synchronize(device)
        
    except Exception as e:
        print(f"[Error] GPU {gpu_index} 负载任务出错: {e}")
    finally:
        # 任务结束，释放状态
        gpu_stressing_status[gpu_index] = False

def main():
    # 初始化 NVML
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"无法初始化 NVML (NVIDIA 驱动管理库): {e}")
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {device_count} 张 NVIDIA 显卡。开始监控...")
        
        # 初始化状态字典
        for i in range(device_count):
            gpu_stressing_status[i] = False

        while True:
            for i in range(device_count):
                # 1. 获取 GPU 句柄
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 2. 获取当前功率 (NVML 返回的是毫瓦 mW，需要除以 1000 转换成瓦 W)
                power_usage_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_usage_w = power_usage_mw / 1000.0
                
                # 3. 打印当前状态
                status_msg = f"GPU {i}: {power_usage_w:.1f}W"
                
                # 4. 判断逻辑
                if power_usage_w < POWER_THRESHOLD:
                    # 如果当前没有在运行负载任务，则启动一个新线程
                    if not gpu_stressing_status[i]:
                        print(f"\n[警报] {status_msg} < {POWER_THRESHOLD}W -> 启动 4096*4096 矩阵运算...")
                        t = threading.Thread(target=stress_gpu_task, args=(i,))
                        t.start()
                    else:
                        # 如果已经在运行负载，则跳过，避免线程爆炸
                        pass
                else:
                    # 功率达标，无需操作
                    pass

            # 简单的动态刷新显示（可选）
            # print(f"监控中... (按 Ctrl+C 停止)", end='\r')
            
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 清理资源
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
