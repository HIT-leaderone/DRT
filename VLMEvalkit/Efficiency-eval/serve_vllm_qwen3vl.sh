#!/bin/bash

# Qwen3-VL-7B VLLM 部署脚本
# 使用方法: bash serve_vllm_qwen3vl.sh <checkpoint_path> [served_model_name]

set -e

# 获取 checkpoint 路径
CKPT_PATH=${1:-"/path/to/Qwen3-VL-8B-Instruct"}
SERVED_MODEL_NAME=${2:-"$(basename "$CKPT_PATH")"}

# 检查 checkpoint 路径是否存在 (如果是本地路径)
if [[ ! "$CKPT_PATH" =~ ^Qwen/ ]] && [ ! -d "$CKPT_PATH" ]; then
    echo "错误: Checkpoint 路径不存在: $CKPT_PATH"
    echo "用法: bash serve_vllm_qwen3vl.sh <checkpoint_path> [served_model_name]"
    exit 1
fi

# 设置日志文件
LOG_FILE="./vllm_server_$(date +%Y%m%d_%H%M%S).log"

echo "================================================"
echo "启动 Qwen3-VL-7B VLLM Server"
echo "Checkpoint: $CKPT_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "日志文件: $LOG_FILE"
echo "================================================"

# 启动 VLLM Server
# 对于 7B 模型，使用 tensor_parallel_size=1 即可
nohup vllm serve "$CKPT_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --limit-mm-per-prompt.video 1 \
    --allowed-local-media-path "${VLLM_ALLOWED_LOCAL_MEDIA_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}" \
    --mm-processor-cache-gb 500 \
    --port 8000 \
    --host 0.0.0.0 \
    > "$LOG_FILE" 2>&1 &

# 获取进程 ID
SERVER_PID=$!

echo ""
echo "VLLM Server 已启动，PID: $SERVER_PID"
echo "等待服务启动完成..."

# 等待服务启动
sleep 15

# 检查服务是否正常运行
MAX_RETRY=20
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRY ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "================================================"
        echo "✓ VLLM Server 启动成功!"
        echo "API 地址: http://localhost:8000"
        echo "PID: $SERVER_PID"
        echo "日志: $LOG_FILE"
        echo "================================================"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "服务检查第 $RETRY_COUNT/$MAX_RETRY 次失败，继续等待..."
        sleep 30
    fi
done

echo ""
echo "================================================"
echo "✗ VLLM Server 启动失败，请检查日志文件:"
echo "  $LOG_FILE"
echo "================================================"

# 显示日志文件最后 50 行
echo ""
echo "日志文件最后 50 行:"
tail -n 50 "$LOG_FILE"

exit 1
