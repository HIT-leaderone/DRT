#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ====== 可配置项 ======
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/eval_vllm_qwen3vl_time.py}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/Qwen3-VL-8B-Instruct}"

# 数据集（可按需修改）
DATASETS=(
  "MathVista_MINI"
  "MathVerse_MINI"
  "LogicVista"
  "Video_Holmes"
  "GSM8K"
)

# prompt 与 setting
PROMPTS=(
  "directly-answer"
  "short-cot-image"
  "custom-prompt"
)

SETTINGS=(
  "sequential"
  "parallel"
)

# 并行线程数（仅 parallel 生效）
PARALLEL_WORKERS="${PARALLEL_WORKERS:-32}"

# 生成参数（与你脚本 argparse 对齐；可按需覆盖）
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"

# 输出根目录：按时间戳分目录，避免覆盖
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results/run_${TS}}"
mkdir -p "${OUT_ROOT}"

# ====== 健康检查 ======
echo "[INFO] Checking vLLM health: ${VLLM_URL}/health"
curl -fsS "${VLLM_URL}/health" >/dev/null
echo "[INFO] vLLM OK"

echo "[INFO] Output root: ${OUT_ROOT}"
echo

# ====== 执行组合评测 ======
for setting in "${SETTINGS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    out_dir="${OUT_ROOT}/${setting}/${prompt}"
    mkdir -p "${out_dir}"

    echo "============================================================"
    echo "[RUN] setting=${setting} prompt=${prompt}"
    echo "[OUT] ${out_dir}"
    echo "============================================================"

    cmd=(python3 "${SCRIPT}"
      --vllm-url "${VLLM_URL}"
      --model "${MODEL}"
      --output-dir "${out_dir}"
      --setting "${setting}"
      --prompt-type "${prompt}"
      --tokenizer-path "${TOKENIZER_PATH}"
      --temperature "${TEMPERATURE}"
      --max-tokens "${MAX_TOKENS}"
      --top-p "${TOP_P}"
      --top-k "${TOP_K}"
      --repetition-penalty "${REPETITION_PENALTY}"
      --presence-penalty "${PRESENCE_PENALTY}"
      --datasets "${DATASETS[@]}"
    )

    # parallel 专属参数
    if [[ "${setting}" == "parallel" ]]; then
      cmd+=(--parallel-workers "${PARALLEL_WORKERS}")
    fi

    # 打印并执行（同时落日志）
    log="${out_dir}/run.log"
    echo "[CMD] ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "${log}"

    echo
  done
done

echo "[DONE] All runs finished. Results in: ${OUT_ROOT}"
