#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-8}"

sanitize_colon_path() {
  local raw="${1:-}"
  local cleaned=()
  local part
  IFS=':' read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ -d "${part}" && -x "${part}" ]]; then
      cleaned+=("${part}")
    fi
  done
  local joined=""
  if [[ ${#cleaned[@]} -gt 0 ]]; then
    joined="${cleaned[0]}"
    for part in "${cleaned[@]:1}"; do
      joined="${joined}:${part}"
    done
  fi
  printf '%s' "${joined}"
}

export PATH="$(sanitize_colon_path "${PATH:-}")"
export LD_LIBRARY_PATH="$(sanitize_colon_path "${LD_LIBRARY_PATH:-}")"

vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3-VL-8B-Instruct \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}" \
  --chat-template-content-format string \
  --generation-config vllm
