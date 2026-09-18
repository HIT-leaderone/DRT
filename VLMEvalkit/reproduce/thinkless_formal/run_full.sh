#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/../results/thinkless_formal_full}"
TOKEN_BUDGET="${TOKEN_BUDGET:-300}"
JUDGE="${JUDGE:-gpt-4.1-2025-04-14}"
API_NPROC="${API_NPROC:-4}"

DATASETS=(
  MathVista_MINI
  MathVerse_MINI
  LogicVista
  GSM8K
  Video_Holmes
)

MODELS=(
  Qwen3-VL-8B-Instruct
  Qwen3-VL-8B-Thinking
)

for model in "${MODELS[@]}"; do
  python "${ROOT_DIR}/run_thinkless_formal.py" \
    --model "${model}" \
    --data "${DATASETS[@]}" \
    --work-dir "${WORK_DIR}" \
    --token-budget "${TOKEN_BUDGET}" \
    --judge "${JUDGE}" \
    --api-nproc "${API_NPROC}"
done
