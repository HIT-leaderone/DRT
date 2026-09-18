#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-Senqiao/VisionThink-Efficient}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/../results}"
TP_SIZE="${TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_IMAGES="${MAX_IMAGES:-64}"
GPU_UTIL="${GPU_UTIL:-0.7}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-2025-04-14}"
API_NPROC="${API_NPROC:-4}"

python3 "${SCRIPT_DIR}/run_vlmeval_bridge.py" \
  --model-path "${MODEL_PATH}" \
  --work-dir "${WORK_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-images "${MAX_IMAGES}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --judge "${JUDGE_MODEL}" \
  --api-nproc "${API_NPROC}" \
  --data MathVista_MINI MathVerse_MINI LogicVista GSM8K Video_Holmes
