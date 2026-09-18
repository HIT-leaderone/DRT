#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/../results/thinkless_formal_smoke}"
TOKEN_BUDGET="${TOKEN_BUDGET:-300}"
LIMIT="${LIMIT:-1}"
JUDGE="${JUDGE:-gpt-4.1-2025-04-14}"

DATASETS=(
  MathVista_MINI
  MathVerse_MINI
  LogicVista
  GSM8K
  Video_Holmes
)

for model in Qwen3-VL-8B-Instruct Qwen3-VL-8B-Thinking; do
  python "${ROOT_DIR}/run_thinkless_formal.py" \
    --model "${model}" \
    --data "${DATASETS[@]}" \
    --work-dir "${WORK_DIR}" \
    --token-budget "${TOKEN_BUDGET}" \
    --limit "${LIMIT}" \
    --judge "${JUDGE}"
done
