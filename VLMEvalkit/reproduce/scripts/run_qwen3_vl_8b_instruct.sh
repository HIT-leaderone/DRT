#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/reproduce/results/vlmeval_qwen3_vl_8b_cod}"
PROMPT_TYPE="${PROMPT_TYPE:-CoD}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-2025-04-14}"
API_NPROC="${API_NPROC:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

DATASETS=(
  MathVista_MINI
  MathVerse_MINI
  LogicVista
  GSM8K
  Video_Holmes
)

export PROMPT_TYPE

cd "${REPO_ROOT}"

cmd=(
  python run.py
  --data "${DATASETS[@]}"
  --model "${MODEL}"
  --work-dir "${WORK_DIR}"
  --judge "${JUDGE_MODEL}"
  --api-nproc "${API_NPROC}"
  --use-vllm
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_ARGS} )
  cmd+=( "${extra_args[@]}" )
fi

"${cmd[@]}"
