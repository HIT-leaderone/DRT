#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-vllm}
if [ "$#" -gt 0 ]; then
    shift
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRT_REPO_ROOT="${DRT_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export DRT_DATA_DIR="${DRT_DATA_DIR:-$DRT_REPO_ROOT/training_data/VisionR1_DAPO}"
export DRT_OUTPUT_DIR="${DRT_OUTPUT_DIR:-$DRT_REPO_ROOT/checkpoints/drt-rl}"
export DRT_SFT_MODEL="${DRT_SFT_MODEL:-${HF_MODEL_PATH:-leaderonehit/DRT-SFT-8B}}"
export DRT_REWARD_MODEL="${DRT_REWARD_MODEL:-${GRM_PATH:-Qwen/Qwen3-235B-A22B-Instruct-2507}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

cd "$SCRIPT_DIR"

TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
AUTO_EVAL_ENABLED="${AUTO_EVAL_ENABLED:-false}"
HDFS_CKPTS_DIR="${HDFS_CKPTS_DIR:-}"
if [ -n "$HDFS_CKPTS_DIR" ]; then
    HDFS_CKPTS_ARG="trainer.default_hdfs_dir=$HDFS_CKPTS_DIR"
else
    HDFS_CKPTS_ARG="trainer.default_hdfs_dir=null"
fi

python3 -m verl.trainer.main_ppo \
    --config-path "$DRT_REPO_ROOT/reproduce/configs" \
    --config-name drt_rl \
    actor_rollout_ref.rollout.name="$ENGINE" \
    trainer.logger="$TRAINER_LOGGER" \
    "$HDFS_CKPTS_ARG" \
    trainer.auto_eval.enabled="$AUTO_EVAL_ENABLED" \
    trainer.auto_eval.prompt_type="${PROMPT_TYPE:-Short-COT-Image}" \
    trainer.auto_eval.model_type="${MODEL_TYPE:-Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL}" \
    trainer.auto_eval.tasks="${TASKS:-[MathVista_MINI,MathVerse_MINI,LogicVista,Video_Holmes,GSM8K,MATH-500]}" \
    trainer.auto_eval.pool="${POOL:-public}" \
    "$@"
