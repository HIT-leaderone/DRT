#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_LOG="${LOG_DIR}/thinkless_full_${TIMESTAMP}.log"
MONITOR_LOG="${LOG_DIR}/thinkless_full_${TIMESTAMP}.monitor.log"

TOKEN_BUDGET="${TOKEN_BUDGET:-300}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/../results/thinkless_formal_full}"
JUDGE="${JUDGE:-gpt-4.1-2025-04-14}"
API_NPROC="${API_NPROC:-4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"

echo "timestamp=${TIMESTAMP}" | tee -a "${MONITOR_LOG}"
echo "exp_log=${EXP_LOG}" | tee -a "${MONITOR_LOG}"
echo "monitor_log=${MONITOR_LOG}" | tee -a "${MONITOR_LOG}"
echo "token_budget=${TOKEN_BUDGET}" | tee -a "${MONITOR_LOG}"
echo "work_dir=${WORK_DIR}" | tee -a "${MONITOR_LOG}"

(
  cd "${ROOT_DIR}"
  TOKEN_BUDGET="${TOKEN_BUDGET}" \
  WORK_DIR="${WORK_DIR}" \
  JUDGE="${JUDGE}" \
  API_NPROC="${API_NPROC}" \
  bash "${ROOT_DIR}/run_full.sh"
) > "${EXP_LOG}" 2>&1 &

EXP_PID=$!
echo "exp_pid=${EXP_PID}" | tee -a "${MONITOR_LOG}"

while kill -0 "${EXP_PID}" 2>/dev/null; do
  {
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    ps -p "${EXP_PID}" -o pid,ppid,etime,%cpu,%mem,cmd
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    echo
  } >> "${MONITOR_LOG}" 2>&1
  sleep "${MONITOR_INTERVAL}"
done

wait "${EXP_PID}"
EXIT_CODE=$?
echo "exp_exit_code=${EXIT_CODE}" | tee -a "${MONITOR_LOG}"

exit "${EXIT_CODE}"
