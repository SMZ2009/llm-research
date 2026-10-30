#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUN_ID=qwen3_0p6b_sft_dolly15k_s42_20261030
RUNTIME_ROOT="${LLM_RESEARCH_RUNTIME:-${HOME}/llm-research-runtime}"
RUN_DIR="${RUNTIME_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "$$" > "${RUN_DIR}/launcher.pid"
date --iso-8601=seconds > "${RUN_DIR}/launcher_started.txt"
export HF_HUB_OFFLINE=1
export HF_HOME="${RUNTIME_ROOT}/huggingface"
export PYTHONUNBUFFERED=1
trap 'code=$?; printf "%s\n" "$code" > "${RUN_DIR}/exit_code.txt"; date --iso-8601=seconds > "${RUN_DIR}/launcher_finished.txt"' EXIT
"${RUNTIME_ROOT}/.venv/bin/python" -m llm_research.train \
  --config configs/${RUN_ID}.yaml \
  --run-id "${RUN_ID}" \
  >> "${RUN_DIR}/train.log" 2>&1
