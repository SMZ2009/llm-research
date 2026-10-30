#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUN_ID=qwen3_0p6b_sft_dolly15k_s42_20261030
RUNTIME_ROOT="${LLM_RESEARCH_RUNTIME:-${HOME}/llm-research-runtime}"
RUN_DIR="${RUNTIME_ROOT}/runs/${RUN_ID}"
OUT="reports/local/${RUN_ID}"
MODEL_PATH="${RUNTIME_ROOT}/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTER="models/adapters/${RUN_ID}"
IFEVAL="data/raw/ifeval_official/instruction_following_eval/data/input_data.jsonl"
OFFICIAL_ROOT="data/raw/ifeval_official/instruction_following_eval"
mkdir -p "${OUT}"
echo "$$" > "${OUT}/evaluator.pid"
date --iso-8601=seconds > "${OUT}/evaluator_started.txt"
trap 'code=$?; printf "%s\n" "$code" > "${OUT}/eval_exit_code.txt"; date --iso-8601=seconds > "${OUT}/evaluator_finished.txt"' EXIT
while [[ ! -f "${RUN_DIR}/exit_code.txt" ]]; do sleep 30; done
if [[ "$(cat "${RUN_DIR}/exit_code.txt")" != 0 ]]; then
  echo "Training failed; evaluation skipped" > "${OUT}/eval_error.txt"
  exit 1
fi
export HF_HUB_OFFLINE=1
export HF_HOME="${RUNTIME_ROOT}/huggingface"
export NLTK_DATA="${RUNTIME_ROOT}/nltk_data"
export PYTHONUNBUFFERED=1
PY="${RUNTIME_ROOT}/.venv/bin/python"
"${PY}" scripts/generate_ifeval.py --input "${IFEVAL}" \
  --output "${OUT}/ifeval_base_responses.jsonl" --model-path "${MODEL_PATH}" \
  --max-new-tokens 512 >> "${OUT}/eval.log" 2>&1
"${PY}" scripts/generate_ifeval.py --input "${IFEVAL}" \
  --output "${OUT}/ifeval_adapter_responses.jsonl" --model-path "${MODEL_PATH}" \
  --adapter "${ADAPTER}" --max-new-tokens 512 >> "${OUT}/eval.log" 2>&1
"${RUNTIME_ROOT}/ifeval-venv/bin/python" scripts/score_ifeval.py \
  --official-root "${OFFICIAL_ROOT}" --input "${IFEVAL}" \
  --base "${OUT}/ifeval_base_responses.jsonl" \
  --adapter "${OUT}/ifeval_adapter_responses.jsonl" \
  --output "${OUT}" >> "${OUT}/eval.log" 2>&1
"${PY}" scripts/evaluate_dolly.py --input data/splits/dolly15k_s42/test.jsonl \
  --output "${OUT}/dolly_base.jsonl" --model-path "${MODEL_PATH}" \
  --max-length 1024 >> "${OUT}/eval.log" 2>&1
"${PY}" scripts/evaluate_dolly.py --input data/splits/dolly15k_s42/test.jsonl \
  --output "${OUT}/dolly_adapter.jsonl" --model-path "${MODEL_PATH}" \
  --adapter "${ADAPTER}" --max-length 1024 >> "${OUT}/eval.log" 2>&1
"${PY}" scripts/evaluate_dolly.py --input data/splits/dolly15k_s42/validation.jsonl \
  --output "${OUT}/validation_base.jsonl" --model-path "${MODEL_PATH}" \
  --max-length 1024 >> "${OUT}/eval.log" 2>&1
"${PY}" scripts/evaluate_dolly.py --input data/splits/dolly15k_s42/validation.jsonl \
  --output "${OUT}/validation_adapter.jsonl" --model-path "${MODEL_PATH}" \
  --adapter "${ADAPTER}" --max-length 1024 >> "${OUT}/eval.log" 2>&1
