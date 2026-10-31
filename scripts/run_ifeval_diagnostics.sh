#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUNTIME_ROOT="${LLM_RESEARCH_RUNTIME:-${HOME}/llm-research-runtime}"
ID=qwen3_0p6b_sft_dolly15k_s42_20261030
OUT="reports/local/${ID}/investigation"
MODEL_PATH="${RUNTIME_ROOT}/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTER="models/adapters/${ID}"
PY="${RUNTIME_ROOT}/.venv/bin/python"
export HF_HUB_OFFLINE=1
export HF_HOME="${RUNTIME_ROOT}/huggingface"
export NLTK_DATA="${RUNTIME_ROOT}/nltk_data"
export PYTHONUNBUFFERED=1
echo "$$" > "${OUT}/launcher.pid"
date --iso-8601=seconds > "${OUT}/started.txt"
trap 'code=$?; printf "%s\n" "$code" > "${OUT}/exit_code.txt"; date --iso-8601=seconds > "${OUT}/finished.txt"' EXIT
"${PY}" scripts/generate_ifeval.py --input "${OUT}/input.jsonl" \
  --output "${OUT}/greedy_1024.jsonl" --model-path "${MODEL_PATH}" \
  --adapter "${ADAPTER}" --max-new-tokens 1024 >> "${OUT}/diagnostic.log" 2>&1
"${PY}" scripts/generate_ifeval.py --input "${OUT}/input.jsonl" \
  --output "${OUT}/sampling_512.jsonl" --model-path "${MODEL_PATH}" \
  --adapter "${ADAPTER}" --max-new-tokens 512 --sampling --seed 42 \
  >> "${OUT}/diagnostic.log" 2>&1
for condition in adapter_512 greedy_1024 sampling_512; do
  "${RUNTIME_ROOT}/ifeval-venv/bin/python" scripts/score_ifeval.py \
    --official-root data/raw/ifeval_official/instruction_following_eval \
    --input "${OUT}/input.jsonl" \
    --base "${OUT}/base_512.jsonl" --adapter "${OUT}/${condition}.jsonl" \
    --output "${OUT}/${condition}_score" >> "${OUT}/diagnostic.log" 2>&1
done
