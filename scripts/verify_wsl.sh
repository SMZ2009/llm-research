#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${LLM_RESEARCH_RUNTIME:-${HOME}/llm-research-runtime}"
REPORT_ROOT="${PROJECT_ROOT}/reports/local"
export HF_HOME="${RUNTIME_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${RUNTIME_ROOT}/datasets_cache"
export UV_CACHE_DIR="${RUNTIME_ROOT}/cache"
export UV_PROJECT_ENVIRONMENT="${RUNTIME_ROOT}/.venv"
mkdir -p "${REPORT_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

cd "${PROJECT_ROOT}"
"${RUNTIME_ROOT}/.venv/bin/python" -m pytest -q
"${RUNTIME_ROOT}/.venv/bin/ruff" check src tests scripts
"${RUNTIME_ROOT}/.venv/bin/python" -m llm_research.smoke \
  --report "${REPORT_ROOT}/smoke.json"
"${RUNTIME_ROOT}/bin/uv" pip freeze --python "${RUNTIME_ROOT}/.venv/bin/python" \
  > "${REPORT_ROOT}/installed_packages.txt"
"${RUNTIME_ROOT}/bin/uv" --version > "${REPORT_ROOT}/uv_version.txt"
sha256sum uv.lock pyproject.toml configs/sft_lora_non_thinking.yaml \
  > "${REPORT_ROOT}/project_sha256.txt"
