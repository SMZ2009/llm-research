#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${LLM_RESEARCH_RUNTIME:-${HOME}/llm-research-runtime}"
UV_VERSION="0.13.6"
PYTHON_VERSION="3.11.18"

mkdir -p "${RUNTIME_ROOT}/bin" "${RUNTIME_ROOT}/cache" "${RUNTIME_ROOT}/python"
export UV_UNMANAGED_INSTALL="${RUNTIME_ROOT}/bin"
export UV_CACHE_DIR="${RUNTIME_ROOT}/cache"
export UV_PYTHON_INSTALL_DIR="${RUNTIME_ROOT}/python"
export UV_PROJECT_ENVIRONMENT="${RUNTIME_ROOT}/.venv"
export UV_PYTHON_PREFERENCE="managed"

if [[ ! -x "${RUNTIME_ROOT}/bin/uv" ]]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${RUNTIME_ROOT}/uv-install-${UV_VERSION}.sh"
  sha256sum "${RUNTIME_ROOT}/uv-install-${UV_VERSION}.sh" \
    > "${RUNTIME_ROOT}/uv_installer.sha256"
  sh "${RUNTIME_ROOT}/uv-install-${UV_VERSION}.sh"
fi

"${RUNTIME_ROOT}/bin/uv" --version
"${RUNTIME_ROOT}/bin/uv" python install "${PYTHON_VERSION}"
cd "${PROJECT_ROOT}"
"${RUNTIME_ROOT}/bin/uv" sync --frozen --extra dev --python "${PYTHON_VERSION}"
"${RUNTIME_ROOT}/.venv/bin/python" --version
