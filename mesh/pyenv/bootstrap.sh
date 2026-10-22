#!/usr/bin/env bash
# Idempotent hermetic Python 3.11 and SGLang fork install under MESH_HOME.

set -euo pipefail

if [[ -z "${MESH_HOME:-}" ]]; then
  echo "MESH_HOME must be set" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${MESH_ARCH:-$(uname -m)}"
case "${ARCH}" in
  aarch64|arm64) LOCK="${SCRIPT_DIR}/requirements.aarch64.lock" ;;
  *) LOCK="${SCRIPT_DIR}/requirements.lock" ;;
esac
VENV="${MESH_HOME}/py-env"
FORK_ROOT="${SGLANG_FORK_SRC:-${MESH_HOME}/sglang-fork}"
UV="${UV:-uv}"

if [[ -z "${MESH_SGLANG_WHEEL:-}" && ! -d "${FORK_ROOT}/python" ]]; then
  echo "SGLANG_FORK_SRC must contain python/ (got ${FORK_ROOT})" >&2
  exit 1
fi

if [[ ! -f "${LOCK}" ]]; then
  echo "Missing requirements lock at ${LOCK}" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]] || [[ "${MESH_PYENV_RESET:-0}" == "1" ]]; then
  if [[ "${MESH_PYENV_RESET:-0}" == "1" ]]; then
    "${UV}" venv "${VENV}" --python 3.11 --clear
  else
    "${UV}" venv "${VENV}" --python 3.11
  fi
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r "${LOCK}"

if [[ -n "${MESH_SGLANG_WHEEL:-}" ]]; then
  "${UV}" pip install --python "${VENV}/bin/python" --no-deps "${MESH_SGLANG_WHEEL}"
else
  SGLANG_BUILD_RUST_EXTS=none "${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${FORK_ROOT}/python"
fi

echo "Bootstrap complete: ${VENV}"
