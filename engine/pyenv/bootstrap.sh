#!/usr/bin/env bash
# Idempotent hermetic Python 3.11 and SGLang fork install under GOETHE_HOME.

set -euo pipefail

if [[ -z "${GOETHE_HOME:-}" ]]; then
  echo "GOETHE_HOME must be set" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${GOETHE_ARCH:-$(uname -m)}"
case "${ARCH}" in
  aarch64|arm64) LOCK="${SCRIPT_DIR}/requirements.aarch64.lock" ;;
  *) LOCK="${SCRIPT_DIR}/requirements.lock" ;;
esac
VENV="${GOETHE_HOME}/py-env"
FORK_ROOT="${SGLANG_FORK_SRC:-${GOETHE_HOME}/sglang-fork}"
UV="${UV:-uv}"

if [[ -z "${GOETHE_SGLANG_WHEEL:-}" && ! -d "${FORK_ROOT}/python" ]]; then
  echo "SGLANG_FORK_SRC must contain python/ (got ${FORK_ROOT})" >&2
  exit 1
fi

if [[ ! -f "${LOCK}" ]]; then
  echo "Missing requirements lock at ${LOCK}" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]] || [[ "${GOETHE_PYENV_RESET:-0}" == "1" ]]; then
  if [[ "${GOETHE_PYENV_RESET:-0}" == "1" ]]; then
    "${UV}" venv "${VENV}" --python 3.11 --clear
  else
    "${UV}" venv "${VENV}" --python 3.11
  fi
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r "${LOCK}"

if [[ -n "${GOETHE_SGLANG_WHEEL:-}" ]]; then
  "${UV}" pip install --python "${VENV}/bin/python" --no-deps "${GOETHE_SGLANG_WHEEL}"
else
  SGLANG_BUILD_RUST_EXTS=none "${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${FORK_ROOT}/python"
fi

echo "Bootstrap complete: ${VENV}"
