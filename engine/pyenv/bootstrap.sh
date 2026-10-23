#!/usr/bin/env bash
# Idempotent hermetic Python 3.11 py-env under GOETHE_HOME from a hash-locked requirements file plus the SGLang fork.

set -euo pipefail

if [[ -z "${GOETHE_HOME:-}" ]]; then
  echo "GOETHE_HOME must be set" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${GOETHE_ARCH:-$(uname -m)}" in
  aarch64 | arm64) LOCK="${SCRIPT_DIR}/requirements.aarch64.lock" ;;
  *) LOCK="${SCRIPT_DIR}/requirements.lock" ;;
esac
VENV="${GOETHE_HOME}/py-env"
FORK_ROOT="${SGLANG_FORK_SRC:-${GOETHE_HOME}/sglang-fork}"
UV="${UV:-uv}"

if [[ -z "${GOETHE_SGLANG_WHEEL:-}" && ! -d "${FORK_ROOT}/python" ]]; then
  echo "set GOETHE_SGLANG_WHEEL or SGLANG_FORK_SRC with a python/ dir (got ${FORK_ROOT})" >&2
  exit 1
fi
if [[ ! -f "${LOCK}" ]]; then
  echo "Missing requirements lock at ${LOCK}" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" || "${GOETHE_PYENV_RESET:-0}" == "1" ]]; then
  "${UV}" venv "${VENV}" --python 3.11 --clear
fi
PREFIX="$("${VENV}/bin/python" -c 'import sys; print(sys.prefix)')"
if [[ "$(cd "${PREFIX}" && pwd -P)" != "$(cd "${VENV}" && pwd -P)" ]]; then
  echo "refusing to install: ${VENV}/bin/python runs the environment at ${PREFIX}, not ${VENV}" >&2
  exit 1
fi

"${UV}" pip install --python "${VENV}/bin/python" --require-hashes -r "${LOCK}"
if [[ -n "${GOETHE_SGLANG_WHEEL:-}" ]]; then
  "${UV}" pip install --python "${VENV}/bin/python" --no-deps "${GOETHE_SGLANG_WHEEL}"
else
  SGLANG_BUILD_RUST_EXTS=none "${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${FORK_ROOT}/python"
fi

echo "Bootstrap complete: ${VENV}"
