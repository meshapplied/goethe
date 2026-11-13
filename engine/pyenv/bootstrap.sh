#!/usr/bin/env bash
# Idempotent hermetic Python 3.11 py-env under GOETHE_HOME, in hash-locked layers: base, engine, sglang (none: all).

set -euo pipefail

if [[ -z "${GOETHE_HOME:-}" ]]; then
  echo "GOETHE_HOME must be set" >&2
  exit 1
fi

LAYER="${1:-all}"
case "${LAYER}" in
  base | engine | sglang | all) ;;
  *)
    echo "usage: bootstrap.sh [base|engine|sglang]: base is the venv with the hf CLI, engine adds goethe-engine (its PIN)" \
      "and its runtime, sglang the full lock and Goethe's SGLang build; no argument installs sglang and engine" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${GOETHE_ARCH:-$(uname -m)}" in
  aarch64 | arm64) SUFFIX=".aarch64" ;;
  *) SUFFIX="" ;;
esac
VENV="${GOETHE_HOME}/py-env"
FORK_ROOT="${SGLANG_FORK_SRC:-${GOETHE_HOME}/sglang-fork}"
UV="${UV:-uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

lock_of() {
  # The arch's lock for a layer: requirements[.<layer>][.aarch64].lock next to this script.
  local path="${SCRIPT_DIR}/requirements${1:+.$1}${SUFFIX}.lock"
  if [[ ! -f "${path}" ]]; then
    echo "Missing requirements lock at ${path}" >&2
    exit 1
  fi
  echo "${path}"
}

install_lock() {
  "${UV}" pip install --python "${VENV}/bin/python" --require-hashes -r "$1"
}

install_base() {
  # The venv (created once, or again with GOETHE_PYENV_RESET=1), then the base lock.
  local lock prefix
  lock="$(lock_of base)"
  if [[ ! -x "${VENV}/bin/python" || "${GOETHE_PYENV_RESET:-0}" == "1" ]]; then
    "${UV}" venv --quiet "${VENV}" --python 3.11 --clear
  fi
  prefix="$("${VENV}/bin/python" -c 'import sys; print(sys.prefix)')"
  if [[ "$(cd "${prefix}" && pwd -P)" != "$(cd "${VENV}" && pwd -P)" ]]; then
    echo "refusing to install: ${VENV}/bin/python runs the environment at ${prefix}, not ${VENV}" >&2
    exit 1
  fi
  install_lock "${lock}"
}

engine_pin() {
  # The goethe-engine wheel URL and sha256: from the environment (goethe passes its compiled-in PIN), else the PIN file.
  local pin="${SCRIPT_DIR}/../goethe-engine/PIN"
  ENGINE_URL="${GOETHE_ENGINE_WHEEL_URL:-}"
  ENGINE_SHA="${GOETHE_ENGINE_WHEEL_SHA256:-}"
  if [[ -z "${ENGINE_URL}" && -f "${pin}" ]]; then
    ENGINE_URL="$(sed -n "s/^GOETHE_ENGINE_WHEEL_URL=//p" "${pin}")"
    ENGINE_SHA="$(sed -n "s/^GOETHE_ENGINE_WHEEL_SHA256=//p" "${pin}")"
  fi
}

install_engine() {
  # goethe-engine's runtime, then the pinned wheel itself with its hash enforced and no further dependencies.
  engine_pin
  if [[ -z "${ENGINE_URL}" || -z "${ENGINE_SHA}" ]]; then
    if [[ "${LAYER}" == all ]]; then
      return 0
    fi
    echo "no goethe-engine wheel pinned: set GOETHE_ENGINE_WHEEL_URL and GOETHE_ENGINE_WHEEL_SHA256" >&2
    exit 1
  fi
  # Next to an installed SGLang the env keeps the torch and transformers SGLang was installed with (an older py-env
  # may pin older ones than today's engine lock); only the wheel is added.
  if [[ "${LAYER}" == all ]] || ! compgen -G "${VENV}/lib/python3*/site-packages/sglang-*.dist-info" >/dev/null; then
    install_lock "$(lock_of engine)"
  fi
  ENGINE_REQ="$(mktemp)"
  trap 'rm -f "${ENGINE_REQ}"' EXIT
  echo "goethe-engine @ ${ENGINE_URL} --hash=sha256:${ENGINE_SHA}" >"${ENGINE_REQ}"
  "${UV}" pip install --python "${VENV}/bin/python" --require-hashes --no-deps -r "${ENGINE_REQ}"
}

install_sglang() {
  # The full lock, then Goethe's SGLang build: the verified wheel, else the fork checkout as an editable install.
  install_lock "$(lock_of "")"
  if [[ -n "${GOETHE_SGLANG_WHEEL:-}" ]]; then
    "${UV}" pip install --python "${VENV}/bin/python" --no-deps "${GOETHE_SGLANG_WHEEL}"
  else
    SGLANG_BUILD_RUST_EXTS=none "${UV}" pip install --python "${VENV}/bin/python" --no-deps -e "${FORK_ROOT}/python"
  fi
}

if [[ "${LAYER}" == sglang || "${LAYER}" == all ]] && [[ -z "${GOETHE_SGLANG_WHEEL:-}" && ! -d "${FORK_ROOT}/python" ]]; then
  echo "set GOETHE_SGLANG_WHEEL or SGLANG_FORK_SRC with a python/ dir (got ${FORK_ROOT})" >&2
  exit 1
fi
install_base
case "${LAYER}" in
  engine) install_engine ;;
  sglang) install_sglang ;;
  all)
    install_sglang
    install_engine
    ;;
esac

if [[ "${LAYER}" == all ]]; then
  echo "Python environment ready: ${VENV}"
else
  echo "Python environment ready: ${VENV} (${LAYER} layer)"
fi
