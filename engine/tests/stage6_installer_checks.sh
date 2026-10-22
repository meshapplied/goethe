#!/usr/bin/env bash
# Stage 6 installer prerequisite checks without network.

set -euo pipefail

GOETHE_PKG="${GOETHE_PKG:-/scratch/ternary-flash/goethe-release/engine}"
INSTALL="${GOETHE_PKG}/cdn/install.sh"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
WORK="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-installer-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -f "${INSTALL}" ]]; then
  echo "install.sh missing"
  exit 77
fi

tool_path() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return
  fi
  for d in /usr/bin /bin; do
    if [[ -x "${d}/${name}" ]]; then
      echo "${d}/${name}"
      return
    fi
  done
  echo "missing tool ${name}" >&2
  exit 1
}

REAL_UNAME="$(tool_path uname)"
REAL_SED="$(tool_path sed)"
REAL_AWK="$(tool_path awk)"
REAL_TR="$(tool_path tr)"
REAL_SHA256SUM="$(tool_path sha256sum)"
REAL_MV="$(tool_path mv)"
REAL_MKDIR="$(tool_path mkdir)"
REAL_PRINTF="$(tool_path printf)"
REAL_ID="$(tool_path id)"
REAL_HEAD="$(tool_path head)"
REAL_BASH="$(tool_path bash)"
REAL_ENV="$(tool_path env)"

make_shim_dir() {
  local dir="$1"
  mkdir -p "${dir}"
  ln -sf "${REAL_BASH}" "${dir}/bash"
  ln -sf "${REAL_ENV}" "${dir}/env"
  ln -sf "${REAL_UNAME}" "${dir}/uname"
  ln -sf "${REAL_SED}" "${dir}/sed"
  ln -sf "${REAL_AWK}" "${dir}/awk"
  ln -sf "${REAL_TR}" "${dir}/tr"
  ln -sf "${REAL_SHA256SUM}" "${dir}/sha256sum"
  ln -sf "${REAL_MV}" "${dir}/mv"
  ln -sf "${REAL_MKDIR}" "${dir}/mkdir"
  ln -sf "${REAL_PRINTF}" "${dir}/printf"
  ln -sf "${REAL_ID}" "${dir}/id"
  ln -sf "${REAL_HEAD}" "${dir}/head"
  echo "${dir}"
}

run_install() {
  local shim_dir="$1"
  local path_tail="/usr/bin:/bin"
  if [[ $# -ge 2 ]]; then
    path_tail="$2"
    shift 2
  else
    shift 1
  fi
  local home
  home="$(mktemp -d "${WORK}/home-XXXXXX")"
  export GOETHE_HOME="${home}"
  export GOETHE_PKG_URL="file:///nonexistent"
  if [[ -n "${path_tail}" ]]; then
    export PATH="${shim_dir}:${path_tail}"
  else
    export PATH="${shim_dir}"
  fi
  set +e
  bash "${INSTALL}" "$@" 2>"${home}/stderr.txt" >/dev/null
  local rc=$?
  set -e
  echo "${rc}"
  /bin/cat "${home}/stderr.txt"
}

REAL_CURL="$(tool_path curl)"
REAL_ZSTD="$(tool_path zstd)"
REAL_OPENSSL="$(tool_path openssl)"
REAL_NVIDIA_SMI="$(tool_path nvidia-smi)"

SHIM="$(make_shim_dir "${WORK}/shim-a")"
printf '%b' '#!/bin/sh\necho "curl 7.40.0"\nexit 0\n' >"${SHIM}/curl"
chmod +x "${SHIM}/curl"
ln -sf "${REAL_ZSTD}" "${SHIM}/zstd"
ln -sf "${REAL_OPENSSL}" "${SHIM}/openssl"
ln -sf "${REAL_NVIDIA_SMI}" "${SHIM}/nvidia-smi"
OUT="$(run_install "${SHIM}" "/usr/bin:/bin")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
if [[ "${RC}" -eq 0 ]]; then
  echo "curl case should fail"
  exit 1
fi
if ! grep -q "curl 7.40.0 is below minimum 7.50" <<<"${REST}"; then
  echo "curl case stderr: ${REST}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-b")"
ln -sf "${REAL_CURL}" "${SHIM}/curl"
printf '%b' '#!/bin/sh\necho "*** Zstandard CLI (64-bit) v1.2.0"\nexit 0\n' >"${SHIM}/zstd"
chmod +x "${SHIM}/zstd"
ln -sf "${REAL_OPENSSL}" "${SHIM}/openssl"
ln -sf "${REAL_NVIDIA_SMI}" "${SHIM}/nvidia-smi"
OUT="$(run_install "${SHIM}" "/usr/bin:/bin")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
if [[ "${RC}" -eq 0 ]]; then
  echo "zstd case should fail"
  exit 1
fi
if ! grep -q "below minimum 1.3" <<<"${REST}"; then
  echo "zstd case stderr: ${REST}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-c")"
ln -sf "${REAL_CURL}" "${SHIM}/curl"
ln -sf "${REAL_ZSTD}" "${SHIM}/zstd"
printf '%b' '#!/bin/sh\necho "OpenSSL 1.0.2k"\nexit 0\n' >"${SHIM}/openssl"
chmod +x "${SHIM}/openssl"
ln -sf "${REAL_NVIDIA_SMI}" "${SHIM}/nvidia-smi"
OUT="$(run_install "${SHIM}" "/usr/bin:/bin")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
if [[ "${RC}" -eq 0 ]]; then
  echo "openssl case should fail"
  exit 1
fi
if ! grep -q "below minimum 1.1.1" <<<"${REST}"; then
  echo "openssl case stderr: ${REST}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-d")"
ln -sf "${REAL_CURL}" "${SHIM}/curl"
ln -sf "${REAL_ZSTD}" "${SHIM}/zstd"
ln -sf "${REAL_OPENSSL}" "${SHIM}/openssl"
printf '%b' '#!/bin/sh\necho "535.104.05"\nexit 0\n' >"${SHIM}/nvidia-smi"
chmod +x "${SHIM}/nvidia-smi"
OUT="$(run_install "${SHIM}" "/usr/bin:/bin")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
if [[ "${RC}" -eq 0 ]]; then
  echo "nvidia case should fail"
  exit 1
fi
if ! grep -q "below minimum 550" <<<"${REST}"; then
  echo "nvidia case stderr: ${REST}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-e")"
ln -sf "${REAL_CURL}" "${SHIM}/curl"
ln -sf "${REAL_ZSTD}" "${SHIM}/zstd"
ln -sf "${REAL_OPENSSL}" "${SHIM}/openssl"
export GOETHE_REQUIRE_GPU=1
OUT="$(run_install "${SHIM}" "")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
unset GOETHE_REQUIRE_GPU
if [[ "${RC}" -eq 0 ]]; then
  echo "GOETHE_REQUIRE_GPU case should fail"
  exit 1
fi
if ! grep -q "nvidia-smi not found" <<<"${REST}"; then
  echo "require gpu stderr: ${REST}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-f")"
ln -sf "${REAL_CURL}" "${SHIM}/curl"
ln -sf "${REAL_ZSTD}" "${SHIM}/zstd"
ln -sf "${REAL_OPENSSL}" "${SHIM}/openssl"
OUT="$(run_install "${SHIM}" "")"
RC="${OUT%%$'\n'*}"
REST="${OUT#*$'\n'}"
if ! grep -q "WARNING: nvidia-smi not found" <<<"${REST}"; then
  echo "warn gpu stderr: ${REST}"
  exit 1
fi
if grep -q "below minimum" <<<"${REST}"; then
  echo "unexpected below minimum: ${REST}"
  exit 1
fi

echo "stage6_installer_checks ok"
