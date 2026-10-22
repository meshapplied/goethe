#!/usr/bin/env bash
# Stage 6 installer checks without network: tool, glibc and driver floors, truncation safety, no kills, no nginx start.

set -euo pipefail

GOETHE_PKG="${GOETHE_PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL="${GOETHE_PKG}/cdn/install.sh"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
WORK="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-installer-XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

tool_path() {
  # Absolute path of a host tool, or exit.
  local name="$1" d
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return
  fi
  for d in /usr/bin /bin /usr/sbin /sbin; do
    if [[ -x "${d}/${name}" ]]; then
      echo "${d}/${name}"
      return
    fi
  done
  echo "missing tool ${name}" >&2
  exit 1
}

make_shim_dir() {
  # A PATH directory with the basic tools install.sh needs plus the real curl, zstd and openssl.
  local dir="$1" t
  mkdir -p "${dir}"
  for t in bash env uname sed awk tr sha256sum mv mkdir printf id head getconf ldd curl zstd openssl; do
    ln -sf "$(tool_path "${t}")" "${dir}/${t}"
  done
  echo "${dir}"
}

fake() {
  # Replace one shim tool with a script that prints a fixed version line.
  rm -f "$1/$2"
  printf '#!/bin/sh\necho "%s"\nexit 0\n' "$3" >"$1/$2"
  chmod +x "$1/$2"
}

run_install() {
  # Run install.sh with PATH=<shim>[:<tail>] against an unreachable CDN; print rc then stderr.
  local shim_dir="$1" path_tail="$2" script="${3:-${INSTALL}}" home
  home="$(mktemp -d "${WORK}/home-XXXXXX")"
  set +e
  env -i HOME="${WORK}" GOETHE_HOME="${home}/goethe" GOETHE_PKG_URL="file:///nonexistent" GOETHE_NO_LINK=1 \
    GOETHE_REQUIRE_GPU="${GOETHE_REQUIRE_GPU:-}" PATH="${shim_dir}${path_tail:+:${path_tail}}" \
    bash "${script}" 2>"${home}/stderr.txt" >/dev/null
  local rc=$?
  set -e
  echo "${rc}"
  /bin/cat "${home}/stderr.txt"
  [[ -e "${home}/goethe" ]] && echo "HOME_CREATED"
  return 0
}

expect_fail() {
  # Run a case that must fail and print pattern on stderr.
  local name="$1" pattern="$2" shim="$3" tail="$4" out rc
  out="$(run_install "${shim}" "${tail}")"
  rc="${out%%$'\n'*}"
  if [[ "${rc}" -eq 0 ]] || ! grep -q "${pattern}" <<<"${out}"; then
    echo "${name}: expected failure with '${pattern}', got rc=${rc}: ${out}"
    exit 1
  fi
}

if grep -nE '(^|[^_a-z])(kill|pkill|killall|fuser)([^_a-z]|$)' "${INSTALL}"; then
  echo "install.sh must never signal processes"
  exit 1
fi
if grep -nE 'NGINX_BIN\}?"? +-|nohup[^#]*nginx|nginx +-[cs] ' "${INSTALL}"; then
  echo "install.sh must not start or signal nginx (goethed manages it)"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-curl")"
fake "${SHIM}" curl "curl 7.40.0"
expect_fail curl "curl 7.40.0 is below minimum 7.50" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-zstd")"
fake "${SHIM}" zstd "*** Zstandard CLI (64-bit) v1.2.0"
expect_fail zstd "below minimum 1.3" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-openssl")"
fake "${SHIM}" openssl "OpenSSL 1.0.2k"
expect_fail openssl "below minimum 1.1.1" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-glibc")"
fake "${SHIM}" getconf "glibc 2.28"
expect_fail glibc "glibc 2.28 is below minimum 2.30" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-driver")"
fake "${SHIM}" nvidia-smi "575.57.08"
expect_fail driver "575.57.08 is below minimum 580" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-nogpu")"
GOETHE_REQUIRE_GPU=1 expect_fail require-gpu "nvidia-smi not found" "${SHIM}" ""

OUT="$(run_install "${SHIM}" "")"
if ! grep -q "WARNING: nvidia-smi not found" <<<"${OUT}" || grep -q "below minimum" <<<"${OUT}"; then
  echo "missing-driver warning case: ${OUT}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-trunc")"
LINES="$(wc -l <"${INSTALL}")"
for keep in $((LINES / 3)) $((LINES / 2)) $((LINES - 1)); do
  head -n "${keep}" "${INSTALL}" >"${WORK}/install-trunc.sh"
  OUT="$(run_install "${SHIM}" "/usr/bin:/bin" "${WORK}/install-trunc.sh")"
  if grep -q "HOME_CREATED" <<<"${OUT}"; then
    echo "install.sh truncated to ${keep} lines still created GOETHE_HOME: ${OUT}"
    exit 1
  fi
done

echo "stage6_installer_checks ok"
