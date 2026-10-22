#!/usr/bin/env bash
# Assemble mesh/dist/cdn, optionally sign, and stage multi-arch binaries.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOETHE="$(cd "${ROOT}/.." && pwd)"
DIST="${ROOT}/dist/cdn"
FORK="${SGLANG_FORK_SRC:-/home/admin/proj/ternary-flash/ternary-kernels/serve/repos/sglang}"
PY="${PYTHON:-/scratch/ternary-flash/mesh-home/py-env/bin/python}"
UV="${UV:-uv}"
WITH_TERNARY=0
ARCHES="x86_64,aarch64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ternary) WITH_TERNARY=1 ;;
    --arch)
      ARCHES="${2:?--arch needs a value}"
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

rm -rf "${DIST}"
mkdir -p "${DIST}/wheels" "${DIST}/pyenv" "${DIST}/v1" "${DIST}/goethe"
IFS=',' read -r -a ARCH_LIST <<<"${ARCHES}"
for a in "${ARCH_LIST[@]}"; do
  mkdir -p "${DIST}/bin/linux-${a}"
done

if [[ ! -d "${FORK}/python" ]]; then
  echo "missing SGLANG_FORK_SRC python at ${FORK}" >&2
  exit 1
fi
BUILD_FORK="$(mktemp -d)"
cleanup_build_fork() {
  rm -rf "${BUILD_FORK}"
}
trap cleanup_build_fork EXIT
rsync -a "${FORK}/" "${BUILD_FORK}/"
if ! "${ROOT}/sglang/apply.sh" "${BUILD_FORK}"; then
  echo "error: sglang patch failed; refusing to build an unpatched wheel" >&2
  exit 1
fi
(
  cd "${BUILD_FORK}/python"
  SGLANG_BUILD_RUST_EXTS=none "${UV}" build --wheel -o "${DIST}/wheels"
)
(cd "${DIST}/wheels" && ls -t sglang-*.whl | head -n1 > LATEST)

if [[ "${WITH_TERNARY}" -eq 1 ]]; then
  if [[ ! -d "${GOETHE}/ternary-kernels" ]]; then
    echo "missing ${GOETHE}/ternary-kernels for --with-ternary" >&2
    exit 1
  fi
  tar -C "${GOETHE}" -cf - ternary-kernels | zstd -19 -T0 -o "${DIST}/goethe/ternary-kernels.tar.zst"
fi

for a in "${ARCH_LIST[@]}"; do
  case "${a}" in
    x86_64) UV_ARCH=x86_64 ;;
    aarch64) UV_ARCH=aarch64 ;;
    *)
      echo "unsupported arch in --arch: ${a}" >&2
      exit 1
      ;;
  esac
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "${DIST}/bin/linux-${a}" --strip-components=1 "uv-${UV_ARCH}-unknown-linux-gnu/uv"
  chmod +x "${DIST}/bin/linux-${a}/uv"
done

for a in "${ARCH_LIST[@]}"; do
  MESH_ARCH="${a}" "${ROOT}/packaging/build_bin.sh"
  case "${a}" in
    x86_64) TRIPLE=x86_64-unknown-linux-gnu ;;
    aarch64) TRIPLE=aarch64-unknown-linux-gnu ;;
  esac
  cp "${ROOT}/target-portable/${TRIPLE}/release/mesh" "${ROOT}/target-portable/${TRIPLE}/release/meshd" "${DIST}/bin/linux-${a}/"
done

cp "${ROOT}/cdn/meshd.service" "${DIST}/meshd.service"
cp "${ROOT}/cdn/v1/registry.json" "${DIST}/v1/registry.json"
cp "${ROOT}/pyenv/requirements.lock" "${ROOT}/pyenv/bootstrap.sh" "${DIST}/pyenv/"

PUB="${ROOT}/packaging/release_pub.pem"
if [[ ! -f "${PUB}" ]]; then
  echo "missing ${PUB}" >&2
  exit 1
fi
"${ROOT}/packaging/subst_install_pubkey.sh" "${ROOT}/cdn/install.sh" "${PUB}" "${DIST}/install.sh"

SIGN_KEY="${MESH_SIGN_KEY:-}"
if [[ -z "${SIGN_KEY}" ]]; then
  if [[ "${MESH_ALLOW_UNSIGNED:-}" != "1" ]]; then
    echo "set MESH_SIGN_KEY or MESH_ALLOW_UNSIGNED=1" >&2
    exit 1
  fi
else
  "${ROOT}/packaging/sign_cdn.sh" "${DIST}" "${SIGN_KEY}"
fi

echo "CDN assembled at ${DIST}"
