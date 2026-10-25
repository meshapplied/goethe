#!/usr/bin/env bash
# Assemble engine/dist/cdn: patched SGLang wheel, pinned uv, portable binaries, hashed locks, BUILDINFO, signature.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
DIST="${ROOT}/dist/cdn"
FORK="${SGLANG_FORK_SRC:-/home/admin/proj/ternary-flash/ternary-kernels/serve/repos/sglang}"
UV="${UV:-uv}"
UV_VERSION="0.13.1"
declare -A UV_SHA256=(
  [x86_64]="eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220"
  [aarch64]="9041c73c99fdc5892cf128140beb354eca9231d502068cc36e6cb53a8e59ad14"
)
WITH_TERNARY=0
RELEASE=0
RELEASE_PUB="${GOETHE_RELEASE_PUB:-${ROOT}/packaging/release_pub.pem}"
ARCHES="x86_64,aarch64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ternary) WITH_TERNARY=1 ;;
    --release) RELEASE=1 ;;
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
IFS=',' read -r -a ARCH_LIST <<<"${ARCHES}"
for a in "${ARCH_LIST[@]}"; do
  if [[ -z "${UV_SHA256[${a}]:-}" ]]; then
    echo "unsupported arch in --arch: ${a}" >&2
    exit 1
  fi
done

pub_fingerprint() {
  openssl pkey -pubin -in "$1" -outform DER | sha256sum | awk '{print $1}'
}

check_signing() {
  # Fail fast: release builds need a signing key and the expected public-key fingerprint of release_pub.pem.
  local pub="${RELEASE_PUB}" fpr
  if [[ ! -f "${pub}" ]]; then
    echo "missing ${pub}" >&2
    exit 1
  fi
  fpr="$(pub_fingerprint "${pub}")"
  if [[ "${RELEASE}" -eq 1 && "${RELEASE_PUB}" != "${ROOT}/packaging/release_pub.pem" ]]; then
    echo "--release always embeds packaging/release_pub.pem; unset GOETHE_RELEASE_PUB" >&2
    exit 1
  fi
  if [[ "${RELEASE}" -eq 1 && ( -z "${GOETHE_SIGN_KEY:-}" || -z "${GOETHE_RELEASE_PUB_FPR:-}" ) ]]; then
    echo "--release needs GOETHE_SIGN_KEY and GOETHE_RELEASE_PUB_FPR (expected sha256 of the DER public key)" >&2
    exit 1
  fi
  if [[ -n "${GOETHE_RELEASE_PUB_FPR:-}" && "${GOETHE_RELEASE_PUB_FPR}" != "${fpr}" ]]; then
    echo "${pub} fingerprint ${fpr} != GOETHE_RELEASE_PUB_FPR ${GOETHE_RELEASE_PUB_FPR}" >&2
    exit 1
  fi
  if [[ -z "${GOETHE_SIGN_KEY:-}" && "${GOETHE_ALLOW_UNSIGNED:-}" != "1" ]]; then
    echo "set GOETHE_SIGN_KEY or GOETHE_ALLOW_UNSIGNED=1" >&2
    exit 1
  fi
}

write_buildinfo() {
  # Record exactly what went into this CDN; a dirty fork tree is shipped but flagged loudly.
  local fork_head fork_dirty=0 fork_diff untracked engine_dirty=0
  fork_head="$(git -C "${FORK}" rev-parse HEAD)"
  fork_diff="$(git -C "${FORK}" diff HEAD | sha256sum | awk '{print $1}')"
  untracked="$(git -C "${FORK}" status --porcelain --untracked-files=all | grep -c '^??' || true)"
  if [[ -n "$(git -C "${FORK}" status --porcelain)" ]]; then
    fork_dirty=1
  fi
  if [[ -n "$(git -C "${REPO}" status --porcelain -- "${ROOT}")" ]]; then
    engine_dirty=1
  fi
  {
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "engine_commit=$(git -C "${REPO}" rev-parse HEAD)"
    echo "engine_dirty=${engine_dirty}"
    echo "fork_head=${fork_head}"
    echo "fork_dirty=${fork_dirty}"
    echo "fork_diff_sha256=${fork_diff}"
    echo "fork_untracked_files=${untracked}"
    echo "patch_sha256=$(sha256sum "${ROOT}/sglang/goethe-marlin.patch" | awk '{print $1}')"
    echo "uv_version=${UV_VERSION}"
    echo "rust_toolchain=$(sed -n 's/^channel = "\(.*\)"/\1/p' "${ROOT}/rust-toolchain.toml")"
    echo "arches=${ARCHES}"
    echo "release=${RELEASE}"
  } >"${DIST}/BUILDINFO"
  if [[ "${fork_dirty}" -eq 1 ]]; then
    {
      echo "################################################################"
      echo "WARNING: SGLang fork ${FORK} has local changes (diff ${fork_diff:0:12}, ${untracked} untracked files)."
      echo "WARNING: the wheel is built from that working tree, not from commit ${fork_head:0:12}."
      echo "################################################################"
    } >&2
  fi
}

check_signing
if [[ ! -d "${FORK}/python" ]]; then
  echo "missing SGLANG_FORK_SRC python at ${FORK}" >&2
  exit 1
fi
rm -rf "${DIST}"
mkdir -p "${DIST}/wheels" "${DIST}/pyenv" "${DIST}/v1" "${DIST}/goethe"
for a in "${ARCH_LIST[@]}"; do
  mkdir -p "${DIST}/bin/linux-${a}"
done
write_buildinfo

BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "${BUILD_TMP}"' EXIT
rsync -a --exclude .git --exclude /python/build/ --exclude /python/dist/ --exclude '/python/*.egg-info/' \
  "${FORK}/" "${BUILD_TMP}/sglang/"
if ! "${ROOT}/sglang/apply.sh" "${BUILD_TMP}/sglang"; then
  echo "error: sglang patch failed; refusing to build an unpatched wheel" >&2
  exit 1
fi
WHEEL_VERSION="0.0.0.dev1+g$(git -C "${FORK}" rev-parse --short=9 HEAD)"
if [[ -n "$(git -C "${FORK}" status --porcelain)" ]]; then
  WHEEL_VERSION="${WHEEL_VERSION}.d$(date -u +%Y%m%d)"
fi
(
  cd "${BUILD_TMP}/sglang/python"
  SETUPTOOLS_SCM_PRETEND_VERSION="${WHEEL_VERSION}" SGLANG_BUILD_RUST_EXTS=none "${UV}" build --wheel -o "${DIST}/wheels"
)
rm -f "${DIST}/wheels/.gitignore"
(cd "${DIST}/wheels" && ls -t sglang-*.whl | head -n1 >LATEST)

if [[ "${WITH_TERNARY}" -eq 1 ]]; then
  if [[ ! -d "${REPO}/ternary-kernels" ]]; then
    echo "missing ${REPO}/ternary-kernels for --with-ternary" >&2
    exit 1
  fi
  tar -C "${REPO}" --exclude='*.egg-info' --exclude='.pytest_cache' --exclude='__pycache__' -cf - ternary-kernels \
    | zstd -19 -T0 -o "${DIST}/goethe/ternary-kernels.tar.zst"
fi

for a in "${ARCH_LIST[@]}"; do
  tarball="${BUILD_TMP}/uv-${a}.tar.gz"
  curl -fsSL -o "${tarball}" \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${a}-unknown-linux-gnu.tar.gz"
  echo "${UV_SHA256[${a}]}  ${tarball}" | sha256sum -c --quiet
  tar -xzf "${tarball}" -C "${DIST}/bin/linux-${a}" --strip-components=1 "uv-${a}-unknown-linux-gnu/uv"
  chmod +x "${DIST}/bin/linux-${a}/uv"
  GOETHE_ARCH="${a}" "${ROOT}/packaging/build_bin.sh"
  cp "${ROOT}/target-portable/${a}-unknown-linux-gnu/release/goethe" \
    "${ROOT}/target-portable/${a}-unknown-linux-gnu/release/goethed" "${DIST}/bin/linux-${a}/"
done

cp "${ROOT}/cdn/goethed.service" "${DIST}/goethed.service"
cp "${ROOT}/../LICENSE" "${ROOT}/../THIRD_PARTY_NOTICES.md" "${ROOT}/../THIRD_PARTY_LICENSES.txt" "${DIST}/"
cp "${ROOT}/cdn/v1/registry.json" "${DIST}/v1/registry.json"
cp "${ROOT}/pyenv/requirements.lock" "${ROOT}/pyenv/requirements.aarch64.lock" "${ROOT}/pyenv/bootstrap.sh" \
  "${DIST}/pyenv/"
"${ROOT}/packaging/subst_install_pubkey.sh" "${ROOT}/cdn/install.sh" "${RELEASE_PUB}" \
  "${DIST}/install.sh"
if [[ -n "${GOETHE_SIGN_KEY:-}" ]]; then
  "${ROOT}/packaging/sign_cdn.sh" "${DIST}" "${GOETHE_SIGN_KEY}"
fi
echo "CDN assembled at ${DIST}"
