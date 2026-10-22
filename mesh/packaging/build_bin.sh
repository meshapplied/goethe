#!/usr/bin/env bash
# Build portable mesh and meshd (glibc 2.31 baseline) in Ubuntu 20.04.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${MESH_ARCH:-x86_64}"
CACHE="${MESH_BUILD_CACHE:-${HOME}/.cache/mesh-build}"
mkdir -p "${CACHE}/cargo" "${CACHE}/rustup" "${ROOT}/target-portable"

case "${ARCH}" in
  x86_64) TRIPLE=x86_64-unknown-linux-gnu ;;
  aarch64) TRIPLE=aarch64-unknown-linux-gnu ;;
  *)
    echo "unsupported MESH_ARCH=${ARCH}" >&2
    exit 1
    ;;
esac

docker run --rm --name "mesh-build-portable-${ARCH}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e MESH_ARCH="${ARCH}" -e MESH_TRIPLE="${TRIPLE}" \
  -e CARGO_HOME=/cache/cargo -e RUSTUP_HOME=/cache/rustup -e CARGO_TARGET_DIR=/src/target-portable \
  -v "${ROOT}:/src" -v "${CACHE}:/cache" -w /src \
  ubuntu:20.04 bash -c '
    trap "chown -R $HOST_UID:$HOST_GID /src/target-portable /src/Cargo.lock /cache" EXIT
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates build-essential pkg-config file binutils >/dev/null
    if [[ "$MESH_ARCH" == aarch64 ]]; then
      apt-get install -y -qq gcc-aarch64-linux-gnu libc6-dev-arm64-cross binutils-aarch64-linux-gnu >/dev/null
      export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
      export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    fi
    [ -x /cache/cargo/bin/cargo ] || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/dev/null
    . /cache/cargo/env
    if [[ "$MESH_ARCH" == aarch64 ]]; then
      rustup target add aarch64-unknown-linux-gnu >/dev/null
    fi
    cargo build --release --locked --target "$MESH_TRIPLE"
    OUT="/src/target-portable/$MESH_TRIPLE/release"
    for bin in mesh meshd; do
      file "$OUT/$bin" | grep -q "ELF 64-bit"
      if [[ "$MESH_ARCH" == aarch64 ]]; then
        file "$OUT/$bin" | grep -q "ARM aarch64"
        GLIBC=$(aarch64-linux-gnu-objdump -T "$OUT/$bin" | sed -n "s/.*GLIBC_\\([0-9.]*\\).*/\\1/p" | sort -V | tail -n1)
      else
        file "$OUT/$bin" | grep -q "x86-64"
        GLIBC=$(objdump -T "$OUT/$bin" | sed -n "s/.*GLIBC_\\([0-9.]*\\).*/\\1/p" | sort -V | tail -n1)
      fi
      if [[ -n "${GLIBC}" ]]; then
        IFS=. read -r gmaj gmin _ <<<"${GLIBC}"
        if (( gmaj > 2 || (gmaj == 2 && gmin > 31) )); then
          echo "glibc ${GLIBC} > 2.31 for ${bin}" >&2
          exit 1
        fi
      fi
    done
  '
