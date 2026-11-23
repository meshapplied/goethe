#!/usr/bin/env bash
# An unsigned x86_64 release-shaped tree from this checkout for the container image (portable goethe and goethed, goethe's nginx, pinned uv, the Ask model); for CI and local image tests, never for publishing.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: build_image_tree.sh OUT_DIR}"
UV_VERSION="0.13.1"
UV_SHA256="eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220"
BIN="${OUT}/bin/linux-x86_64"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

bash "${ROOT}/packaging/build_bin.sh"
bash "${ROOT}/packaging/build_nginx_linux.sh"
mkdir -p "${BIN}" "${OUT}/ask"
cp "${ROOT}/target-portable/x86_64-unknown-linux-gnu/release/goethe" "${ROOT}/target-portable/x86_64-unknown-linux-gnu/release/goethed" "${BIN}/"
cp "${ROOT}/dist/cdn/bin/linux-x86_64/nginx" "${BIN}/"
curl -fsSL -o "${TMP}/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
echo "${UV_SHA256}  ${TMP}/uv.tar.gz" | sha256sum -c --quiet
tar -xzf "${TMP}/uv.tar.gz" -C "${BIN}" --strip-components=1 "uv-x86_64-unknown-linux-gnu/uv"
python3 -I "${ROOT}/packaging/build_ask_model.py" --cache "${GOETHE_ASK_CACHE:-${ROOT}/.cache/ask-model}" --out "${OUT}/ask/ask-model-v1.bin"
cp "${ROOT}/../LICENSE" "${ROOT}/../NOTICE" "${OUT}/"
sed -n 's/^version = "\(.*\)"/\1/p' "${ROOT}/crates/goethe/Cargo.toml" | head -1 >"${OUT}/VERSION"
