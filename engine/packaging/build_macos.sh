#!/usr/bin/env bash
# Build goethe, goethed and a self-contained nginx (static OpenSSL, PCRE2) for Apple Silicon into dist/cdn/bin/darwin-aarch64.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/dist/cdn/bin/darwin-aarch64"
WORK="${WORK:-${ROOT}/target/macos-build}"
export MACOSX_DEPLOYMENT_TARGET=14.0

. "${ROOT}/packaging/nginx_versions.sh"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "build_macos.sh runs on an Apple Silicon Mac" >&2
  exit 1
fi

fetch() {
  # One source tarball at its pinned sha256, unpacked into WORK.
  local url="$1" sha="$2" file
  file="${WORK}/$(basename "${url}")"
  [[ -f "${file}" ]] || curl -fsSL -o "${file}" "${url}"
  echo "${sha}  ${file}" | shasum -a 256 -c --quiet
  tar -xzf "${file}" -C "${WORK}"
}

mkdir -p "${OUT}" "${WORK}"
(cd "${ROOT}" && cargo build --release --locked --bin goethe --bin goethed)
cp "${ROOT}/target/release/goethe" "${ROOT}/target/release/goethed" "${OUT}/"

fetch "${NGINX_URL}" "${NGINX_SHA256}"
fetch "${OPENSSL_URL}" "${OPENSSL_SHA256}"
fetch "${PCRE2_URL}" "${PCRE2_SHA256}"
(
  cd "${WORK}/nginx-${NGINX_VERSION}"
  ./configure \
    --prefix=/usr/local/goethe-nginx \
    --with-http_ssl_module \
    --with-http_v2_module \
    --with-http_realip_module \
    --with-openssl="${WORK}/openssl-${OPENSSL_VERSION}" \
    --with-openssl-opt="no-shared no-tests no-docs" \
    --with-pcre="${WORK}/pcre2-${PCRE2_VERSION}" \
    --with-pcre-jit \
    --with-cc-opt="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -O2" \
    --with-ld-opt="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}" >"${WORK}/nginx-configure.log"
  make -j"$(sysctl -n hw.ncpu)" >"${WORK}/nginx-make.log"
)
cp "${WORK}/nginx-${NGINX_VERSION}/objs/nginx" "${OUT}/nginx"
strip "${OUT}/nginx"

for b in goethe goethed nginx; do
  file "${OUT}/${b}" | grep -q "arm64" || { echo "${b} is not an arm64 binary" >&2; exit 1; }
  if otool -L "${OUT}/${b}" | tail -n +2 | grep -vqE "/usr/lib/|/System/Library/"; then
    echo "${b} links a library outside macOS:" >&2
    otool -L "${OUT}/${b}" >&2
    exit 1
  fi
done
"${OUT}/nginx" -V 2>&1 | head -3
"${OUT}/goethe" --version
echo "built ${OUT}"
