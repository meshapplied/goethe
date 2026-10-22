#!/usr/bin/env bash
# Write VERSION, SHA256SUMS and SHA256SUMS.sig for a dist/cdn tree and verify with the key embedded in install.sh.

set -euo pipefail

DIST="${1:?usage: sign_cdn.sh <dist/cdn> <signing-key>}"
KEY="${2:?usage: sign_cdn.sh <dist/cdn> <signing-key>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="$(grep -E '^version\s*=' "${ROOT}/crates/goethe/Cargo.toml" | head -n1 | sed 's/.*"\(.*\)".*/\1/')"
printf '%s\n' "${VER}" >"${DIST}/VERSION"
(
  cd "${DIST}"
  find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sig | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do
    sha256sum "${f}" | awk -v p="${f}" '{print $1 "  " p}'
  done
) >"${DIST}/SHA256SUMS"
openssl dgst -sha256 -sign "${KEY}" -out "${DIST}/SHA256SUMS.sig" "${DIST}/SHA256SUMS"
EMBEDDED="$(mktemp)"
trap 'rm -f "${EMBEDDED}"' EXIT
awk '/-----BEGIN PUBLIC KEY-----/,/-----END PUBLIC KEY-----/' "${DIST}/install.sh" \
  | sed "s/^GOETHE_RELEASE_PUB_PEM='//; s/'\$//" >"${EMBEDDED}"
if ! openssl dgst -sha256 -verify "${EMBEDDED}" -signature "${DIST}/SHA256SUMS.sig" "${DIST}/SHA256SUMS" >/dev/null; then
  echo "signing key does not match the public key embedded in ${DIST}/install.sh" >&2
  exit 1
fi
echo "signed ${DIST} (version ${VER}); verified against the key embedded in install.sh"
