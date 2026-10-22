#!/usr/bin/env bash
# Write VERSION, SHA256SUMS, and SHA256SUMS.sig for a dist/cdn tree.

set -euo pipefail

DIST="${1:?usage: sign_cdn.sh <dist/cdn> <signing-key>}"
KEY="${2:?usage: sign_cdn.sh <dist/cdn> <signing-key>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="$(grep -E '^version\s*=' "${ROOT}/crates/mesh/Cargo.toml" | head -n1 | sed 's/.*"\(.*\)".*/\1/')"
printf '%s\n' "${VER}" >"${DIST}/VERSION"
(
  cd "${DIST}"
  find . -type f ! -path './SHA256SUMS' ! -path './SHA256SUMS.sig' ! -name SHA256SUMS ! -name SHA256SUMS.sig \
    | sed 's|^\./||' | sort | while read -r f; do
    sha256sum "${f}" | awk -v p="${f}" '{print $1 "  " p}'
  done
) >"${DIST}/SHA256SUMS"
openssl dgst -sha256 -sign "${KEY}" -out "${DIST}/SHA256SUMS.sig" "${DIST}/SHA256SUMS"
PUB_CHECK="$(mktemp)"
openssl ec -in "${KEY}" -pubout -out "${PUB_CHECK}" 2>/dev/null
openssl dgst -sha256 -verify "${PUB_CHECK}" -signature "${DIST}/SHA256SUMS.sig" "${DIST}/SHA256SUMS"
rm -f "${PUB_CHECK}"
