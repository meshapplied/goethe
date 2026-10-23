#!/usr/bin/env bash
# Create a P-256 release.key (mode 600) and release_pub.pem in a directory, refusing to overwrite either.

set -euo pipefail

DIR="${1:?usage: keygen.sh <dir>}"
KEY="${DIR}/release.key"
PUB="${DIR}/release_pub.pem"
umask 077
for f in "${KEY}" "${PUB}"; do
  if [[ -e "${f}" ]]; then
    echo "refusing to overwrite ${f}" >&2
    exit 1
  fi
done
mkdir -p "${DIR}"
openssl ecparam -name prime256v1 -genkey -noout -out "${KEY}"
openssl ec -in "${KEY}" -pubout -out "${PUB}" 2>/dev/null
chmod 644 "${PUB}"
echo "public key fingerprint (GOETHE_RELEASE_PUB_FPR): $(openssl pkey -pubin -in "${PUB}" -outform DER | sha256sum | cut -d' ' -f1)"
