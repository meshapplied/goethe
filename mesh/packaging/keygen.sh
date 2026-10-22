#!/usr/bin/env bash
# Create P-256 release.key and release_pub.pem; prod: openssl ecparam -name prime256v1 -genkey -noout -out release.key; openssl ec -in release.key -pubout -out release_pub.pem; openssl dgst -sha256 -sign release.key -out SHA256SUMS.sig SHA256SUMS.

set -euo pipefail

DIR="${1:?usage: keygen.sh <dir>}"
mkdir -p "${DIR}"
KEY="${DIR}/release.key"
PUB="${DIR}/release_pub.pem"
openssl ecparam -name prime256v1 -genkey -noout -out "${KEY}"
openssl ec -in "${KEY}" -pubout -out "${PUB}"
chmod 600 "${KEY}"
chmod 644 "${PUB}"
