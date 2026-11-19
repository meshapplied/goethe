#!/usr/bin/env bash
# Embed packaging/release_pub.pem into the cdn/install.sh pubkey placeholder.

set -euo pipefail

TEMPLATE="${1:?usage: subst_install_pubkey.sh <install.sh> <release_pub.pem> <out>}"
PUB="${2:?usage: subst_install_pubkey.sh <install.sh> <release_pub.pem> <out>}"
OUT="${3:?usage: subst_install_pubkey.sh <install.sh> <release_pub.pem> <out>}"
MARKER="__GOETHE_RELEASE_PUB_PEM__"
if ! grep -q "${MARKER}" "${TEMPLATE}"; then
  echo "missing ${MARKER} in ${TEMPLATE}" >&2
  exit 1
fi
python3 - "${TEMPLATE}" "${PUB}" "${OUT}" "${MARKER}" <<'PY'
import sys
from pathlib import Path

template, pub, out, marker = sys.argv[1:5]
body = Path(pub).read_text().strip()
text = Path(template).read_text()
if marker not in text:
    raise SystemExit(f"missing {marker}")
Path(out).write_text(text.replace(marker, body, 1))
PY
