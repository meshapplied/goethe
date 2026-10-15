#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
FILE="$NVFP4_HOME/models/$MODEL_FILE"

check() {
  have="$(sha256sum "$FILE" | cut -d' ' -f1)"
  [ "$have" = "$MODEL_SHA256" ] || [ "$have" = "$MODEL_PATCHED_SHA256" ] || { echo "error: unexpected sha256 $have" >&2; exit 1; }
}

mkdir -p "$NVFP4_HOME/models"
if [ ! -f "$FILE" ] || [ "$(stat -L -c %s "$FILE")" != "$MODEL_BYTES" ]; then
  curl -fL -C - -o "$FILE" "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$MODEL_FILE"
  check
elif [ "${VERIFY:-0}" != 0 ]; then
  check
fi

if [ "${NVFP4_HEAD:-1}" = 0 ]; then
  python3 "$HERE/patches/apply_head.py" "$FILE" --revert
else
  python3 "$HERE/patches/apply_head.py" "$FILE"
fi
