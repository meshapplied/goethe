#!/usr/bin/env bash
# Download the model, check its hash, apply the proposal-head patch. NVFP4_HEAD=0 leaves (or puts back) the original head.
# Run install.sh first. The patch needs python3 with torch and numpy.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
FILE="$NVFP4_HOME/models/$MODEL_FILE"

mkdir -p "$NVFP4_HOME/models"
if [ ! -f "$FILE" ] || [ "$(stat -L -c %s "$FILE")" != "$MODEL_BYTES" ]; then
  command -v hf >/dev/null || { echo "error: hf not found (pip install -U huggingface_hub)" >&2; exit 1; }
  hf download "$MODEL_REPO" "$MODEL_FILE" --revision "$MODEL_REV" --local-dir "$NVFP4_HOME/models"
fi
have="$(sha256sum "$FILE" | cut -d' ' -f1)"
[ "$have" = "$MODEL_SHA256" ] || [ "$have" = "$MODEL_PATCHED_SHA256" ] || { echo "error: unexpected sha256 $have" >&2; exit 1; }

python3 "$HERE/patches/apply_head.py" "$FILE" $([ "${NVFP4_HEAD:-1}" = 0 ] && echo --revert)
