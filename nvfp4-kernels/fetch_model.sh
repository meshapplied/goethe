#!/usr/bin/env bash
# Download the pinned artifact from Hugging Face and verify its SHA-256. Needs `hf` (pip install -U huggingface_hub).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
DEST="$NVFP4_HOME/models"

command -v hf >/dev/null || { echo "error: hf not found (pip install -U huggingface_hub)" >&2; exit 1; }
mkdir -p "$DEST"
hf download "$MODEL_REPO" "$MODEL_FILE" --revision "$MODEL_REV" --local-dir "$DEST"
printf '%s  %s\n' "$MODEL_SHA256" "$DEST/$MODEL_FILE" | sha256sum --check
