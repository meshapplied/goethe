#!/usr/bin/env bash
# One command: build NInfer if needed, fetch and patch the model if needed, start the server. Arguments go to serve.sh.
set -euo pipefail
cd "$(dirname "$0")"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
[ -x "$NVFP4_HOME/ninfer/build/apps/ninfer-serve" ] || ./install.sh
[ -f "$NVFP4_HOME/models/qwen3_8_27b_nvfp4.ninfer" ] || ./fetch_model.sh
exec ./serve.sh "$@"
