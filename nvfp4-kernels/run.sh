#!/usr/bin/env bash
# One command: build NInfer if needed, fetch and patch the model, start the server. Arguments go to serve.sh.
set -euo pipefail
cd "$(dirname "$0")"
[ -x "${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}/ninfer/build/apps/ninfer-serve" ] || ./install.sh
./fetch_model.sh
exec ./serve.sh "$@"
