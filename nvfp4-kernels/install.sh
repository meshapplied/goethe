#!/usr/bin/env bash
# Clone NInfer at the pinned revision and build it. Nothing is vendored in this repository.
# Needs: Linux, RTX 5090 (sm_120a), CUDA toolkit >= 13.1 (nvcc on PATH or CUDA_HOME),
# cmake >= 3.28, ninja, a C++20 compiler, pkg-config, libcurl, FFmpeg dev libraries.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
SRC="$NVFP4_HOME/ninfer"

die() { echo "error: $*" >&2; exit 1; }
[ -n "${CUDA_HOME:-}" ] && PATH="$CUDA_HOME/bin:$PATH"
for tool in git cmake ninja nvcc; do
  command -v "$tool" >/dev/null || die "$tool not found"
done
echo "nvcc: $(nvcc --version | tail -n 1)"

mkdir -p "$NVFP4_HOME"
[ -d "$SRC/.git" ] || git clone "$NINFER_REPO" "$SRC"
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$NINFER_REV"

cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" -j "${JOBS:-$(nproc)}"
echo "built $SRC/build/apps/ninfer-serve at $(git -C "$SRC" rev-parse --short HEAD)"
