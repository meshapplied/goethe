#!/usr/bin/env bash
# Clone NInfer at the pinned commit, apply our patch, build. NVFP4_PATCHES=0 builds it unpatched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
SRC="$NVFP4_HOME/ninfer"

[ -n "${CUDA_HOME:-}" ] && PATH="$CUDA_HOME/bin:$PATH"
for tool in git cmake ninja nvcc; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
done

mkdir -p "$NVFP4_HOME"
[ -d "$SRC/.git" ] || git clone "$NINFER_REPO" "$SRC"
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet --force "$NINFER_REV"
if [ "${NVFP4_PATCHES:-1}" != 0 ]; then
  git -C "$SRC" apply "$HERE/patches/ninfer-dn-sched.patch"
  git -C "$SRC" apply "$HERE/patches/openai-tool-name-128.patch"
fi

cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" -j "${JOBS:-$(nproc)}"
echo "built $SRC/build/apps/ninfer-serve"
