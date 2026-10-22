#!/usr/bin/env bash
# Build mesh_kernels for one SM and install a single .so under MESH_HOME/kernels/<sm>/.
set -euo pipefail

SM="${1:?usage: build.sh <sm80|sm86|sm89|sm90|sm100|sm120>}"
case "$SM" in
  sm80) ARCH="8.0" ;;
  sm86) ARCH="8.6" ;;
  sm89) ARCH="8.9" ;;
  sm90) ARCH="9.0" ;;
  sm100) ARCH="10.0" ;;
  sm120) ARCH="12.0" ;;
  *) echo "unknown sm: $SM" >&2; exit 1 ;;
esac

export MESH_HOME="${MESH_HOME:-/scratch/ternary-flash/mesh-home}"
export PYTHON="${PYTHON:-$MESH_HOME/py-env/bin/python}"
export CUDA_HOME=/usr/local/cuda
unset TORCH_CUDA_ARCH_LIST
export MESH_KERNEL_SM="$SM"
export MESH_KERNEL_ARCH="$ARCH"

SRC="$(cd "$(dirname "$0")" && pwd)"
BUILD="/tmp/mesh-kernels-build"
rm -rf "$BUILD"
mkdir -p "$BUILD"
cp -a "$SRC/csrc" "$SRC/setup.py" "$SRC/pyproject.toml" "$BUILD/"
cd "$BUILD"
LOG="$BUILD/build.log"
if ! "$PYTHON" setup.py build_ext --inplace >"$LOG" 2>&1; then
  tail -40 "$LOG" >&2
  exit 1
fi
mapfile -t SO < <(find . -maxdepth 2 -name "libmesh_kernels_${SM}*.so" -type f)
if [[ ${#SO[@]} -ne 1 ]]; then
  echo "built .so not found (count=${#SO[@]})" >&2
  exit 1
fi

INSTALL="${MESH_KERNEL_INSTALL:-1}"
if [[ "$INSTALL" == "1" ]]; then
  DEST="$MESH_HOME/kernels/$SM"
  mkdir -p "$DEST"
  rm -f "$DEST"/*.so
  cp "${SO[0]}" "$DEST/libmesh_kernels_${SM}.so"
  echo "installed $DEST/libmesh_kernels_${SM}.so"
fi
rm -rf "$BUILD"
