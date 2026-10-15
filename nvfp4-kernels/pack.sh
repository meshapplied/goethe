#!/usr/bin/env bash
# Pack ninfer-serve and libcudart.so.13 so the binary finds the runtime beside itself.
# Refuses to run until patchelf is on PATH. Does not install it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
command -v patchelf >/dev/null || { echo "error: patchelf not found" >&2; exit 1; }
src="$NVFP4_HOME/ninfer/build/apps/ninfer-serve"
[ -x "$src" ] || { echo "error: $src is not built" >&2; exit 1; }
cudart="$(ldd "$src" | awk '/libcudart\.so\.13/ { print $3; exit }')"
[ -n "${cudart:-}" ] && [ -f "$cudart" ] || { echo "error: libcudart.so.13 not found next to $src" >&2; exit 1; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/bin"
cp -a "$src" "$stage/bin/ninfer-serve"
cp -a "$cudart" "$stage/bin/libcudart.so.13"
patchelf --set-rpath '$ORIGIN' "$stage/bin/ninfer-serve"
out="${1:-$PWD/ninfer-serve-${NINFER_REV:0:12}-cu13-linux-x86_64.tar}"
tar -C "$stage" -cf "$out" bin
echo "$out"
