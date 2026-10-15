#!/usr/bin/env bash
set -euo pipefail
tarfile="${1:?pack.sh tarball}"
[ -f "$tarfile" ] || { echo "error: $tarfile not found" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
root="$(mktemp -d)"
trap 'rm -rf "$root"' EXIT
tar -C "$root" -xf "$tarfile"
cp -a "$HERE/../goethe" "$HERE/../nvfp4-kernels" "$HERE/../Dockerfile" "$root/"
podman build -f "$root/Dockerfile" -t goethe:pack "$root"
echo "goethe:pack"
