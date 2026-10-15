#!/usr/bin/env bash
set -euo pipefail
REPO="${GOETHE_REPO:-https://github.com/meshapplied/Goethe.git}"
REF="${GOETHE_REF:-main}"
SRC="${GOETHE_SRC:-${XDG_DATA_HOME:-$HOME/.local/share}/goethe/src}"
BIN="${GOETHE_BIN:-$HOME/.local/bin}"

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || { echo "error: python3 3.11 or newer is required" >&2; exit 1; }
command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }

[ -d "$SRC/.git" ] || git clone --quiet "$REPO" "$SRC"
git -C "$SRC" fetch --quiet origin "$REF"
git -C "$SRC" checkout --quiet --force FETCH_HEAD

mkdir -p "$BIN"
printf '#!/bin/sh\nexec env PYTHONPATH=%q python3 -m goethe "$@"\n' "$SRC" > "$BIN/goethe"
chmod +x "$BIN/goethe"
echo "installed $BIN/goethe"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "add $BIN to your PATH" ;; esac
