#!/usr/bin/env bash
# Goethe from source (default: the dev branch): curl -fsSL https://raw.githubusercontent.com/meshapplied/goethe/dev/install.sh | bash
# GOETHE_REF picks a branch; CARGO_HOME and CARGO_INSTALL_ROOT are cargo's own. The pinned engine and its Python env install on first use.
set -euo pipefail
REF="${GOETHE_REF:-dev}"
die() { echo "error: $*" >&2; exit 1; }
command -v git >/dev/null || die "git is required (sudo apt install git)"
command -v cc >/dev/null || command -v gcc >/dev/null || command -v clang >/dev/null || die "a C compiler and linker are required (sudo apt install build-essential)"
ORIG_PATH="$PATH"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
if ! command -v cargo >/dev/null; then
  echo "installing the Rust toolchain (rustup, minimal profile) ..."
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
echo "building goethe and goethed from $REF ..."
cargo install --quiet --locked --force --git https://github.com/meshapplied/goethe --branch "$REF" goethe goethed
BIN="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin"
echo "installed $BIN/goethe and $BIN/goethed"
case ":$ORIG_PATH:" in *":$BIN:"*) ;; *) echo "add $BIN to your PATH: export PATH=\"$BIN:\$PATH\"" ;; esac
"$BIN/goethe" --version
