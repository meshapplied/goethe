#!/usr/bin/env bash
# Write llama-cpp/ARCHS: the model architectures (general.architecture names) the pinned llama.cpp build serves, from src/llama-arch.cpp at its commit.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/PIN"
{
  echo "# llama.cpp ${LLAMA_CPP_TAG} (${LLAMA_CPP_COMMIT}) src/llama-arch.cpp"
  curl -fsSL "https://raw.githubusercontent.com/ggml-org/llama.cpp/${LLAMA_CPP_COMMIT}/src/llama-arch.cpp" |
    grep -oE '\{ *LLM_ARCH_[A-Z0-9_]+, *"[^"]+" *\}' | sed -E 's/.*"([^"]+)".*/\1/' | grep -vxE "clip|\(unknown\)" | LC_ALL=C sort -u
} >"${HERE}/ARCHS"
echo "$(($(wc -l <"${HERE}/ARCHS") - 1)) architectures for ${LLAMA_CPP_TAG}"
