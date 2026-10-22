#!/usr/bin/env bash
# Apply goethe-marlin.patch to an SGLang fork directory (idempotent).

set -euo pipefail

FORK_DIR="${1:?usage: apply.sh <fork_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/goethe-marlin.patch"
MARKER="${FORK_DIR}/python/sglang/srt/layers/quantization/goethe_marlin.py"
MOE_MARKER="${FORK_DIR}/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py"

if [[ -f "${MARKER}" ]] && grep -q 'def mode' "${MARKER}" \
  && [[ -f "${MOE_MARKER}" ]] && grep -q 'goethe_marlin_mode' "${MOE_MARKER}"; then
  echo "Goethe Marlin patch already applied under ${FORK_DIR}"
  exit 0
fi

if [[ ! -f "${PATCH}" ]]; then
  echo "Missing patch file ${PATCH}" >&2
  exit 1
fi

(
  cd "${FORK_DIR}"
  if ! patch -p1 --forward --batch < "${PATCH}"; then
    echo "ERROR: goethe-marlin.patch did not apply cleanly to ${FORK_DIR}" >&2
    exit 1
  fi
)

echo "Goethe Marlin patch applied under ${FORK_DIR}"
