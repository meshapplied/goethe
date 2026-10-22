#!/usr/bin/env bash
# Apply mesh-hook.patch to an SGLang fork directory (idempotent).

set -euo pipefail

FORK_DIR="${1:?usage: apply_hook.sh <fork_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/mesh-hook.patch"
MARKER="${FORK_DIR}/python/sglang/srt/mesh_plugin.py"

if [[ -f "${MARKER}" ]] && grep -q 'def load_mesh_kernels' "${MARKER}"; then
  if grep -q 'enable_mesh_fused_gemv' "${FORK_DIR}/python/sglang/srt/arg_groups/fields/exec_.py" 2>/dev/null; then
    echo "Mesh hook already applied under ${FORK_DIR}"
    exit 0
  fi
fi

if [[ ! -f "${PATCH}" ]]; then
  echo "Missing patch file ${PATCH}" >&2
  exit 1
fi

(
  cd "${FORK_DIR}"
  if ! patch -p1 --forward --batch < "${PATCH}"; then
    echo "ERROR: mesh-hook.patch did not apply cleanly to ${FORK_DIR}" >&2
    exit 1
  fi
)

echo "Mesh hook applied under ${FORK_DIR}"
