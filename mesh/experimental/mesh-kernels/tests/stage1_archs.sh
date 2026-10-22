#!/usr/bin/env bash
# Gate sm86 and sm120 kernel packs; optional MESH_ARCHS extends the matrix.

set -euo pipefail

MESH_PKG="${MESH_PKG:-/scratch/ternary-flash/goethe-release/mesh}"
BUILD="${MESH_PKG}/kernels/build.sh"
GATE_ID="${MESH_GATE_RUN_ID:-$$}"
MESH_PY="/scratch/ternary-flash/mesh-home/py-env/bin/python"
LOGDIR="$(mktemp -d "/tmp/mesh-test-${GATE_ID}-arch-XXXXXX")"
trap 'rm -rf "$LOGDIR"' EXIT
FAIL=0

if [[ -n "${MESH_ARCHS:-}" ]]; then
  read -r -a ARCHS <<< "${MESH_ARCHS//,/ }"
else
  ARCHS=(sm86 sm120)
fi

for sm in "${ARCHS[@]}"; do
  if [[ "$sm" == "sm120" ]]; then
    TMPHOME="$(mktemp -d "/tmp/mesh-test-${GATE_ID}-kernels-XXXXXX")"
    if MESH_HOME="$TMPHOME" PYTHON="$MESH_PY" MESH_KERNEL_INSTALL=1 bash "$BUILD" "$sm" >"${LOGDIR}/${sm}.log" 2>&1; then
      SO="${TMPHOME}/kernels/sm120/libmesh_kernels_sm120.so"
      if [[ -f "$SO" ]] && command -v cuobjdump >/dev/null 2>&1; then
        ELF="$(cuobjdump --list-elf "$SO" 2>/dev/null || true)"
        if echo "$ELF" | grep -q 'sm_120' && echo "$ELF" | grep -q 'sm_121'; then
          echo "${sm} OK"
        else
          echo "${sm} FAIL (cuobjdump missing sm_120 or sm_121)"
          FAIL=1
        fi
      else
        echo "${sm} FAIL (installed .so or cuobjdump missing)"
        FAIL=1
      fi
    else
      echo "${sm} FAIL"
      FAIL=1
    fi
    rm -rf "$TMPHOME"
  else
    if MESH_KERNEL_INSTALL=0 bash "$BUILD" "$sm" >"${LOGDIR}/${sm}.log" 2>&1; then
      echo "${sm} OK"
    else
      echo "${sm} FAIL"
      FAIL=1
    fi
  fi
done

exit "$FAIL"
