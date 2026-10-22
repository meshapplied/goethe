#!/usr/bin/env bash
# Gate sm86 and sm120 kernel packs; optional GOETHE_ARCHS extends the matrix.

set -euo pipefail

GOETHE_PKG="${GOETHE_PKG:-/scratch/ternary-flash/goethe-release/engine}"
BUILD="${GOETHE_PKG}/kernels/build.sh"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
GOETHE_PY="/scratch/ternary-flash/goethe-home/py-env/bin/python"
LOGDIR="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-arch-XXXXXX")"
trap 'rm -rf "$LOGDIR"' EXIT
FAIL=0

if [[ -n "${GOETHE_ARCHS:-}" ]]; then
  read -r -a ARCHS <<< "${GOETHE_ARCHS//,/ }"
else
  ARCHS=(sm86 sm120)
fi

for sm in "${ARCHS[@]}"; do
  if [[ "$sm" == "sm120" ]]; then
    TMPHOME="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-kernels-XXXXXX")"
    if GOETHE_HOME="$TMPHOME" PYTHON="$GOETHE_PY" GOETHE_KERNEL_INSTALL=1 bash "$BUILD" "$sm" >"${LOGDIR}/${sm}.log" 2>&1; then
      SO="${TMPHOME}/kernels/sm120/libgoethe_kernels_sm120.so"
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
    if GOETHE_KERNEL_INSTALL=0 bash "$BUILD" "$sm" >"${LOGDIR}/${sm}.log" 2>&1; then
      echo "${sm} OK"
    else
      echo "${sm} FAIL"
      FAIL=1
    fi
  fi
done

exit "$FAIL"
