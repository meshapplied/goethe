#!/usr/bin/env bash
# Stage 1 verification: py-env, goethe plugin loader, hook patch, NVFP4 layer.

set -euo pipefail

GOETHE_HOME="${GOETHE_HOME:?GOETHE_HOME must be set}"
cd "${GOETHE_HOME}"
PY="${GOETHE_HOME}/py-env/bin/python"
GOETHE_PKG="/scratch/ternary-flash/goethe-release/engine"
KERNEL_DIR="${GOETHE_KERNEL_DIR:-${GOETHE_HOME}/kernels/sm120}"
export GOETHE_HOME
GPU="${CUDA_VISIBLE_DEVICES:-2}"
LOCK="/tmp/goethe-gpu2.lock"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}; run engine/pyenv/bootstrap.sh first" >&2
  exit 1
fi

echo "== (i) load_goethe_kernels with kernel dir =="
if compgen -G "${KERNEL_DIR}/*.so" > /dev/null; then
  GOETHE_KERNEL_DIR="${KERNEL_DIR}" "${PY}" -c "
import torch
import sglang.srt.goethe_plugin as mp

assert mp.load_goethe_kernels()
names = {n for n in dir(torch.ops.goethe_kernels) if not n.startswith('_')}
assert 'fused_nvfp4_gemv' in names
assert 'custom_moe_dispatch' in names
assert mp.goethe_nvfp4_mode() in ('stock', 'marlin_all', 'marlin_small')
print(sorted(names))
"
else
  echo "FAIL: no .so in ${KERNEL_DIR}" >&2
  exit 1
fi

echo "== (ii) load_goethe_kernels negative cases =="
"${PY}" -c "
import sglang.srt.goethe_plugin as mp
import os
for key in ('GOETHE_KERNEL_DIR',):
    os.environ.pop(key, None)
assert mp.load_goethe_kernels() is False
os.environ['GOETHE_KERNEL_DIR'] = ''
assert mp.load_goethe_kernels() is False
os.environ['GOETHE_KERNEL_DIR'] = '/nonexistent/goethe-kernels-missing'
assert mp.load_goethe_kernels() is False
print('negative cases ok')
"

echo "== (iii) apply_hook.sh idempotency =="
TMP_FORK="/tmp/goethe-stage1-fork-$$"
rm -rf "${TMP_FORK}"
rsync -a "${GOETHE_HOME}/sglang-fork/" "${TMP_FORK}/"
(
  cd "${TMP_FORK}"
  patch -p1 -R --batch < "${GOETHE_PKG}/sglang/goethe-hook.patch" >/dev/null 2>&1 || true
)
"${GOETHE_PKG}/sglang/apply_hook.sh" "${TMP_FORK}"
"${GOETHE_PKG}/sglang/apply_hook.sh" "${TMP_FORK}"
rm -rf "${TMP_FORK}"

echo "== (iv) server_args CLI flag =="
"${PY}" -c "
import argparse
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
flags = {opt for action in parser._actions for opt in action.option_strings if opt}
assert '--enable-goethe-fused-gemv' in flags
print('server_args ok', sorted(f for f in flags if 'goethe' in f))
"

echo "== (v) stage1_layer.py =="
flock -w 1800 "${LOCK}" env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  GOETHE_KERNEL_DIR="${KERNEL_DIR}" GOETHE_HOME="${GOETHE_HOME}" \
  "${PY}" "${GOETHE_PKG}/tests/stage1_layer.py"

echo "stage1_hook ok"
