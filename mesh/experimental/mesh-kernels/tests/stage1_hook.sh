#!/usr/bin/env bash
# Stage 1 verification: py-env, mesh plugin loader, hook patch, NVFP4 layer.

set -euo pipefail

MESH_HOME="${MESH_HOME:?MESH_HOME must be set}"
cd "${MESH_HOME}"
PY="${MESH_HOME}/py-env/bin/python"
MESH_PKG="/scratch/ternary-flash/goethe-release/mesh"
KERNEL_DIR="${MESH_KERNEL_DIR:-${MESH_HOME}/kernels/sm120}"
export MESH_HOME
GPU="${CUDA_VISIBLE_DEVICES:-2}"
LOCK="/tmp/mesh-gpu2.lock"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}; run mesh/pyenv/bootstrap.sh first" >&2
  exit 1
fi

echo "== (i) load_mesh_kernels with kernel dir =="
if compgen -G "${KERNEL_DIR}/*.so" > /dev/null; then
  MESH_KERNEL_DIR="${KERNEL_DIR}" "${PY}" -c "
import torch
import sglang.srt.mesh_plugin as mp

assert mp.load_mesh_kernels()
names = {n for n in dir(torch.ops.mesh_kernels) if not n.startswith('_')}
assert 'fused_nvfp4_gemv' in names
assert 'custom_moe_dispatch' in names
assert mp.mesh_nvfp4_mode() in ('stock', 'marlin_all', 'marlin_small')
print(sorted(names))
"
else
  echo "FAIL: no .so in ${KERNEL_DIR}" >&2
  exit 1
fi

echo "== (ii) load_mesh_kernels negative cases =="
"${PY}" -c "
import sglang.srt.mesh_plugin as mp
import os
for key in ('MESH_KERNEL_DIR',):
    os.environ.pop(key, None)
assert mp.load_mesh_kernels() is False
os.environ['MESH_KERNEL_DIR'] = ''
assert mp.load_mesh_kernels() is False
os.environ['MESH_KERNEL_DIR'] = '/nonexistent/mesh-kernels-missing'
assert mp.load_mesh_kernels() is False
print('negative cases ok')
"

echo "== (iii) apply_hook.sh idempotency =="
TMP_FORK="/tmp/mesh-stage1-fork-$$"
rm -rf "${TMP_FORK}"
rsync -a "${MESH_HOME}/sglang-fork/" "${TMP_FORK}/"
(
  cd "${TMP_FORK}"
  patch -p1 -R --batch < "${MESH_PKG}/sglang/mesh-hook.patch" >/dev/null 2>&1 || true
)
"${MESH_PKG}/sglang/apply_hook.sh" "${TMP_FORK}"
"${MESH_PKG}/sglang/apply_hook.sh" "${TMP_FORK}"
rm -rf "${TMP_FORK}"

echo "== (iv) server_args CLI flag =="
"${PY}" -c "
import argparse
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
flags = {opt for action in parser._actions for opt in action.option_strings if opt}
assert '--enable-mesh-fused-gemv' in flags
print('server_args ok', sorted(f for f in flags if 'mesh' in f))
"

echo "== (v) stage1_layer.py =="
flock -w 1800 "${LOCK}" env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  MESH_KERNEL_DIR="${KERNEL_DIR}" MESH_HOME="${MESH_HOME}" \
  "${PY}" "${MESH_PKG}/tests/stage1_layer.py"

echo "stage1_hook ok"
