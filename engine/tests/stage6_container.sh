#!/usr/bin/env bash
# Stage 6 container install gate without nvidia-docker runtime.

set -euo pipefail

if [[ "${GOETHE_TEST_DOCKER:-0}" != "1" ]]; then
  echo "set GOETHE_TEST_DOCKER=1 for container gate"
  exit 75
fi

GOETHE_PKG="${GOETHE_PKG:-/scratch/ternary-flash/goethe-release/engine}"
CDN="${GOETHE_TEST_CDN:-${GOETHE_PKG}/dist/cdn}"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
WORK="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-docker-XXXXXX")"
trap 'rm -rf "$WORK"; docker rm -f goethe-stage6-test 2>/dev/null || true' EXIT

if [[ ! -f "${CDN}/install.sh" || ! -f "${CDN}/SHA256SUMS.sig" ]]; then
  echo "signed dist/cdn not built yet"
  exit 77
fi

mkdir -p "${WORK}/workspace"
NV_DEV=()
GPUS="${GOETHE_TEST_GPUS:-2,3}"
GPU_DEVS=()
for idx in ${GPUS//,/ }; do
  GPU_DEVS+=("/dev/nvidia$(nvidia-smi -q -i "$idx" | awk '/Minor Number/ {print $4}')")
done
for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset "${GPU_DEVS[@]}"; do
  [[ -e "$d" ]] && NV_DEV+=(--device "$d:$d")
done
LIB_MOUNTS=()
for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*; do
  [[ -e "$lib" ]] && LIB_MOUNTS+=(-v "$lib:$lib:ro")
done

docker run --rm --name goethe-stage6-test \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "${WORK}/workspace:/workspace" \
  -v "${CDN}:/cdn:ro" \
  "${NV_DEV[@]}" \
  "${LIB_MOUNTS[@]}" \
  ubuntu:22.04 \
  bash -lc 'trap "chown -R $HOST_UID:$HOST_GID /workspace" EXIT; apt-get update -qq && apt-get install -y -qq curl ca-certificates openssl && export GOETHE_HOME=/workspace/.goethe GOETHE_PKG_URL=file:///cdn && bash /cdn/install.sh && test -d /workspace/.goethe && pgrep goethed && pgrep nginx && goethe allowlist add 127.0.0.1/32 && goethe key create --name c --raw >/tmp/key.txt && curl -k -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(grep goethe_sk_ /tmp/key.txt | head -1)" https://127.0.0.1:8443/v1/models | grep -q 200'

echo "stage6_container ok"
