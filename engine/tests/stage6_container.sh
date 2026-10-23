#!/usr/bin/env bash
# Stage 6 container gate: install.sh as root in Ubuntu 22.04 (no systemd, no nvidia runtime) must leave a working gateway.

set -euo pipefail

if [[ "${GOETHE_TEST_DOCKER:-0}" != "1" ]]; then
  echo "set GOETHE_TEST_DOCKER=1 for container gate"
  exit 75
fi

GOETHE_PKG="${GOETHE_PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CDN="${GOETHE_TEST_CDN:-${GOETHE_PKG}/dist/cdn}"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
NAME="goethe-stage6-${GATE_ID}"
WORK="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-docker-XXXXXX")"
trap 'docker rm -f "${NAME}" >/dev/null 2>&1 || true; rm -rf "${WORK}"' EXIT

if [[ ! -f "${CDN}/install.sh" || ! -f "${CDN}/SHA256SUMS.sig" ]]; then
  echo "GOETHE_TEST_DOCKER=1 needs a signed CDN at ${CDN} (packaging/build_cdn.sh)" >&2
  exit 1
fi
mkdir -p "${WORK}/workspace"
MOUNTS=()
GPUS="${GOETHE_TEST_GPUS:-2,3}"
for g in ${GPUS//,/ }; do
  MOUNTS+=(--device "/dev/nvidia$(nvidia-smi -q -i "${g}" | awk '/Minor Number/ {print $4}')")
done
for d in /dev/nvidiactl /dev/nvidia-uvm; do
  [[ -e "${d}" ]] && MOUNTS+=(--device "${d}")
done
for lib in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*; do
  [[ -e "${lib}" ]] && MOUNTS+=(-v "${lib}:${lib}:ro")
done

docker run --rm --name "${NAME}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "${WORK}/workspace:/workspace" -v "${CDN}:/cdn:ro" "${MOUNTS[@]}" ubuntu:22.04 bash -c '
    set -euo pipefail
    trap "chown -R ${HOST_UID}:${HOST_GID} /workspace" EXIT
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates openssl >/dev/null
    export GOETHE_HOME=/workspace/.goethe GOETHE_PKG_URL=file:///cdn
    H=/workspace/.goethe
    t0=$(date +%s)
    if ! bash /cdn/install.sh >/workspace/install.log 2>&1; then
      tail -n 30 /workspace/install.log
      exit 1
    fi
    echo "install took $(( $(date +%s) - t0 ))s"
    grep -q "Goethe install complete" /workspace/install.log
    grep -q "Mux (loopback OpenAI router)" /workspace/install.log
    "${H}/nginx-root/usr/sbin/nginx" -v
    goethe ps >/dev/null
    grep -qx "GOETHE_PKG_URL=file:///cdn" "${H}/goethe.env"
    grep -qx "GOETHE_REGISTRY_URL=file:///cdn/v1/registry.json" "${H}/goethe.env"
    test "$(stat -c %a "${H}/goethe.env")" = 600
    port=$(cat "${H}/nginx/public_port")
    test "$(curl -sk -o /dev/null -w "%{http_code}" "https://127.0.0.1:${port}/v1/models")" = 403
    goethe allowlist add 127.0.0.1/32 >/dev/null
    token=$(goethe key create --name c --raw | grep "^goethe_sk_")
    test "$(curl -sk -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${token}" "https://127.0.0.1:${port}/v1/models")" = 200
    goethe stop >/dev/null
    echo "container gateway ok on ${port}"
  '
echo "stage6_container ok"
