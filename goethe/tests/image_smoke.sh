#!/usr/bin/env bash
# The container image without a GPU: start, health check, headless GOETHE_SETUP, imported API key, refusals, restart and stop.

set -euo pipefail

IMAGE="${1:?usage: image_smoke.sh IMAGE}"
NAME="goethe-smoke-$$"
VOL="goethe-smoke-$$"
KEY="goethe_sk_$(openssl rand -hex 16)"
fails=0

pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { local name="$1"; shift; if "$@"; then pass "${name}"; else fail "${name}"; fi; }

cleanup() {
  docker rm -f "${NAME}" "${NAME}-bad" >/dev/null 2>&1 || true
  docker volume rm -f "${VOL}" "${VOL}-bad" >/dev/null 2>&1 || true
}
trap cleanup EXIT

start() { docker run -d --name "$1" -v "$2:/workspace" -e GOETHE_API_KEY="${KEY}" -e GOETHE_SETUP="$3" "${IMAGE}" >/dev/null; }
ready() {
  for _ in $(seq 90); do
    if docker exec "$1" goethe container --check 2>/dev/null; then return 0; fi
    sleep 1
  done
  docker logs "$1" 2>&1 | tail -20
  return 1
}
in_c() { docker exec "${NAME}" "$@"; }
quiet() { "$@" >/dev/null; }
code_for() { in_c curl -sk -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $1" "https://127.0.0.1:8443/v1/models"; }
log_has() {
  for _ in $(seq 10); do
    if docker logs "$1" 2>&1 | grep -qF "$2"; then return 0; fi
    sleep 1
  done
  return 1
}
refuses() { local out; if out="$(in_c goethe "$@" 2>&1)"; then return 1; fi; grep -q container <<<"${out}"; }
no_secret_in_log() { ! docker logs "${NAME}" 2>&1 | grep -qF "${KEY}"; }
setup_not_run() { ! docker exec "$1" test -e /workspace/.goethe/container-setup.done && ! docker exec "$1" goethe api keys --json | grep -q '"name": "smoke"'; }

start "${NAME}" "${VOL}" $'api allow add 127.0.0.1/32\napi keys create smoke-extra'
check "goethed runs and the health check passes" ready "${NAME}"
check "goethe status answers through docker exec" quiet in_c goethe status
check "GOETHE_SETUP ran its lines" log_has "${NAME}" "GOETHE_SETUP: goethe api keys create smoke-extra"
check "the imported key reaches the API" test "$(code_for "${KEY}")" = 200
check "a wrong key is refused" test "$(code_for "goethe_sk_$(openssl rand -hex 16)")" = 401
check "the key secret never shows in the log" no_secret_in_log
for c in update uninstall up down; do check "goethe ${c} refuses in the container" refuses "${c}"; done

id_before="$(in_c cat /workspace/.goethe/machine-id)"
docker restart "${NAME}" >/dev/null
check "it comes back after a restart" ready "${NAME}"
check "an unchanged GOETHE_SETUP is not run again" log_has "${NAME}" "GOETHE_SETUP: unchanged since it last ran"
check "the node id outlives the restart" test "$(in_c cat /workspace/.goethe/machine-id)" = "${id_before}"
check "the key still works after the restart" test "$(code_for "${KEY}")" = 200

start "${NAME}-bad" "${VOL}-bad" $'api keys create smoke\napi keys create'
check "a container with a bad GOETHE_SETUP still serves" ready "${NAME}-bad"
check "a bad GOETHE_SETUP is refused whole" log_has "${NAME}-bad" "GOETHE_SETUP is not run: line 2"
check "nothing of a refused GOETHE_SETUP ran" setup_not_run "${NAME}-bad"

t0="$(date +%s)"
docker stop "${NAME}" >/dev/null
check "it stops on SIGTERM before the stop timeout" test $(($(date +%s) - t0)) -lt 10

if [[ ${fails} -gt 0 ]]; then
  echo "image smoke: ${fails} failed"
  exit 1
fi
echo "image smoke: all passed"
