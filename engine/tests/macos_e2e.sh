#!/usr/bin/env bash
# End to end on an Apple Silicon Mac: install from packaging/build_macos.sh's output, serve a small GGUF, then take it all down.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/goethe-mac-e2e.XXXXXX")"
CDN="${WORK}/cdn"
export GOETHE_HOME="${WORK}/home"
export GOETHE_NO_LINK=1 GOETHE_NO_COMPLETION=1 GOETHE_NO_AUTOSTART=1 GOETHE_KEEP_PKG_URL=1 GOETHE_LOGIN=0
export GOETHE_MUX_PORT="${GOETHE_MUX_PORT:-31180}" GOETHE_PUBLIC_PORT="${GOETHE_PUBLIC_PORT:-38443}"
MODEL="${MODEL:-bartowski/SmolLM2-135M-Instruct-GGUF}"
G="${GOETHE_HOME}/bin/goethe"
fails=0

pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { local name="$1"; shift; if "$@"; then pass "${name}"; else fail "${name}"; fi; }

cleanup() {
  "${G}" down >/dev/null 2>&1 || true
  if [[ "${KEEP:-0}" != 1 ]]; then rm -rf "${WORK}"; fi
}
trap cleanup EXIT

mkdir -p "${CDN}/bin" "${CDN}/v1"
cp -R "${ROOT}/dist/cdn/bin/darwin-aarch64" "${CDN}/bin/"
(cd "${ROOT}" && cargo build --release --locked --features goethe/test-release-key --bin goethe --bin goethed -q)
cp "${ROOT}/target/release/goethe" "${ROOT}/target/release/goethed" "${CDN}/bin/darwin-aarch64/"
cp "${ROOT}/cdn/v1/registry.json" "${CDN}/v1/registry.json"
for f in LICENSE NOTICE; do [[ -f "${REPO}/${f}" ]] && cp "${REPO}/${f}" "${CDN}/${f}"; done
grep -E '^version\s*=' "${ROOT}/crates/goethe/Cargo.toml" | head -n1 | sed 's/.*"\(.*\)".*/\1/' >"${CDN}/VERSION"

openssl ecparam -genkey -name prime256v1 -noout -out "${WORK}/test.key"
openssl ec -in "${WORK}/test.key" -pubout -out "${WORK}/test_pub.pem" 2>/dev/null
export GOETHE_RELEASE_PUB="${WORK}/test_pub.pem"
bash "${ROOT}/packaging/subst_install_pubkey.sh" "${ROOT}/cdn/install.sh" "${WORK}/test_pub.pem" "${CDN}/install.sh"
(cd "${CDN}" && find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sig | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do
  echo "$(shasum -a 256 "${f}" | awk '{print $1}')  ${f}"
done >SHA256SUMS)
openssl dgst -sha256 -sign "${WORK}/test.key" -out "${CDN}/SHA256SUMS.sig" "${CDN}/SHA256SUMS"

echo "== installer"
GOETHE_PKG_URL="file://${CDN}" bash "${CDN}/install.sh" </dev/null >"${WORK}/install.log" 2>&1 || { cat "${WORK}/install.log"; exit 1; }
cat "${WORK}/install.log"
check "the package signature was verified" grep -q "signed by the Goethe release key" "${WORK}/install.log"
cp "${CDN}/bin/darwin-aarch64/goethe" "${WORK}/goethe.orig"
printf 'x' >>"${CDN}/bin/darwin-aarch64/goethe"
check "a tampered binary is refused" bash -c '! GOETHE_HOME="$0/tampered" GOETHE_PKG_URL="file://$1" bash "$1/install.sh" </dev/null >/dev/null 2>&1' "${WORK}" "${CDN}"
cp "${WORK}/goethe.orig" "${CDN}/bin/darwin-aarch64/goethe"
check "goethe runs" "${G}" --version
check "no Python environment was set up" test ! -e "${GOETHE_HOME}/py-env"
check "no uv was installed" test ! -e "${GOETHE_HOME}/bin/uv"
check "our nginx is in place" test -x "${GOETHE_HOME}/nginx-root/usr/sbin/nginx"
check "the gateway certificate exists" test -s "${GOETHE_HOME}/nginx/certs/server.crt"
check "goethe setup finished the install" grep -q "Goethe install complete" "${WORK}/install.log"
GOETHE_PKG_URL="file://${CDN}" bash "${CDN}/install.sh" </dev/null >"${WORK}/again.log" 2>&1
check "a second install changes nothing" grep -q "already installed; nothing to replace" "${WORK}/again.log"

echo "== machine"
hw="$("${G}" doctor --hardware --json)"
echo "${hw}" | head -c 600; echo
check "one Apple GPU with Metal" grep -q '"compute_capability": *"metal"' <<<"${hw}"
doctor="$("${G}" doctor 2>&1 || true)"
echo "${doctor}"
check "doctor finds Apple Silicon" grep -q "Apple" <<<"${doctor}"
check "doctor runs no NVIDIA checks" bash -c '! grep -qiE "nvidia_driver|glibc|py-env" <<<"$0"' "${doctor}"
check "doctor has no separate RAM check on a Mac" bash -c '! grep -qE "^ram " <<<"$0"' "${doctor}"
hwtext="$("${G}" doctor --hardware 2>&1 || true)"
echo "${hwtext}"
check "the hardware table says GPU memory and has no PCIe column" bash -c 'grep -q "GPU MEMORY TOTAL" <<<"$0" && ! grep -q "PCIE" <<<"$0" && ! grep -q "SM metal" <<<"$0"' "${hwtext}"

echo "== refusals before any download"
out="$("${G}" install nvidia/Llama-3.1-8B-Instruct-NVFP4 --yes 2>&1 || true)"
echo "${out}" | tail -2
check "a model without GGUF files is refused on a Mac" grep -q "no GGUF files" <<<"${out}"
check "nothing was downloaded for it" test ! -e "${GOETHE_HOME}/models/nvidia__Llama-3.1-8B-Instruct-NVFP4"

echo "== install"
"${G}" install "${MODEL}" --yes </dev/null
check "the GGUF is in place" bash -c 'ls "$0"/models/*/*.gguf >/dev/null' "${GOETHE_HOME}"
check "llama.cpp (Metal) was fetched" test -x "${GOETHE_HOME}/engines/llama.cpp/llama-server"

echo "== serve"
"${G}" up --no-boot
"${G}" load "${MODEL}" --no-wait || true
for _ in $(seq 1 120); do
  "${G}" ps --json 2>/dev/null | grep -q '"RUNNING"' && break
  sleep 2
done
"${G}" ps || true
check "the model runs" bash -c '"$0" ps --json | grep -q RUNNING' "${G}"
if ! "${G}" ps --json | grep -q RUNNING; then
  for f in "${GOETHE_HOME}"/logs/*; do echo "--- ${f}"; tail -n 40 "${f}"; done
fi
"${G}" api allow add 127.0.0.1/32 >/dev/null
key="$("${G}" api keys create mac-e2e --raw)"
api="$("${G}" api --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["api"])' 2>/dev/null || echo "https://127.0.0.1:${GOETHE_PUBLIC_PORT:-8443}/v1")"
reply="$(curl -sS --cacert "${GOETHE_HOME}/nginx/certs/server.crt" "${api}/chat/completions" -H "Authorization: Bearer ${key}" \
  -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi\"}], \"max_tokens\": 16}" || true)"
echo "${reply}" | head -c 400; echo
check "the gateway answers a chat request" grep -q '"choices"' <<<"${reply}"
models="$(curl -sS --cacert "${GOETHE_HOME}/nginx/certs/server.crt" "${api}/models" -H "Authorization: Bearer ${key}" || true)"
check "the model is listed" grep -q "${MODEL}" <<<"${models}"
"${G}" doctor --snapshot >"${ROOT}/dist/snapshot-mac.json"
check "the heartbeat names this Mac and its Metal GPU" bash -c 'grep -q "\"os\":\"macOS" "$0" && grep -q "Apple" "$0" && ! grep -q "\"sm\":\"metal\"" "$0"' "${ROOT}/dist/snapshot-mac.json"
check "the heartbeat says Metal, unified memory, and laptop or desktop" bash -c 'grep -q "\"backend\":\"metal\"" "$0" && grep -q "\"memory\":\"unified\"" "$0" && grep -qE "\"form_factor\":\"(laptop|desktop)\"" "$0" && ! grep -q "apple-gpu" "$0"' "${ROOT}/dist/snapshot-mac.json"
gpu="$("${G}" gpu --json)"
check "goethe gpu lists the Apple GPU" grep -qi "apple" <<<"${gpu}"
status="$("${G}" status)"
echo "${status}"
check "status says running" grep -q "running" <<<"${status}"

echo "== take down"
"${G}" unload --all
"${G}" down
check "goethed stopped" bash -c '! "$0" status | grep -q "goethe: running"' "${G}"
"${G}" uninstall --yes </dev/null
check "uninstall removed the home" bash -c '[[ ! -e "$0/bin/goethe" ]]' "${GOETHE_HOME}"

echo
if [[ ${fails} -gt 0 ]]; then
  echo "${fails} FAILED"
  exit 1
fi
echo "ALL PASS"
