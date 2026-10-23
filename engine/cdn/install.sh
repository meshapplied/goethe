#!/usr/bin/env bash
# Goethe CDN installer: verified CDN artifacts, py-env, nginx binary, TLS material, goethed.

set -euo pipefail

GOETHE_RELEASE_PUB_PEM='__GOETHE_RELEASE_PUB_PEM__'
DEFAULT_PKG_URL="https://pkg.meshapplied.com"
DRIVER_MIN=580
GLIBC_MIN=2.30

default_home() {
  # /workspace/.goethe (RunPod, Vast), else /var/lib/goethe for root or when writable, else ~/.goethe.
  if [[ -d /workspace && -w /workspace ]]; then
    echo "/workspace/.goethe"
  elif [[ "$(id -u)" -eq 0 || ( -d /var/lib/goethe && -w /var/lib/goethe ) ]]; then
    echo "/var/lib/goethe"
  else
    echo "${HOME}/.goethe"
  fi
}

normalize_arch() {
  case "${1:-}" in
    arm64 | aarch64) echo aarch64 ;;
    x86_64 | amd64) echo x86_64 ;;
    *) echo "${1:-}" ;;
  esac
}

version_major() {
  # Major component of a dotted version, tolerating an OpenSSL prefix and suffixes.
  local ver="${1:-}"
  ver="${ver#OpenSSL }"
  ver="${ver%% *}"
  ver="${ver%%-*}"
  echo "${ver%%.*}"
}

version_ge() {
  # True when dotted version $1 >= $2.
  local found="$1" need="$2"
  local IFS=.
  local -a f n
  read -r -a f <<<"${found}"
  read -r -a n <<<"${need}"
  local i max=${#n[@]}
  if [[ ${#f[@]} -gt ${max} ]]; then
    max=${#f[@]}
  fi
  for ((i = 0; i < max; i++)); do
    local fv="${f[i]:-0}" nv="${n[i]:-0}"
    fv="${fv%%[!0-9]*}"
    nv="${nv%%[!0-9]*}"
    if ((10#${fv:-0} > 10#${nv:-0})); then
      return 0
    fi
    if ((10#${fv:-0} < 10#${nv:-0})); then
      return 1
    fi
  done
  return 0
}

require_tool_version() {
  # Exit unless tool reports a version >= min.
  local tool="$1" min="$2" ver_cmd="$3" ver stripped
  ver="$(${ver_cmd} 2>/dev/null | head -n1 || true)"
  if [[ -z "${ver}" ]]; then
    echo "${tool} is required" >&2
    exit 1
  fi
  stripped="${ver}"
  case "${tool}" in
    curl) stripped="${ver#curl }"; stripped="${stripped%% *}" ;;
    zstd) stripped="$(echo "${ver}" | sed -n 's/.* v\([0-9][0-9.]*\).*/\1/p' | head -n1)" ;;
    openssl) stripped="${ver#OpenSSL }"; stripped="${stripped%% *}"; stripped="${stripped%%-*}" ;;
  esac
  if ! version_ge "${stripped}" "${min}"; then
    echo "${tool} ${stripped} is below minimum ${min}" >&2
    exit 1
  fi
}

check_glibc() {
  # The portable binaries need glibc 2.30+; musl or an unknown libc cannot run them.
  local ver
  ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
  if [[ -z "${ver}" ]]; then
    ver="$(ldd --version 2>/dev/null | head -n1 | awk '{print $NF}' || true)"
  fi
  if [[ ! "${ver}" =~ ^[0-9]+\.[0-9]+ ]]; then
    echo "could not determine the glibc version; goethe needs glibc ${GLIBC_MIN}+" >&2
    exit 1
  fi
  if ! version_ge "${ver}" "${GLIBC_MIN}"; then
    echo "glibc ${ver} is below minimum ${GLIBC_MIN}" >&2
    exit 1
  fi
}

check_nvidia_driver() {
  # torch cu130 in the py-env needs a CUDA 13 driver (580+) whenever a GPU driver is present.
  local ver major
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "${GOETHE_REQUIRE_GPU:-}" == "1" ]]; then
      echo "nvidia-smi not found (GOETHE_REQUIRE_GPU=1)" >&2
      exit 1
    fi
    echo "WARNING: nvidia-smi not found; GPU workloads need an NVIDIA driver ${DRIVER_MIN}+" >&2
    return 0
  fi
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [[ -z "${ver}" ]]; then
    echo "WARNING: could not read NVIDIA driver version from nvidia-smi" >&2
    return 0
  fi
  major="$(version_major "${ver}")"
  if [[ -z "${major}" ]] || ((major < DRIVER_MIN)); then
    echo "NVIDIA driver ${ver} is below minimum ${DRIVER_MIN} (CUDA 13)" >&2
    exit 1
  fi
}

curl_auth() {
  # curl with the optional private-package bearer token.
  local url="$1"
  shift
  local auth=()
  if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${GOETHE_PKG_TOKEN}")
  fi
  curl -fsSL "${auth[@]}" "${url}" "$@"
}

fetch_raw() {
  curl_auth "${CDN}/$1" -o "$2"
}

verify_file_sum() {
  # Exit unless path matches the signed SHA256SUMS entry for rel.
  local rel="$1" path="$2" expected got
  expected="$(awk -v p="${rel}" '$2 == p { print $1; exit }' "${SUMS_FILE}")"
  if [[ -z "${expected}" ]]; then
    echo "SHA256SUMS missing entry for ${rel}" >&2
    exit 1
  fi
  got="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${got}" != "${expected}" ]]; then
    echo "checksum mismatch for ${rel}" >&2
    exit 1
  fi
}

fetch() {
  # Download rel to dest; verified against SHA256SUMS unless GOETHE_INSECURE=1.
  fetch_raw "$1" "$2.part"
  if [[ "${SECURE}" -eq 1 ]]; then
    verify_file_sum "$1" "$2.part"
  fi
  mv "$2.part" "$2"
}

verify_cdn() {
  # Verify SHA256SUMS against the embedded key; set SECURE=1, or 0 with GOETHE_INSECURE=1.
  local signed=1
  if [[ "${GOETHE_RELEASE_PUB_PEM}" == *"__GOETHE_RELEASE_PUB_PEM__"* ]]; then
    signed=0
  else
    printf '%s\n' "${GOETHE_RELEASE_PUB_PEM}" >"${GOETHE_HOME}/release_pub.pem"
  fi
  if [[ "${GOETHE_INSECURE:-}" == "1" ]]; then
    echo "WARNING: GOETHE_INSECURE=1 skips CDN signature and checksum verification" >&2
    SECURE=0
    return
  fi
  if [[ "${signed}" -eq 0 ]]; then
    echo "unsigned install.sh; set GOETHE_INSECURE=1 for dev only" >&2
    exit 1
  fi
  SUMS_FILE="${GOETHE_HOME}/SHA256SUMS"
  fetch_raw "SHA256SUMS.sig" "${GOETHE_HOME}/SHA256SUMS.sig"
  fetch_raw "SHA256SUMS" "${SUMS_FILE}"
  if ! openssl dgst -sha256 -verify "${GOETHE_HOME}/release_pub.pem" -signature "${GOETHE_HOME}/SHA256SUMS.sig" \
    "${SUMS_FILE}" >/dev/null 2>&1; then
    echo "signature verification failed" >&2
    exit 1
  fi
  local rel
  for rel in "bin/linux-${ARCH}/goethe" "bin/linux-${ARCH}/goethed" "bin/linux-${ARCH}/uv" "pyenv/${LOCK}"; do
    if ! awk -v a="${rel}" '$2 == a { found = 1; exit } END { exit !found }' "${SUMS_FILE}"; then
      echo "CDN missing ${rel} for arch ${ARCH}" >&2
      exit 1
    fi
  done
  SECURE=1
}

install_pyenv() {
  # Fetch uv, the arch lock and bootstrap.sh, then build the hermetic py-env with the SGLang wheel.
  local wheel_name
  if [[ ! -x "${GOETHE_HOME}/bin/uv" ]]; then
    fetch "bin/linux-${ARCH}/uv" "${GOETHE_HOME}/bin/uv"
    chmod +x "${GOETHE_HOME}/bin/uv"
  fi
  mkdir -p "${GOETHE_HOME}/pyenv"
  fetch "pyenv/${LOCK}" "${GOETHE_HOME}/pyenv/${LOCK}"
  fetch "pyenv/bootstrap.sh" "${GOETHE_HOME}/pyenv/bootstrap.sh"
  if [[ "${GOETHE_VERIFY_ONLY:-}" == "1" ]]; then
    echo "verified CDN artifacts for ${ARCH}"
    exit 0
  fi
  fetch "wheels/LATEST" "${GOETHE_HOME}/pyenv/LATEST"
  wheel_name="$(tr -d '[:space:]' <"${GOETHE_HOME}/pyenv/LATEST")"
  if [[ ! "${wheel_name}" =~ ^sglang-[A-Za-z0-9_.+-]+\.whl$ ]]; then
    echo "invalid wheel name in wheels/LATEST: ${wheel_name}" >&2
    exit 1
  fi
  fetch "wheels/${wheel_name}" "${GOETHE_HOME}/pyenv/${wheel_name}"
  GOETHE_ARCH="${ARCH}" UV="${GOETHE_HOME}/bin/uv" GOETHE_SGLANG_WHEEL="${GOETHE_HOME}/pyenv/${wheel_name}" \
    bash "${GOETHE_HOME}/pyenv/bootstrap.sh"
}

install_ternary() {
  # Optional ternary kernel pack: 401/403/404 means not entitled and is not an error.
  local want=0 tar code
  if [[ "${GOETHE_INSTALL_TERNARY:-}" == "1" ]]; then
    want=1
  elif [[ "${SECURE}" -eq 1 ]] && grep -q 'goethe/ternary-kernels.tar.zst' "${SUMS_FILE}"; then
    want=1
  fi
  if [[ "${want}" -eq 0 || -d "${GOETHE_HOME}/goethe/ternary-kernels" ]]; then
    return
  fi
  tar="${GOETHE_HOME}/goethe/ternary-kernels.tar.zst"
  local auth=()
  if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${GOETHE_PKG_TOKEN}")
  fi
  code="$(curl -sSL "${auth[@]}" -w '%{http_code}' -o "${tar}.part" "${CDN}/goethe/ternary-kernels.tar.zst" || echo 000)"
  if [[ "${code}" != "200" ]]; then
    rm -f "${tar}.part"
    echo "warning: ternary pack HTTP ${code}" >&2
    return
  fi
  if [[ "${SECURE}" -eq 1 ]]; then
    verify_file_sum "goethe/ternary-kernels.tar.zst" "${tar}.part"
  fi
  zstd -dc "${tar}.part" | tar -xf - -C "${GOETHE_HOME}/goethe"
  rm -f "${tar}.part"
}

install_binaries() {
  # Fetch goethe and goethed, and link the CLI into a PATH directory unless GOETHE_NO_LINK is set.
  local bin dir
  for bin in goethe goethed; do
    if [[ ! -x "${GOETHE_HOME}/bin/${bin}" ]]; then
      fetch "bin/linux-${ARCH}/${bin}" "${GOETHE_HOME}/bin/${bin}"
      chmod +x "${GOETHE_HOME}/bin/${bin}"
    fi
  done
  if [[ -n "${GOETHE_NO_LINK:-}" ]]; then
    return
  fi
  for dir in /usr/local/bin "${HOME}/.local/bin"; do
    mkdir -p "${dir}" 2>/dev/null || true
    if [[ -w "${dir}" ]]; then
      ln -sf "${GOETHE_HOME}/bin/goethe" "${dir}/goethe"
      LINK_DIR="${dir}"
      return
    fi
  done
}

nginx_missing() {
  echo "the gateway needs nginx: $1. Install it (apt-get install nginx-core) or set NGINX_BIN, then re-run" >&2
  exit 1
}

resolve_nginx() {
  # A working nginx: NGINX_BIN, a system nginx, else nginx, nginx-core or nginx-light extracted under GOETHE_HOME.
  local cand pkg deb root="${GOETHE_HOME}/nginx-root"
  for cand in "${NGINX_BIN:-}" "$(command -v nginx 2>/dev/null || true)" /usr/sbin/nginx /usr/local/sbin/nginx \
    "${root}/usr/sbin/nginx"; do
    if [[ -n "${cand}" && -x "${cand}" ]] && "${cand}" -v >/dev/null 2>&1; then
      NGINX_BIN="${cand}"
      return
    fi
  done
  if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
    nginx_missing "none found and no apt-get/dpkg-deb to extract one"
  fi
  mkdir -p "${GOETHE_HOME}/nginx/debs" "${root}"
  NGINX_BIN="${root}/usr/sbin/nginx"
  for pkg in nginx nginx-core nginx-light; do
    (cd "${GOETHE_HOME}/nginx/debs" && apt-get download "${pkg}" nginx-common) >/dev/null 2>&1 || continue
    for deb in "${GOETHE_HOME}/nginx/debs"/*.deb; do
      dpkg-deb -x "${deb}" "${root}"
    done
    if [[ -x "${NGINX_BIN}" ]] && "${NGINX_BIN}" -v >/dev/null 2>&1; then
      return
    fi
  done
  nginx_missing "nginx, nginx-core and nginx-light extracted under ${root} gave no working ${NGINX_BIN}"
}

write_goethe_env() {
  # Record the CDN (and its registry and token) in GOETHE_HOME/goethe.env, mode 0600, for the CLI and goethed.
  local registry="${GOETHE_REGISTRY_URL:-}"
  if [[ -z "${registry}" && "${CDN}" != "${DEFAULT_PKG_URL}" ]]; then
    registry="${CDN}/v1/registry.json"
  fi
  (
    umask 077
    {
      echo "GOETHE_PKG_URL=${CDN}"
      if [[ -n "${registry}" && "${registry}" != "${DEFAULT_PKG_URL}/v1/registry.json" ]]; then
        echo "GOETHE_REGISTRY_URL=${registry}"
      fi
      if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
        echo "GOETHE_PKG_TOKEN=${GOETHE_PKG_TOKEN}"
      fi
    } >"${GOETHE_HOME}/goethe.env"
  )
  chmod 600 "${GOETHE_HOME}/goethe.env"
}

ensure_certs() {
  # Self-signed TLS pair for the gateway, created only when missing.
  local certs="${GOETHE_HOME}/nginx/certs"
  if [[ -s "${certs}/server.key" && -s "${certs}/server.crt" ]]; then
    return
  fi
  openssl req -x509 -newkey rsa:2048 -keyout "${certs}/server.key" -out "${certs}/server.crt" -days 3650 -nodes \
    -subj "/CN=goethe-local" 2>/dev/null
  chmod 600 "${certs}/server.key"
}

goethed_answers() {
  "${GOETHE_HOME}/bin/goethe" ps >/dev/null 2>&1
}

start_goethed() {
  # systemd unit for root on systemd hosts, otherwise a detached nohup goethed; skipped when one answers.
  if goethed_answers; then
    echo "goethed already running for ${GOETHE_HOME}"
    return
  fi
  if [[ "$(id -u)" -eq 0 && "$(ps -p 1 -o comm= 2>/dev/null || true)" == "systemd" ]]; then
    fetch "goethed.service" "${GOETHE_HOME}/goethed.service"
    sed "s|__GOETHE_HOME__|${GOETHE_HOME}|g" "${GOETHE_HOME}/goethed.service" >/etc/systemd/system/goethed.service
    mkdir -p /etc/goethe
    printf 'NGINX_BIN=%s\n' "${NGINX_BIN}" >/etc/goethe/goethe.env
    systemctl daemon-reload
    systemctl enable --now goethed.service
    return
  fi
  local launcher=(nohup)
  if command -v setsid >/dev/null 2>&1; then
    launcher=(setsid nohup)
  fi
  NGINX_BIN="${NGINX_BIN}" "${launcher[@]}" "${GOETHE_HOME}/bin/goethed" </dev/null \
    >>"${GOETHE_HOME}/logs/goethed.log" 2>&1 &
}

wait_goethed() {
  # Wait up to 30 s until goethed answers on its socket, or fail with the log tail.
  local i
  for ((i = 0; i < 60; i++)); do
    if goethed_answers; then
      return
    fi
    sleep 0.5
  done
  echo "goethed did not answer within 30s; last log lines:" >&2
  tail -n 20 "${GOETHE_HOME}/logs/goethed.log" >&2 2>/dev/null || true
  exit 1
}

wait_gateway() {
  # Install only succeeds once nginx answers on the public port goethed chose.
  local i port="" code=""
  for ((i = 0; i < 60; i++)); do
    port="$(cat "${GOETHE_HOME}/nginx/public_port" 2>/dev/null || true)"
    if [[ -n "${port}" ]]; then
      code="$(curl -sk -o /dev/null -w '%{http_code}' "https://127.0.0.1:${port}/health" || true)"
      if [[ -n "${code}" && "${code}" != "000" ]]; then
        return
      fi
    fi
    sleep 0.5
  done
  tail -n 5 "${GOETHE_HOME}/logs/goethed.log" >&2 2>/dev/null || true
  nginx_missing "the gateway did not answer on https://127.0.0.1:${port:-?} (see ${GOETHE_HOME}/logs)"
}

summary() {
  # Ports, endpoints, cloud mappings and how the linked CLI finds a non-default home.
  local port mux mapped
  port="$(cat "${GOETHE_HOME}/nginx/public_port")"
  mux="$(cat "${GOETHE_HOME}/nginx/mux_port" 2>/dev/null || echo "?")"
  echo "Mux (loopback OpenAI router): http://127.0.0.1:${mux}"
  echo "Gateway: https://127.0.0.1:${port} (default-deny: goethe allowlist add <cidr>; goethe key create --name <n>)"
  mapped="RUNPOD_TCP_PORT_${port}"
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    if [[ -n "${RUNPOD_PUBLIC_IP:-}" && -n "${!mapped:-}" ]]; then
      echo "RunPod public endpoint: https://${RUNPOD_PUBLIC_IP}:${!mapped}"
    else
      echo "RunPod: expose TCP port ${port} in the pod settings to reach the gateway from outside"
    fi
  fi
  mapped="VAST_TCP_PORT_${port}"
  if [[ -n "${PUBLIC_IPADDR:-}" && -n "${!mapped:-}" ]]; then
    echo "Vast public endpoint: https://${PUBLIC_IPADDR}:${!mapped}"
  elif [[ -n "${VAST_CONTAINERLABEL:-}" ]]; then
    echo "Vast: map container port ${port} (-p ${port}:${port}) to reach the gateway from outside"
  fi
  if [[ "${GOETHE_HOME}" != "$(default_home)" ]]; then
    echo "GOETHE_HOME=${GOETHE_HOME} is not the default: ${GOETHE_HOME}/bin/goethe${LINK_DIR:+ and ${LINK_DIR}/goethe}" \
      "find it from their own location, no export needed"
  fi
}

main() {
  # Verify prerequisites and the signed CDN, install everything under GOETHE_HOME, start goethed and wait for it.
  CDN="${GOETHE_PKG_URL:-https://pkg.meshapplied.com}"
  CDN="${CDN%/}"
  ARCH="$(normalize_arch "${GOETHE_ARCH:-$(uname -m)}")"
  if [[ "${ARCH}" != "x86_64" && "${ARCH}" != "aarch64" ]]; then
    echo "unsupported architecture: ${ARCH} (need x86_64 or aarch64)" >&2
    exit 1
  fi
  LOCK="requirements.lock"
  if [[ "${ARCH}" == "aarch64" ]]; then
    LOCK="requirements.aarch64.lock"
  fi
  SECURE=0
  SUMS_FILE=""
  LINK_DIR=""
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates openssl zstd
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required" >&2
    exit 1
  fi
  require_tool_version curl 7.50 'curl --version'
  require_tool_version zstd 1.3 'zstd --version'
  require_tool_version openssl 1.1.1 'openssl version'
  check_glibc
  check_nvidia_driver
  GOETHE_HOME="${GOETHE_HOME:-$(default_home)}"
  export GOETHE_HOME
  mkdir -p "${GOETHE_HOME}/bin" "${GOETHE_HOME}/logs" "${GOETHE_HOME}/models" "${GOETHE_HOME}/nginx/certs" \
    "${GOETHE_HOME}/goethe"
  verify_cdn
  install_pyenv
  install_ternary
  install_binaries
  export PATH="${GOETHE_HOME}/bin:${PATH}"
  NGINX_BIN="${NGINX_BIN:-}"
  resolve_nginx
  export NGINX_BIN
  ensure_certs
  write_goethe_env
  start_goethed
  wait_goethed
  wait_gateway
  goethe hw profile || echo "WARNING: goethe hw profile failed" >&2
  local rec
  if rec="$(goethe recommend 2>&1)"; then
    { printf '%s\n' "${rec}" | head -n 20; } || true
  else
    echo "WARNING: goethe recommend failed: $(printf '%s' "${rec}" | tail -n 2)" >&2
  fi
  summary
  if [[ -n "${LINK_DIR}" && ":${PATH}:" != *":${LINK_DIR}:"* ]]; then
    echo "Add ${LINK_DIR} to PATH to use goethe"
  fi
  echo "Goethe install complete: GOETHE_HOME=${GOETHE_HOME}"
}

main "$@"
