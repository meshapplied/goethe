#!/usr/bin/env bash
# Goethe CDN installer: verified CDN artifacts, py-env, nginx, TLS, goethed.

set -euo pipefail

GOETHE_PKG_URL="${GOETHE_PKG_URL:-https://pkg.meshapplied.com}"
CDN="${GOETHE_PKG_URL%/}"
SUMS_FILE=""

resolve_goethe_home() {
  if [[ -n "${GOETHE_HOME:-}" ]]; then
    echo "${GOETHE_HOME}"
    return
  fi
  if [[ -d /workspace && -w /workspace ]]; then
    echo "/workspace/.goethe"
    return
  fi
  if [[ -d /var/lib/goethe && -w /var/lib/goethe ]]; then
    echo "/var/lib/goethe"
    return
  fi
  echo "${HOME}/.goethe"
}

normalize_arch() {
  local raw="${1:-}"
  case "${raw}" in
    arm64 | aarch64) echo aarch64 ;;
    x86_64 | amd64) echo x86_64 ;;
    *)
      echo "${raw}"
      ;;
  esac
}

ARCH="$(normalize_arch "$(uname -m)")"
if [[ -n "${GOETHE_ARCH:-}" ]]; then
  ARCH="$(normalize_arch "${GOETHE_ARCH}")"
fi
if [[ "${ARCH}" != "x86_64" && "${ARCH}" != "aarch64" ]]; then
  echo "unsupported architecture: $(uname -m) (need x86_64 or aarch64)" >&2
  exit 1
fi

version_major() {
  local ver="${1:-}"
  ver="${ver#OpenSSL }"
  ver="${ver%% *}"
  ver="${ver%%-*}"
  echo "${ver%%.*}"
}

version_ge() {
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
    local fv="${f[i]:-0}"
    local nv="${n[i]:-0}"
    if ((10#${fv} > 10#${nv})); then
      return 0
    fi
    if ((10#${fv} < 10#${nv})); then
      return 1
    fi
  done
  return 0
}

require_tool_version() {
  local tool="$1" min="$2" ver_cmd="$3"
  local ver
  ver="$(${ver_cmd} 2>/dev/null | head -n1 || true)"
  if [[ -z "${ver}" ]]; then
    echo "${tool} is required" >&2
    exit 1
  fi
  local stripped="${ver}"
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

check_nvidia_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "${GOETHE_REQUIRE_GPU:-}" == "1" ]]; then
      echo "nvidia-smi not found (GOETHE_REQUIRE_GPU=1)" >&2
      exit 1
    fi
    echo "WARNING: nvidia-smi not found; GPU workloads need an NVIDIA driver" >&2
    return 0
  fi
  local ver major
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [[ -z "${ver}" ]]; then
    echo "WARNING: could not read NVIDIA driver version from nvidia-smi" >&2
    return 0
  fi
  major="$(version_major "${ver}")"
  if [[ -z "${major}" ]] || ((major < 550)); then
    echo "NVIDIA driver ${ver} is below minimum 550" >&2
    exit 1
  fi
}

GOETHE_HOME="$(resolve_goethe_home)"
export GOETHE_HOME
mkdir -p "${GOETHE_HOME}/bin" "${GOETHE_HOME}/logs" "${GOETHE_HOME}/nginx/certs" "${GOETHE_HOME}/goethe"

GOETHE_RELEASE_PUB_PEM='__GOETHE_RELEASE_PUB_PEM__'
printf '%s\n' "${GOETHE_RELEASE_PUB_PEM}" >"${GOETHE_HOME}/release_pub.pem"

curl_auth() {
  local url="$1"
  shift
  local auth=()
  if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${GOETHE_PKG_TOKEN}")
  fi
  curl -fsSL "${auth[@]}" "$url" "$@"
}

fetch_raw() {
  curl_auth "${CDN}/$1" -o "$2"
}

lookup_sum() {
  awk -v p="$1" '$2 == p { print $1; exit }' "${SUMS_FILE}"
}

verify_file_sum() {
  local rel="$1" path="$2"
  local expected got
  expected="$(lookup_sum "${rel}")"
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

fetch_verified() {
  local rel="$1" dest="$2"
  fetch_raw "${rel}" "${dest}.part"
  verify_file_sum "${rel}" "${dest}.part"
  mv "${dest}.part" "${dest}"
}

verify_cdn_or_exit() {
  if [[ "${GOETHE_INSECURE:-}" == "1" ]]; then
    echo "WARNING: GOETHE_INSECURE=1 skips CDN signature and checksum verification" >&2
    return 1
  fi
  if [[ "${GOETHE_RELEASE_PUB_PEM}" == *"__GOETHE_RELEASE_PUB_PEM__"* ]]; then
    echo "unsigned install.sh; set GOETHE_INSECURE=1 for dev only" >&2
    exit 1
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required for CDN verification" >&2
    exit 1
  fi
  SUMS_FILE="${GOETHE_HOME}/SHA256SUMS"
  fetch_raw "SHA256SUMS.sig" "${GOETHE_HOME}/SHA256SUMS.sig"
  fetch_raw "SHA256SUMS" "${SUMS_FILE}"
  if ! openssl dgst -sha256 -verify "${GOETHE_HOME}/release_pub.pem" -signature "${GOETHE_HOME}/SHA256SUMS.sig" "${SUMS_FILE}" >/dev/null 2>&1; then
    echo "signature verification failed" >&2
    exit 1
  fi
  if [[ ! -s "${SUMS_FILE}" ]]; then
    echo "empty SHA256SUMS" >&2
    exit 1
  fi
  if ! awk -v a="bin/linux-${ARCH}/goethe" '$2 == a { found=1; exit } END { exit !found }' "${SUMS_FILE}"; then
    echo "CDN missing bin/linux-${ARCH}/goethe for arch ${ARCH}" >&2
    exit 1
  fi
  return 0
}

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
check_nvidia_driver

SECURE=0
if verify_cdn_or_exit; then
  SECURE=1
fi

fetch() {
  if [[ "${SECURE}" -eq 1 ]]; then
    fetch_verified "$1" "$2"
  else
    fetch_raw "$1" "$2"
  fi
}

UV_BIN="${GOETHE_HOME}/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  if [[ "${SECURE}" -eq 1 ]]; then
    fetch_verified "bin/linux-${ARCH}/uv" "${UV_BIN}"
    chmod +x "${UV_BIN}"
  else
    fetch_raw "bin/linux-${ARCH}/uv" "${UV_BIN}" 2>/dev/null && chmod +x "${UV_BIN}" || true
    if [[ ! -x "${UV_BIN}" ]]; then
      curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${ARCH}-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "${GOETHE_HOME}/bin" --strip-components=1 "uv-${ARCH}-unknown-linux-gnu/uv" 2>/dev/null || true
    fi
    if [[ ! -x "${UV_BIN}" ]]; then
      curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${GOETHE_HOME}/bin" sh
      UV_BIN="${GOETHE_HOME}/bin/uv"
    fi
  fi
fi
export PATH="${GOETHE_HOME}/bin:${PATH}"

mkdir -p "${GOETHE_HOME}/pyenv"
fetch "pyenv/requirements.lock" "${GOETHE_HOME}/pyenv/requirements.lock"
fetch "pyenv/bootstrap.sh" "${GOETHE_HOME}/pyenv/bootstrap.sh"
if [[ "${GOETHE_VERIFY_ONLY:-}" == "1" ]]; then
  exit 0
fi
WHEEL_NAME="$(curl_auth "${CDN}/wheels/LATEST" 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
WHEEL_FILE=""
if [[ -n "${WHEEL_NAME}" ]]; then
  if [[ "${SECURE}" -eq 1 ]]; then
    fetch_verified "wheels/LATEST" "${GOETHE_HOME}/pyenv/LATEST"
    WHEEL_NAME="$(tr -d '[:space:]' <"${GOETHE_HOME}/pyenv/LATEST")"
  fi
  fetch "wheels/${WHEEL_NAME}" "${GOETHE_HOME}/pyenv/${WHEEL_NAME}"
  WHEEL_FILE="${GOETHE_HOME}/pyenv/${WHEEL_NAME}"
fi
if [[ -n "${WHEEL_FILE}" && -f "${WHEEL_FILE}" ]]; then
  export GOETHE_SGLANG_WHEEL="${WHEEL_FILE}"
fi
bash "${GOETHE_HOME}/pyenv/bootstrap.sh"

want_ternary=0
if [[ "${GOETHE_INSTALL_TERNARY:-}" == "1" ]]; then
  want_ternary=1
elif [[ "${SECURE}" -eq 1 ]] && grep -q 'goethe/ternary-kernels.tar.zst' "${SUMS_FILE}"; then
  want_ternary=1
fi
fetch_ternary_pack() {
  local dest="$1"
  local auth=() url code
  if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${GOETHE_PKG_TOKEN}")
  fi
  url="${CDN}/goethe/ternary-kernels.tar.zst"
  code="$(curl -sSL "${auth[@]}" -w '%{http_code}' -o "${dest}.part" "${url}" || echo 000)"
  if [[ "${code}" == "401" || "${code}" == "403" || "${code}" == "404" ]]; then
    rm -f "${dest}.part"
    echo "warning: ternary pack HTTP ${code}" >&2
    return 1
  fi
  if [[ "${code}" != "200" ]]; then
    rm -f "${dest}.part"
    echo "warning: ternary pack HTTP ${code}" >&2
    return 1
  fi
  if [[ "${SECURE}" -eq 1 ]]; then
    verify_file_sum "goethe/ternary-kernels.tar.zst" "${dest}.part"
  fi
  mv "${dest}.part" "${dest}"
}

if [[ "${want_ternary}" -eq 1 && ! -d "${GOETHE_HOME}/goethe/ternary-kernels" ]]; then
  TERNARY_TAR="${GOETHE_HOME}/goethe/ternary-kernels.tar.zst"
  fetch_ternary_pack "${TERNARY_TAR}" || true
  if [[ -f "${TERNARY_TAR}" ]]; then
    zstd -d -f "${TERNARY_TAR}" -o "${GOETHE_HOME}/goethe/ternary-kernels.tar"
    tar -xf "${GOETHE_HOME}/goethe/ternary-kernels.tar" -C "${GOETHE_HOME}/goethe"
    rm -f "${TERNARY_TAR}" "${GOETHE_HOME}/goethe/ternary-kernels.tar"
  fi
fi

for bin in goethe goethed; do
  if [[ ! -x "${GOETHE_HOME}/bin/${bin}" ]]; then
    if [[ "${SECURE}" -eq 1 ]]; then
      fetch_verified "bin/linux-${ARCH}/${bin}" "${GOETHE_HOME}/bin/${bin}"
      chmod +x "${GOETHE_HOME}/bin/${bin}"
    else
      fetch "bin/linux-${ARCH}/${bin}" "${GOETHE_HOME}/bin/${bin}" && chmod +x "${GOETHE_HOME}/bin/${bin}" || true
    fi
  fi
done
export PATH="${GOETHE_HOME}/bin:${PATH}"

LINK_DIR=""
if [[ -z "${GOETHE_NO_LINK:-}" ]]; then
  for dir in /usr/local/bin "${HOME}/.local/bin"; do
    mkdir -p "${dir}" 2>/dev/null || true
    if [[ -w "${dir}" ]]; then
      ln -sf "${GOETHE_HOME}/bin/goethe" "${dir}/goethe"
      LINK_DIR="${dir}"
      break
    fi
  done
fi

NGINX_ROOT="${GOETHE_HOME}/nginx-root"
NGINX_BIN="${NGINX_ROOT}/usr/sbin/nginx"
if [[ ! -x "${NGINX_BIN}" ]]; then
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    apt-get install -y -qq nginx
    NGINX_BIN="$(command -v nginx)"
  else
    DEB_DIR="${GOETHE_HOME}/nginx/debs"
    mkdir -p "${DEB_DIR}" "${NGINX_ROOT}"
    (cd "${DEB_DIR}" && apt-get download nginx nginx-common)
    for deb in "${DEB_DIR}"/*.deb; do
      dpkg-deb -x "${deb}" "${NGINX_ROOT}"
    done
  fi
fi
export NGINX_BIN

openssl req -x509 -newkey rsa:2048 -keyout "${GOETHE_HOME}/nginx/certs/server.key" -out "${GOETHE_HOME}/nginx/certs/server.crt" -days 3650 -nodes -subj "/CN=goethe-local" 2>/dev/null

PID1="$(ps -p 1 -o comm= 2>/dev/null || true)"
if [[ "$(id -u)" -eq 0 && "${PID1}" == "systemd" ]]; then
  fetch "goethed.service" "/etc/systemd/system/goethed.service"
  if [[ -f "/etc/systemd/system/goethed.service" ]]; then
    systemctl daemon-reload || echo "WARNING: systemctl daemon-reload failed" >&2
    systemctl enable --now goethed.service || echo "WARNING: systemctl enable --now goethed.service failed" >&2
    systemctl reload nginx 2>/dev/null || true
  fi
else
  if [[ -x "${GOETHE_HOME}/bin/goethed" ]] && ! pgrep -f "${GOETHE_HOME}/bin/goethed" >/dev/null 2>&1; then
    nohup "${GOETHE_HOME}/bin/goethed" >>"${GOETHE_HOME}/logs/goethed.log" 2>&1 &
  fi
  if [[ -x "${NGINX_BIN}" ]] && ! pgrep -f "${NGINX_BIN}" >/dev/null 2>&1; then
    nohup "${NGINX_BIN}" -c "${GOETHE_HOME}/nginx/nginx.conf" >>"${GOETHE_HOME}/logs/nginx.log" 2>&1 &
  fi
fi

if command -v goethe >/dev/null 2>&1; then
  goethe hw profile || true
  goethe recommend --json 2>/dev/null | head -c 4096 || true
fi

if [[ -n "${LINK_DIR}" && ":${PATH}:" != *":${LINK_DIR}:"* ]]; then
  echo "Add ${LINK_DIR} to PATH to use goethe"
fi
echo "Goethe install complete: GOETHE_HOME=${GOETHE_HOME}"
