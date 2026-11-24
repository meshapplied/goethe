#!/usr/bin/env bash
# Goethe installer for Linux, macOS, and WSL (Windows uses install.ps1, which runs this script inside Ubuntu).

set -euo pipefail

GOETHE_RELEASE_PUB_PEM='__GOETHE_RELEASE_PUB_PEM__'
DEFAULT_PKG_URL="https://pkg.goethe.sh"
GLIBC_MIN=2.30

die() {
  # die CODE message: tell the user, and log one row to <home>/errors.log (see errcode.rs) unless the file is over 64k or the last row has the same code.
  local code="$1" msg="$2" home="${GOETHE_HOME:-$(default_home 2>/dev/null || true)}" v="${VERSION:-latest}" log detail
  echo "goethe install: ${msg}" >&2
  log="${home}/errors.log"
  [[ "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.]+)?$ ]] || v=unknown
  detail="$(printf '%s' "${msg}" | head -n1 | sed -E "s#[A-Za-z]+://[^ ]+#<url>#g; s#(^|[ (=])[/~][^ ]*#\\1<path>#g" | tr '\t' ' ' | cut -c1-120)"
  if [[ -n "${home}" ]] && mkdir -p "${home}" 2>/dev/null && [[ ! -e "${log}" || -z "$(find "${log}" -size +64k 2>/dev/null)" ]] &&
    [[ "$(tail -n1 "${log}" 2>/dev/null | cut -f2)" != "${code}" ]]; then
    (umask 077 && printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "${code}" "${v}" "${detail}" >>"${log}") 2>/dev/null || true
  fi
  exit 1
}

default_home() {
  # macOS ~/.goethe; Linux /workspace/.goethe (RunPod, Vast), else /var/lib/goethe for root or when writable, else ~/.goethe.
  if [[ "$(uname -s)" == Darwin ]]; then
    echo "${HOME}/.goethe"
  elif [[ -d /workspace && -w /workspace ]]; then
    echo "/workspace/.goethe"
  elif [[ "$(id -u)" -eq 0 || (-d /var/lib/goethe && -w /var/lib/goethe) ]]; then
    echo "/var/lib/goethe"
  else
    echo "${HOME}/.goethe"
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

version_ge() {
  # True when dotted version $1 >= $2.
  local IFS=. i
  local -a f n
  read -r -a f <<<"$1"
  read -r -a n <<<"$2"
  for ((i = 0; i < ${#n[@]} || i < ${#f[@]}; i++)); do
    local a="${f[i]:-0}" b="${n[i]:-0}"
    a="${a%%[!0-9]*}" b="${b%%[!0-9]*}"
    ((10#${a:-0} > 10#${b:-0})) && return 0
    ((10#${a:-0} < 10#${b:-0})) && return 1
  done
  return 0
}

parse_args() {
  # --authkey KEY (or GOETHE_AUTHKEY) links the node; --platform-url picks the platform; --version (or GOETHE_VERSION) pins a release.
  AUTHKEY="${GOETHE_AUTHKEY:-}"
  PLATFORM_URL=""
  VERSION="${GOETHE_VERSION:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --authkey | --platform-url | --version)
        [[ $# -ge 2 ]] || die INSTALL_ARGS "$1 needs a value"
        case "$1" in --authkey) AUTHKEY="$2" ;; --platform-url) PLATFORM_URL="$2" ;; --version) VERSION="$2" ;; esac
        shift 2
        ;;
      --authkey=*) AUTHKEY="${1#*=}"; shift ;;
      --platform-url=*) PLATFORM_URL="${1#*=}"; shift ;;
      --version=*) VERSION="${1#*=}"; shift ;;
      *) echo "note: ignoring unknown installer argument $1" >&2; shift ;;
    esac
  done
  VERSION="${VERSION#v}"
  [[ -z "${VERSION}" || "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.]+)?$ ]] || die INSTALL_ARGS "--version ${VERSION} is not a version like 0.3.3"
}

check_machine() {
  # Stop before downloading anything on a machine goethe cannot run on.
  local ver
  case "${OS}" in
    Darwin)
      [[ "${ARCH}" == aarch64 ]] || die INSTALL_ARCH "this is an Intel Mac: goethe needs Apple Silicon (M1 or newer)"
      ver="$(sw_vers -productVersion 2>/dev/null || echo 0)"
      version_ge "${ver}" 14 || die INSTALL_OS "macOS ${ver} is too old: goethe needs macOS 14 (Sonoma) or newer"
      ;;
    Linux)
      [[ "${ARCH}" == x86_64 || "${ARCH}" == aarch64 ]] || die INSTALL_ARCH "unsupported architecture ${ARCH} (goethe runs on x86_64 and aarch64)"
      ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
      [[ -n "${ver}" ]] || ver="$(ldd --version 2>/dev/null | head -n1 | awk '{print $NF}' || true)"
      [[ "${ver}" =~ ^[0-9]+\.[0-9]+ ]] || die INSTALL_GLIBC "goethe needs glibc ${GLIBC_MIN} or newer, and this system's C library is not glibc (Alpine and other musl systems are not supported)"
      version_ge "${ver}" "${GLIBC_MIN}" || die INSTALL_GLIBC "glibc ${ver} is older than ${GLIBC_MIN}"
      ;;
    *) die INSTALL_OS "unsupported system ${OS}: goethe runs on Linux and macOS" ;;
  esac
}

need_tools() {
  # curl, openssl (to check the package signature), a sha256 tool and CA certificates; root on Linux may install the missing ones.
  local missing=() t mgr
  for t in curl openssl; do
    command -v "${t}" >/dev/null 2>&1 || missing+=("${t}")
  done
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || missing+=(coreutils)
  if [[ "${OS}" == Linux && ! -e /etc/ssl/certs/ca-certificates.crt && ! -e /etc/pki/tls/certs/ca-bundle.crt && ! -e /etc/ssl/ca-bundle.pem ]]; then
    missing+=(ca-certificates)
  fi
  [[ ${#missing[@]} -eq 0 ]] && return 0
  [[ "${OS}" == Darwin ]] && die INSTALL_TOOLS "${missing[*]} missing; they are part of macOS, so check your PATH"
  mgr="$(command -v apt-get || command -v dnf || command -v yum || command -v zypper || true)"
  if [[ "$(id -u)" -ne 0 || -z "${mgr}" || "${GOETHE_INSTALL_DEPS:-1}" == 0 ]]; then
    die INSTALL_TOOLS "missing ${missing[*]}: install it with your package manager (for example: sudo apt-get install -y ${missing[*]}) and run the installer again"
  fi
  echo "installing ${missing[*]} with ${mgr##*/} (GOETHE_INSTALL_DEPS=0 skips this)"
  case "${mgr##*/}" in
    apt-get) { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"; } >/dev/null ;;
    zypper) zypper -n -q install "${missing[@]}" >/dev/null ;;
    *) "${mgr}" -y -q install "${missing[@]}" >/dev/null ;;
  esac
}

auth_conf() {
  # The optional private-package bearer token as a curl config line, passed through a pipe so it never shows in the process list.
  local t="${GOETHE_PKG_TOKEN:-}"
  t="${t//[$'\r\n']/}"
  t="${t//\\/\\\\}"
  [[ -z "${t}" ]] || printf 'header = "Authorization: Bearer %s"\n' "${t//\"/\\\"}"
}

get() {
  curl -fsSL --retry 3 --retry-delay 2 -K <(auth_conf) "${BASE}/$1" -o "$2"
}

check_pkg_url() {
  # https or file:// only; plain http just for a loopback host, or a test host with GOETHE_ALLOW_HTTP=1, because the package token travels with every request.
  local host
  case "$1" in
    https://* | file://*) return 0 ;;
    http://*)
      host="${1#http://}"
      host="${host%%[/?#]*}"
      host="${host##*@}"
      if [[ "${host}" == \[* ]]; then host="${host%%]*}]"; else host="${host%%:*}"; fi
      if [[ "${host}" == localhost || "${host}" == "[::1]" || "${host}" =~ ^127\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
      fi
      [[ "${GOETHE_ALLOW_HTTP:-0}" == 1 ]] && return 0
      die INSTALL_URL "${1} is plain http; use https, file:// or a loopback address (GOETHE_ALLOW_HTTP=1 allows a test host)"
      ;;
    *) die INSTALL_URL "GOETHE_PKG_URL must be an https:// or file:// URL, got ${1}" ;;
  esac
}

pick_base() {
  # Every file of one install comes from one release: a pinned version's folder, else the newest release's folder (the package root when it has none).
  local latest
  if [[ -n "${VERSION}" ]]; then
    BASE="${CDN}/v/${VERSION}"
    return
  fi
  BASE="${CDN}"
  latest="$(curl -fsSL --retry 3 -K <(auth_conf) "${CDN}/VERSION" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "${latest}" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] && curl -fsSL -K <(auth_conf) -o /dev/null "${CDN}/v/${latest}/SHA256SUMS" 2>/dev/null; then
    BASE="${CDN}/v/${latest}"
  fi
}

verify_package() {
  # SHA256SUMS signed by the release key embedded above, then goethe for this platform against it.
  local want
  [[ "${GOETHE_RELEASE_PUB_PEM}" != *"__GOETHE_RELEASE_PUB_PEM__"* ]] || die INSTALL_NO_KEY "this install.sh has no release key; use the one from ${DEFAULT_PKG_URL}"
  printf '%s\n' "${GOETHE_RELEASE_PUB_PEM}" >"${TMP}/release_pub.pem"
  get SHA256SUMS "${TMP}/SHA256SUMS" || die INSTALL_DOWNLOAD "could not download ${BASE}/SHA256SUMS${VERSION:+ (is ${VERSION} a published version?)}"
  get SHA256SUMS.sig "${TMP}/SHA256SUMS.sig" || die INSTALL_DOWNLOAD "could not download ${BASE}/SHA256SUMS.sig"
  openssl dgst -sha256 -verify "${TMP}/release_pub.pem" -signature "${TMP}/SHA256SUMS.sig" "${TMP}/SHA256SUMS" >/dev/null 2>&1 ||
    die INSTALL_SIGNATURE "the package list is not signed by the Goethe release key; nothing was installed"
  echo "  SHA256SUMS signed by the Goethe release key; every download is checked against it"
  want="$(awk -v p="bin/${PLATFORM}/goethe" '$2 == p { print $1; exit }' "${TMP}/SHA256SUMS")"
  [[ -n "${want}" ]] || die INSTALL_NO_BUILD "this release has no goethe for ${PLATFORM}"
  get "bin/${PLATFORM}/goethe" "${TMP}/goethe" || die INSTALL_DOWNLOAD "could not download goethe for ${PLATFORM}"
  [[ "$(sha256_of "${TMP}/goethe")" == "${want}" ]] || die INSTALL_SIGNATURE "the downloaded goethe does not match the signed package list; nothing was installed"
  chmod +x "${TMP}/goethe"
  "${TMP}/goethe" --version >/dev/null 2>&1 || die INSTALL_BINARY "the downloaded goethe does not run on this machine; nothing was installed"
}

older_release() {
  # A pinned release from before goethe setup: its own signed install.sh installs it, from its folder.
  local want
  want="$(awk '$2 == "install.sh" { print $1; exit }' "${TMP}/SHA256SUMS")"
  [[ -n "${want}" ]] || die INSTALL_NO_BUILD "release ${VERSION:-${BASE}} has no installer in its package list"
  get install.sh "${TMP}/release-install.sh" || die INSTALL_DOWNLOAD "could not download that release's install.sh"
  [[ "$(sha256_of "${TMP}/release-install.sh")" == "${want}" ]] || die INSTALL_SIGNATURE "that release's install.sh does not match its signed package list"
  echo "  this release predates goethe setup; running its own installer"
  local args=() script
  [[ -n "${PLATFORM_URL}" ]] && args+=(--platform-url "${PLATFORM_URL}")
  script="$(cat "${TMP}/release-install.sh")"
  rm -rf "${TMP}"
  trap - EXIT
  GOETHE_PKG_URL="${BASE}" GOETHE_AUTHKEY="${AUTHKEY}" exec bash -c "${script}" install.sh ${args[@]+"${args[@]}"}
}

show_banner() {
  [[ -t 1 && ! -x "${GOETHE_HOME}/bin/goethe" ]] || return 0
  [[ -z "${NO_COLOR:-}" ]] && printf '\033[1;38;5;117m'
  if (($(tput cols 2>/dev/null || echo 80) >= 52)); then
    cat <<'BANNER'
 ██████╗  ██████╗ ███████╗████████╗██╗  ██╗███████╗
██╔════╝ ██╔═══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝
██║  ███╗██║   ██║█████╗     ██║   ███████║█████╗
██║   ██║██║   ██║██╔══╝     ██║   ██╔══██║██╔══╝
╚██████╔╝╚██████╔╝███████╗   ██║   ██║  ██║███████╗
 ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
BANNER
  else
    echo "GOETHE"
  fi
  [[ -z "${NO_COLOR:-}" ]] && printf '\033[0m'
  echo
}

main() {
  HOME="${HOME:-$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)}"
  export HOME
  parse_args "$@"
  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "${ARCH}" in arm64 | aarch64) ARCH=aarch64 ;; amd64) ARCH=x86_64 ;; esac
  if [[ "${OS}" == Darwin ]]; then PLATFORM="darwin-${ARCH}"; else PLATFORM="linux-${ARCH}"; fi
  GOETHE_HOME="${GOETHE_HOME:-$(default_home)}"
  export GOETHE_HOME
  show_banner
  echo "Checking this machine and the signed package"
  check_machine
  need_tools
  CDN="${GOETHE_PKG_URL:-${DEFAULT_PKG_URL}}"
  CDN="${CDN%/}"
  check_pkg_url "${CDN}"
  pick_base
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/goethe-install.XXXXXX")"
  trap 'rm -rf "${TMP}"' EXIT
  verify_package
  if ! "${TMP}/goethe" setup --help >/dev/null 2>&1; then
    older_release
  fi
  mkdir -p "${GOETHE_HOME}"
  local args=(setup --pkg-url "${CDN}" --from "${BASE}")
  [[ -n "${PLATFORM_URL}" ]] && args+=(--platform-url "${PLATFORM_URL}")
  [[ "${GOETHE_KEEP_PKG_URL:-0}" == 1 ]] && args+=(--keep-file-url)
  GOETHE_AUTHKEY="${AUTHKEY}" "${TMP}/goethe" "${args[@]}"
}

main "$@"
