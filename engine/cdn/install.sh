#!/usr/bin/env bash
# Goethe CDN installer: verified CDN artifacts, py-env, nginx binary, TLS material; goethe start brings goethed up.

set -euo pipefail

GOETHE_RELEASE_PUB_PEM='__GOETHE_RELEASE_PUB_PEM__'
DEFAULT_PKG_URL="https://pkg.meshapplied.com"
DRIVER_MIN=580
GLIBC_MIN=2.30
DRIVER_REBOOT=0

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
    echo "${tool} is required: install it (apt-get install ${tool}, dnf install ${tool}) and rerun" >&2
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

driver_version() {
  # Loaded NVIDIA driver version from nvidia-smi, else from the kernel module; empty unless it looks like 580.82.07.
  local v=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    v="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | awk 'NR == 1 {print $1}' || true)"
  fi
  if [[ ! "${v}" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    v="$(sed -n 's/^NVRM version:.*  \([0-9][0-9.]*\)  .*/\1/p' "${GOETHE_NVIDIA_VERSION_FILE:-/proc/driver/nvidia/version}" 2>/dev/null || true)"
  fi
  if [[ "${v}" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    echo "${v}"
  fi
}

nvidia_pci_ids() {
  # PCI device ids of the NVIDIA display or 3D controllers, one per line.
  local d vendor class device
  for d in "${GOETHE_PCI_DIR:-/sys/bus/pci/devices}"/*; do
    vendor="" class="" device=""
    read -r vendor <"${d}/vendor" 2>/dev/null || true
    read -r class <"${d}/class" 2>/dev/null || true
    read -r device <"${d}/device" 2>/dev/null || true
    if [[ "${vendor}" == "0x10de" && "${class}" == 0x03* ]]; then
      echo "${device:-0x0000}"
    fi
  done
}

nvidia_pci_gpus() {
  # Number of NVIDIA display or 3D controllers on the PCI bus.
  nvidia_pci_ids | grep -c . || true
}

blackwell_pci() {
  # True when an NVIDIA GPU on the PCI bus is Blackwell or newer (device id 0x2900 and up), which needs the open module.
  local id
  while read -r id; do
    if [[ "${id}" =~ ^0x[0-9a-fA-F]{4}$ ]] && ((16#${id#0x} >= 16#2900)); then
      return 0
    fi
  done < <(nvidia_pci_ids)
  return 1
}

open_module() {
  grep -q "Open Kernel Module" "${GOETHE_NVIDIA_VERSION_FILE:-/proc/driver/nvidia/version}" 2>/dev/null
}

module_installed() {
  modinfo nvidia >/dev/null 2>&1
}

started_gpus() {
  # GPUs nvidia-smi can open.
  nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
}

root_cmd() {
  # "" as root, "sudo" when sudo exists; fails when neither can run root commands.
  if [[ "$(id -u)" -eq 0 ]]; then
    echo ""
  elif command -v sudo >/dev/null 2>&1; then
    echo "sudo"
  else
    return 1
  fi
}

pkg_manager() {
  local m
  for m in apt-get dnf yum zypper; do
    if command -v "${m}" >/dev/null 2>&1; then
      echo "${m}"
      return 0
    fi
  done
  return 1
}

can_ask() {
  { : </dev/tty >/dev/tty; } 2>/dev/null
}

agree() {
  # Consent for a system change: setting $3 is 1 (yes), 0 (no) or unset (ask on the terminal); $4=auto lets root skip asking.
  local why="$1" question="$2" setting="$3" choice="${!3:-ask}" answer
  case "${choice}" in
    0 | no) echo "${why}; skipped (${setting}=0)" >&2; return 1 ;;
    1 | yes) echo "${why}; going ahead (${setting}=1): ${question}" >&2; return 0 ;;
  esac
  if [[ "${4:-}" == auto && "$(id -u)" -eq 0 ]]; then
    echo "${why}; going ahead as root: ${question}" >&2
    return 0
  fi
  if ! can_ask; then
    echo "${why}; rerun with ${setting}=1 to let the installer do it" >&2
    return 1
  fi
  printf '%s.\n%s? [Y/n] ' "${why}" "${question}" >/dev/tty
  read -r answer </dev/tty || answer="n"
  [[ -z "${answer}" || "${answer}" == [Yy]* ]]
}

install_packages() {
  # Install packages with the system package manager as root, after consent; $1 says why.
  local why="$1" mgr sudo
  shift
  if ! mgr="$(pkg_manager)"; then
    echo "${why}; install $* with your package manager" >&2
    return 1
  fi
  if ! sudo="$(root_cmd)"; then
    echo "${why}; install $* as root (${mgr} install $*)" >&2
    return 1
  fi
  agree "${why}" "Install $* with ${mgr}${sudo:+ (uses sudo)}" GOETHE_INSTALL_DEPS auto || return 1
  case "${mgr}" in
    apt-get)
      ${sudo} apt-get update -qq >/dev/null || echo "WARNING: apt-get update reported errors; installing from the current package lists" >&2
      ${sudo} env DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes apt-get install -y -qq "$@" >/dev/null
      ;;
    dnf | yum) ${sudo} "${mgr}" -y -q install "$@" >/dev/null ;;
    zypper) ${sudo} zypper -n -q install "$@" >/dev/null ;;
  esac
}

ensure_prerequisites() {
  # curl, zstd, openssl and CA certificates; missing ones are installed (root without asking).
  local missing=() tool
  for tool in curl zstd openssl; do
    command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
  done
  if [[ ! -e /etc/ssl/certs/ca-certificates.crt && ! -e /etc/pki/tls/certs/ca-bundle.crt && ! -e /etc/ssl/ca-bundle.pem ]]; then
    missing+=(ca-certificates)
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    install_packages "missing ${missing[*]}" "${missing[@]}" || true
  fi
}

compiler_packages() {
  case "$(pkg_manager || true)" in
    apt-get) echo build-essential ;;
    dnf | yum | zypper) echo gcc gcc-c++ make ;;
  esac
}

check_compiler() {
  # nvcc builds kernels on a model's first load with the system C++ compiler (gcc 9 to 15); offer one when missing.
  local cxx major pkgs=()
  for cxx in "${CXX:-}" c++ g++ clang++; do
    if [[ -n "${cxx}" ]] && command -v "${cxx}" >/dev/null 2>&1; then
      major="$("${cxx}" -dumpversion 2>/dev/null | cut -d. -f1 || true)"
      if [[ "${major}" =~ ^[0-9]+$ ]] && ((major < 9)) && ! "${cxx}" --version 2>/dev/null | grep -q clang; then
        echo "WARNING: ${cxx} is gcc ${major}; GPU kernel builds need gcc 9 or newer (goethe doctor shows the fix)" >&2
      else
        done_line "C++ compiler: $(command -v "${cxx}") ${major}"
      fi
      return 0
    fi
  done
  read -r -a pkgs <<<"$(compiler_packages)"
  if install_packages "no C++ compiler, which Goethe needs to build GPU kernels on a model's first load" "${pkgs[@]}"; then
    return 0
  fi
  echo "WARNING: without a C++ compiler, models whose kernels are built on first load (NVFP4) cannot start; goethe doctor shows the fix" >&2
}

no_gpu() {
  # Warn and continue, or stop when GOETHE_REQUIRE_GPU=1 asks for a working GPU.
  if [[ "${GOETHE_REQUIRE_GPU:-}" == "1" ]]; then
    echo "$1 (GOETHE_REQUIRE_GPU=1)" >&2
    exit 1
  fi
  echo "WARNING: $1" >&2
}

check_gpu_arch() {
  # GPUs older than compute capability 8.0 are not used; say so instead of failing at the first load.
  local idx name cc legacy=() total=0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while IFS=, read -r idx name cc; do
    cc="${cc// /}"
    [[ "${cc}" =~ ^[0-9]+\.[0-9]+$ ]] || continue
    total=$((total + 1))
    if ((${cc%%.*} * 10 + ${cc#*.} < 80)); then
      legacy+=("gpu${idx// /}${name:+ ${name# }} (SM ${cc})")
    fi
  done < <(nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null || true)
  if [[ ${#legacy[@]} -eq 0 ]]; then
    return 0
  fi
  if [[ ${#legacy[@]} -eq ${total} ]]; then
    no_gpu "no supported GPU: $(IFS=';'; echo "${legacy[*]}" | sed 's/;/, /g'). Goethe needs compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)"
  else
    echo "note: $(IFS=';'; echo "${legacy[*]}" | sed 's/;/, /g') is older than Goethe supports (compute capability 8.0+) and will not be used" >&2
  fi
}

FAILED_INIT="the card failed to initialise (sudo dmesg | grep -i nvrm shows why): power the machine fully off and on (on a VM, stop and start the instance), and check the GPU power cables and that Above 4G decoding is on in the BIOS"

in_container() {
  [[ -n "${container:-}" || -e /.dockerenv || -e /run/.containerenv ]]
}

os_field() {
  # One field of os-release, unquoted.
  sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" "${GOETHE_OS_RELEASE:-/etc/os-release}" 2>/dev/null | head -n 1
}

driver_install_steps() {
  # Install the open NVIDIA driver 580+ with the distribution's tools or NVIDIA's CUDA repository; $1 is "" or sudo.
  local sudo="$1" id like ver repo_arch tag
  id="$(os_field ID)"
  like="$(os_field ID_LIKE)"
  ver="$(os_field VERSION_ID)"
  repo_arch="x86_64"
  [[ "${ARCH:-$(uname -m)}" == "aarch64" ]] && repo_arch="sbsa"
  case " ${id} ${like} " in
    *" ubuntu "*)
      ${sudo} apt-get update -qq || true
      ${sudo} env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common "linux-headers-$(uname -r)" || true
      if ${sudo} timeout 900 ubuntu-drivers install --gpgpu "nvidia:${DRIVER_MIN}-server-open"; then
        return 0
      fi
      if ${sudo} env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "nvidia-driver-${DRIVER_MIN}-server-open"; then
        return 0
      fi
      tag="ubuntu${ver//./}"
      ;;
    *" debian "*)
      ${sudo} apt-get update -qq || true
      ${sudo} env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-headers-$(uname -r)" || true
      tag="debian${ver%%.*}"
      ;;
    *" rhel "* | *" centos "* | *" fedora "*)
      [[ "${id}" == "fedora" ]] && tag="fedora${ver%%.*}" || tag="rhel${ver%%.*}"
      ${sudo} dnf -y install dnf-plugins-core "kernel-devel-$(uname -r)" kernel-headers || true
      [[ "${id}" == "fedora" ]] || ${sudo} dnf -y install epel-release || true
      ${sudo} dnf config-manager --add-repo \
        "https://developer.download.nvidia.com/compute/cuda/repos/${tag}/${repo_arch}/cuda-${tag}.repo"
      ${sudo} dnf -y install nvidia-open || ${sudo} dnf -y module install nvidia-driver:open-dkms
      return
      ;;
    *)
      echo "no automatic driver install for ${id:-this distribution}" >&2
      return 1
      ;;
  esac
  local keyring
  keyring="$(mktemp)"
  curl -fsSL -o "${keyring}" "https://developer.download.nvidia.com/compute/cuda/repos/${tag}/${repo_arch}/cuda-keyring_1.1-1_all.deb"
  ${sudo} dpkg -i "${keyring}"
  rm -f "${keyring}"
  ${sudo} apt-get update -qq
  ${sudo} env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-open
}

offer_driver_install() {
  # Offer, then run, the fix for $1: $2 is load (modprobe), open (switch to the open module) or install; true when done.
  local why="$1" action="$2" sudo question err
  if in_container; then
    echo "${why}; this is a container, so fix the NVIDIA driver (${DRIVER_MIN}+) on the host and start the container with --gpus all" >&2
    return 1
  fi
  if ! sudo="$(root_cmd)"; then
    echo "${why}; fix it as root (rerun this installer as root to have it done)" >&2
    return 1
  fi
  case "${action}" in
    load) question="Load it now with modprobe nvidia${sudo:+ (uses sudo)}" ;;
    open) question="Switch to NVIDIA's open kernel module ${DRIVER_MIN}+ with $(os_field ID) packages${sudo:+ (uses sudo)}; needs a reboot" ;;
    *) question="Install the NVIDIA driver ${DRIVER_MIN}+ now with $(os_field ID) packages${sudo:+ (uses sudo)}; may need a reboot" ;;
  esac
  agree "${why}" "${question}" GOETHE_INSTALL_DRIVER || return 1
  if [[ "${action}" == load ]]; then
    if err="$(${sudo} modprobe nvidia 2>&1)"; then
      return 0
    fi
    if [[ "${err}" == *"Key was rejected"* ]]; then
      echo "Secure Boot blocks the NVIDIA module: enroll its key (sudo mokutil --import /var/lib/shim-signed/mok/MOK.der, then reboot and confirm) or turn Secure Boot off" >&2
    else
      echo "modprobe nvidia failed: ${err}; after a driver or kernel update a reboot usually fixes this" >&2
      DRIVER_REBOOT=1
    fi
    return 1
  fi
  if mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    echo "Secure Boot is on: if the driver module is built locally, set the MOK password it asks for and enroll it at the next boot" >&2
  fi
  if command -v nvidia-uninstall >/dev/null 2>&1; then
    echo "removing the driver installed from NVIDIA's .run file first, so the packaged driver can replace it" >&2
    ${sudo} nvidia-uninstall --silent || true
  fi
  if ! driver_install_steps "${sudo}"; then
    echo "WARNING: the NVIDIA driver install failed; goethe doctor names the manual fix" >&2
    return 1
  fi
  ${sudo} modprobe nvidia 2>/dev/null || true
  local ver
  ver="$(driver_version)"
  if [[ "${action}" == open || -z "${ver}" || "$(version_major "${ver}")" -lt "${DRIVER_MIN}" ]]; then
    DRIVER_REBOOT=1
  fi
  return 0
}

check_nvidia_driver() {
  # Driver and GPU state; offers to install, load or replace the driver (with consent), otherwise warns and continues.
  local pci ver major why="" action=install listed
  pci="$(nvidia_pci_gpus)"
  ver="$(driver_version)"
  major="$(version_major "${ver}")"
  if [[ -z "${ver}" && "${pci}" -eq 0 ]]; then
    if in_container; then
      no_gpu "no NVIDIA GPU in this container; start it with the GPU attached (docker run --gpus all)"
    else
      no_gpu "no NVIDIA GPU found on this machine (on a VM, attach the GPU with PCI passthrough or vGPU); Goethe installs, but needs one to serve models"
    fi
    return 0
  fi
  if [[ -n "${ver}" ]] && in_container && [[ ! -e "${GOETHE_DEV_DIR:-/dev}/nvidiactl" ]]; then
    no_gpu "this container has no GPU attached; start it with the GPU passed through (docker run --gpus all)"
    return 0
  fi
  if [[ -z "${ver}" ]]; then
    if module_installed; then
      why="the NVIDIA driver is installed but its kernel module is not loaded"
      action=load
    else
      why="NVIDIA GPU found but no NVIDIA driver is installed"
    fi
  elif ((major < DRIVER_MIN)); then
    why="NVIDIA driver ${ver} is older than ${DRIVER_MIN} (CUDA 13)"
  elif ! in_container && command -v nvidia-smi >/dev/null 2>&1; then
    listed="$(started_gpus)"
    if ((listed == 0)) && ! open_module && blackwell_pci; then
      why="NVIDIA driver ${ver} is the proprietary kernel module, which cannot start Blackwell GPUs"
      action=open
    elif ((listed == 0 && pci > 0)); then
      no_gpu "NVIDIA driver ${ver} is loaded but the GPU did not start; ${FAILED_INIT}"
      return 0
    elif ((listed < pci)); then
      echo "WARNING: $((pci - listed)) of ${pci} NVIDIA GPUs did not start and will be skipped; ${FAILED_INIT}" >&2
    fi
  fi
  if [[ -z "${why}" ]]; then
    check_gpu_arch
    if [[ -n "${listed:-}" && "${listed}" -gt 0 ]]; then
      done_line "NVIDIA driver ${ver}, ${listed} GPU$( ((listed == 1)) || echo s) ready"
    else
      done_line "NVIDIA driver ${ver}"
    fi
    return 0
  fi
  if offer_driver_install "${why}" "${action}"; then
    return 0
  fi
  no_gpu "${why}; Goethe installs anyway, and goethe doctor names the fix"
}

check_disk() {
  # The Python environment needs about 12 GB; stop before downloading when the disk cannot hold it.
  local need_gb=12 free_kb
  if [[ -x "${GOETHE_HOME}/py-env/bin/python" ]]; then
    return 0
  fi
  free_kb="$(df -Pk "${GOETHE_HOME}" 2>/dev/null | awk 'NR == 2 {print $4}')"
  if [[ "${free_kb}" =~ ^[0-9]+$ ]] && ((free_kb < need_gb * 1024 * 1024)); then
    echo "only $((free_kb / 1024 / 1024)) GB free at ${GOETHE_HOME}; Goethe needs about ${need_gb} GB for its Python environment, plus room for models. Free space or set GOETHE_HOME to a larger disk, then rerun" >&2
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
  # Download rel to dest; with a label ($3) on a terminal, show it and curl's progress bar.
  if [[ -n "${3:-}" && "${PROGRESS:-0}" -eq 1 ]]; then
    local auth=()
    if [[ -n "${GOETHE_PKG_TOKEN:-}" ]]; then
      auth=(-H "Authorization: Bearer ${GOETHE_PKG_TOKEN}")
    fi
    echo "  $3"
    curl -fL --progress-bar "${auth[@]}" "${CDN}/$1" -o "$2"
    return
  fi
  curl_auth "${CDN}/$1" -o "$2"
}

expected_sum() {
  awk -v p="$1" '$2 == p { print $1; exit }' "${SUMS_FILE}"
}

sum_matches() {
  # True when path matches the signed SHA256SUMS entry for rel; a missing entry is fatal.
  local rel="$1" path="$2" expected
  expected="$(expected_sum "${rel}")"
  if [[ -z "${expected}" ]]; then
    echo "SHA256SUMS missing entry for ${rel}" >&2
    exit 1
  fi
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

verify_file_sum() {
  # Exit unless path matches; a fresh release can mix old and new files in the CDN cache, so retry every 30 s first.
  local rel="$1" path="$2" try tries="${GOETHE_FETCH_RETRIES:-24}"
  for ((try = 1; try <= tries; try++)); do
    if sum_matches "${rel}" "${path}"; then
      return 0
    fi
    [[ ${try} -eq 1 ]] && echo "the package host is still switching to a new release (its cache keeps files up to 10 minutes); retrying every 30 s" >&2
    sleep "${GOETHE_FETCH_RETRY_SECS:-30}"
    load_sums
    fetch_raw "${rel}" "${path}"
  done
  sum_matches "${rel}" "${path}" && return 0
  echo "checksum mismatch for ${rel}; rerun the installer in a few minutes" >&2
  exit 1
}

fetch() {
  # Download rel to dest (with a progress label $3 on a terminal); verified against SHA256SUMS unless GOETHE_INSECURE=1.
  fetch_raw "$1" "$2.part" "${3:-}"
  if [[ "${SECURE}" -eq 1 ]]; then
    verify_file_sum "$1" "$2.part"
  fi
  mv "$2.part" "$2"
}

load_sums() {
  # Fetch SHA256SUMS and its signature and check them against the embedded key (a mismatched pair is retried too).
  local try=0
  while true; do
    fetch_raw "SHA256SUMS.sig" "${GOETHE_HOME}/SHA256SUMS.sig"
    fetch_raw "SHA256SUMS" "${SUMS_FILE}"
    if openssl dgst -sha256 -verify "${GOETHE_HOME}/release_pub.pem" -signature "${GOETHE_HOME}/SHA256SUMS.sig" \
      "${SUMS_FILE}" >/dev/null 2>&1; then
      return 0
    fi
    try=$((try + 1))
    [[ ${try} -gt ${GOETHE_FETCH_RETRIES:-24} ]] && break
    [[ ${try} -eq 1 ]] && echo "SHA256SUMS and its signature do not match yet (new release still propagating); retrying every 30 s" >&2
    sleep "${GOETHE_FETCH_RETRY_SECS:-30}"
  done
  echo "signature verification failed" >&2
  exit 1
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
  load_sums
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
    fetch "bin/linux-${ARCH}/uv" "${GOETHE_HOME}/bin/uv" "Downloading uv, the Python package installer"
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
  fetch "wheels/${wheel_name}" "${GOETHE_HOME}/pyenv/${wheel_name}" "Downloading Goethe's SGLang build"
  echo "  Installing PyTorch, SGLang and the CUDA libraries (about 8 GB; a few minutes on a fast connection)"
  GOETHE_ARCH="${ARCH}" UV="${GOETHE_HOME}/bin/uv" GOETHE_SGLANG_WHEEL="${GOETHE_HOME}/pyenv/${wheel_name}" \
    bash "${GOETHE_HOME}/pyenv/bootstrap.sh"
}

install_ternary() {
  # Optional ternary kernel pack (401/403/404: not entitled, not an error); records its verified sha256.
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
  sha256sum "${tar}.part" | awk '{print $1}' >"${GOETHE_HOME}/goethe/ternary-kernels.sha256"
  rm -f "${tar}.part"
}

binary_current() {
  # True when bin/<name> already matches the CDN's signed sha256 (never true without a signed SHA256SUMS).
  local want
  if [[ "${SECURE}" -ne 1 || ! -x "${GOETHE_HOME}/bin/$1" ]]; then
    return 1
  fi
  want="$(awk -v p="bin/linux-${ARCH}/$1" '$2 == p { print $1; exit }' "${SUMS_FILE}")"
  [[ -n "${want}" && "$(sha256sum "${GOETHE_HOME}/bin/$1" | awk '{print $1}')" == "${want}" ]]
}

install_binaries() {
  # Stage goethe and goethed (verified) when missing or different from the CDN's; link the CLI unless GOETHE_NO_LINK.
  local bin dir
  STAGED=()
  for bin in goethe goethed; do
    if binary_current "${bin}"; then
      continue
    fi
    fetch "bin/linux-${ARCH}/${bin}" "${GOETHE_HOME}/bin/.${bin}.new" "Downloading ${bin}"
    chmod +x "${GOETHE_HOME}/bin/.${bin}.new"
    if [[ "${bin}" == goethe ]] && ! "${GOETHE_HOME}/bin/.goethe.new" --version >/dev/null; then
      echo "the downloaded goethe does not run on this host; nothing was replaced" >&2
      exit 1
    fi
    if [[ -x "${GOETHE_HOME}/bin/${bin}" ]] && cmp -s "${GOETHE_HOME}/bin/.${bin}.new" "${GOETHE_HOME}/bin/${bin}"; then
      rm -f "${GOETHE_HOME}/bin/.${bin}.new"
      continue
    fi
    STAGED+=("${bin}")
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
  echo "the gateway needs nginx: $1. Install it (apt-get install nginx-core, dnf install nginx) or set NGINX_BIN, then re-run" >&2
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
    if install_packages "the gateway needs nginx" nginx && command -v nginx >/dev/null 2>&1; then
      NGINX_BIN="$(command -v nginx)"
      return
    fi
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
  if install_packages "the gateway needs nginx and none could be extracted" nginx-core && command -v nginx >/dev/null 2>&1; then
    NGINX_BIN="$(command -v nginx)"
    return
  fi
  nginx_missing "nginx, nginx-core and nginx-light extracted under ${root} gave no working ${NGINX_BIN}"
}

install_notices() {
  # Goethe's license and the third-party notices and license texts, when this package carries them.
  local f got=0
  for f in LICENSE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.txt; do
    if [[ "${SECURE}" -eq 1 ]] && [[ -z "$(expected_sum "${f}")" ]]; then
      continue
    fi
    if [[ "${SECURE}" -eq 1 ]]; then
      fetch "${f}" "${GOETHE_HOME}/${f}"
    else
      fetch_raw "${f}" "${GOETHE_HOME}/${f}" 2>/dev/null || continue
    fi
    got=1
  done
  if [[ ${got} -eq 1 ]]; then
    done_line "licenses: ${GOETHE_HOME}/THIRD_PARTY_NOTICES.md"
  fi
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

system_unit_home() {
  # GOETHE_HOME pinned by the existing goethed system unit, empty when there is none.
  local unit="${GOETHE_SYSTEMD_UNIT_DIR:-/etc/systemd/system}/goethed.service" pinned
  if [[ ! -f "${unit}" ]]; then
    return 0
  fi
  pinned="$(grep -m1 -o 'GOETHE_HOME=[^"]*' "${unit}" || true)"
  printf '%s\n' "${pinned#GOETHE_HOME=}"
}

systemd_root() {
  [[ "$(id -u)" -eq 0 && "$(ps -p 1 -o comm= 2>/dev/null || true)" == "systemd" ]]
}

write_system_unit() {
  # Write and load this home's goethed system unit; fails without writing when another home owns the unit.
  local other
  other="$(system_unit_home)"
  if [[ -n "${other}" && "${other}" != "${GOETHE_HOME}" ]]; then
    return 1
  fi
  fetch "goethed.service" "${GOETHE_HOME}/goethed.service"
  sed "s|__GOETHE_HOME__|${GOETHE_HOME}|g" "${GOETHE_HOME}/goethed.service" >/etc/systemd/system/goethed.service
  mkdir -p /etc/goethe
  printf 'NGINX_BIN=%s\n' "${NGINX_BIN}" >/etc/goethe/goethe.env
  systemctl daemon-reload
}

restart_goethed() {
  # Bring back the goethed an upgrade stopped: root's systemd unit unless another home owns it, else goethe up, else nohup.
  local other
  if [[ "${STARTED:-0}" -eq 1 ]] || goethed_answers; then
    return
  fi
  if systemd_root; then
    if write_system_unit; then
      systemctl enable --now goethed.service
      return
    fi
    other="$(system_unit_home)"
    echo "goethed.service already starts goethed for ${other}; this home starts now but gets no autostart." >&2
    echo "To move autostart here: GOETHE_HOME=${other} goethe service disable, then goethe service enable." >&2
  fi
  if [[ "${GOETHE_NO_AUTOSTART:-0}" != "1" ]] && "${GOETHE_HOME}/bin/goethe" up >/dev/null; then
    return
  fi
  local launcher=(nohup)
  if command -v setsid >/dev/null 2>&1; then
    launcher=(setsid nohup)
  fi
  NGINX_BIN="${NGINX_BIN}" "${launcher[@]}" "${GOETHE_HOME}/bin/goethed" </dev/null \
    >>"${GOETHE_HOME}/logs/goethed.log" 2>&1 &
}

apply_binaries() {
  # Move staged binaries into place; a running goethed is stopped first, keeping its engines when it can.
  local bin rc=0 old="" new unit=0 stop_err=""
  UPGRADE_NOTE=""
  STARTED=0
  RESTART=0
  if [[ ${#STAGED[@]} -eq 0 ]]; then
    return
  fi
  if [[ -x "${GOETHE_HOME}/bin/goethe" ]]; then
    old="$("${GOETHE_HOME}/bin/goethe" --build-info 2>/dev/null || "${GOETHE_HOME}/bin/goethe" --version 2>/dev/null || echo "goethe unknown build")"
  fi
  if [[ -n "${old}" && " ${STAGED[*]} " == *" goethed "* ]] && goethed_answers; then
    if systemd_root && write_system_unit; then
      unit=1
    fi
    RESTART=1
    { stop_err="$("${GOETHE_HOME}/bin/goethe" down --keep-engines 2>&1 1>&3)"; } 3>&1 || rc=$?
    if [[ ${rc} -eq 2 ]]; then
      rc=0
      { stop_err="$("${GOETHE_HOME}/bin/goethe" stop --keep-engines 2>&1 1>&3)"; } 3>&1 || rc=$?
    fi
    if [[ ${rc} -eq 2 ]]; then
      "${GOETHE_HOME}/bin/goethe" stop
      UPGRADE_NOTE="engines were stopped (this goethed cannot hand them over); the new goethed restores them"
    elif [[ ${rc} -ne 0 ]]; then
      echo "${stop_err}" >&2
      echo "could not stop goethed for the upgrade (exit ${rc}); nothing was replaced" >&2
      exit 1
    else
      [[ -z "${stop_err}" ]] || echo "${stop_err}" >&2
      UPGRADE_NOTE="engines kept serving; the new goethed adopts them"
    fi
  fi
  for bin in "${STAGED[@]}"; do
    mv -f "${GOETHE_HOME}/bin/.${bin}.new" "${GOETHE_HOME}/bin/${bin}"
  done
  if [[ ${unit} -eq 1 ]]; then
    systemctl restart goethed.service
    STARTED=1
  fi
  if [[ -n "${old}" ]]; then
    new="$("${GOETHE_HOME}/bin/goethe" --build-info 2>/dev/null || "${GOETHE_HOME}/bin/goethe" --version)"
    UPGRADE_NOTE="upgraded goethe ${old#goethe } -> ${new#goethe }${UPGRADE_NOTE:+; ${UPGRADE_NOTE}}"
  fi
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
  # How the linked CLI finds a non-default home, then the driver reboot and PATH notes.
  if [[ "${GOETHE_HOME}" != "$(default_home)" ]]; then
    echo "GOETHE_HOME=${GOETHE_HOME} is not the default: ${GOETHE_HOME}/bin/goethe${LINK_DIR:+ and ${LINK_DIR}/goethe}" \
      "find it from their own location, no export needed"
  fi
  if [[ "${DRIVER_REBOOT}" == "1" ]]; then
    echo "Reboot to load the new NVIDIA driver, then check it with goethe doctor"
  fi
  if [[ -n "${LINK_DIR}" && ":${PATH}:" != *":${LINK_DIR}:"* ]]; then
    echo "Add ${LINK_DIR} to PATH to use goethe"
  fi
  echo "Goethe install complete: GOETHE_HOME=${GOETHE_HOME}"
}

stage() {
  # A numbered step header, light blue on a terminal.
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    printf '\n\033[1;38;5;117m[%s/%s]\033[0m \033[1m%s\033[0m\n' "$1" "${STAGES}" "$2"
  else
    printf '\n[%s/%s] %s\n' "$1" "${STAGES}" "$2"
  fi
}

done_line() {
  # One indented result line under a step.
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    printf '  \033[32m%s\033[0m %s\n' "●" "$1"
  else
    printf '  ok: %s\n' "$1"
  fi
}

get_started() {
  printf '\nGet started:\n\n\t→ goethe start\n'
}

show_banner() {
  # The Goethe banner in light blue on a fresh install: full, compact or one line, by terminal width.
  local cols
  [[ -t 1 && ! -x "${GOETHE_HOME:-$(default_home)}/bin/goethe" ]] || return 0
  cols="$(tput cols 2>/dev/null || true)"
  [[ "${cols}" =~ ^[0-9]+$ && "${cols}" -gt 0 ]] || cols="$(stty size </dev/tty 2>/dev/null | awk '{print $2}' || true)"
  [[ "${cols}" =~ ^[0-9]+$ ]] || cols=80
  [[ -z "${NO_COLOR:-}" ]] && printf '\033[1;38;5;117m'
  if ((cols >= 143)); then
    banner_full
  elif ((cols >= 60)); then
    banner_compact
  else
    echo "OWN YOUR INFERENCE -> GOETHE"
  fi
  [[ -z "${NO_COLOR:-}" ]] && printf '\033[0m'
  echo
}

banner_compact() {
  cat <<'BANNER'
 ██████╗  ██████╗ ███████╗████████╗██╗  ██╗███████╗
██╔════╝ ██╔═══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝
██║  ███╗██║   ██║█████╗     ██║   ███████║█████╗
██║   ██║██║   ██║██╔══╝     ██║   ██╔══██║██╔══╝
╚██████╔╝╚██████╔╝███████╗   ██║   ██║  ██║███████╗
 ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
            O W N   Y O U R   I N F E R E N C E
BANNER
}

banner_full() {
  cat <<'BANNER'
 ██████╗ ██╗    ██╗███╗   ██╗    ██╗   ██╗ ██████╗ ██╗   ██╗██████╗     ██╗███╗   ██╗███████╗███████╗██████╗ ███████╗███╗   ██╗ ██████╗███████╗
██╔═══██╗██║    ██║████╗  ██║    ╚██╗ ██╔╝██╔═══██╗██║   ██║██╔══██╗    ██║████╗  ██║██╔════╝██╔════╝██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝
██║   ██║██║ █╗ ██║██╔██╗ ██║     ╚████╔╝ ██║   ██║██║   ██║██████╔╝    ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝█████╗  ██╔██╗ ██║██║     █████╗
██║   ██║██║███╗██║██║╚██╗██║      ╚██╔╝  ██║   ██║██║   ██║██╔══██╗    ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══╝
╚██████╔╝╚███╔███╔╝██║ ╚████║       ██║   ╚██████╔╝╚██████╔╝██║  ██║    ██║██║ ╚████║██║     ███████╗██║  ██║███████╗██║ ╚████║╚██████╗███████╗
 ╚═════╝  ╚══╝╚══╝ ╚═╝  ╚═══╝       ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝

     ██╗       ██████╗  ██████╗ ███████╗████████╗██╗  ██╗███████╗
     ╚██╗     ██╔════╝ ██╔═══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝
█████╗╚██╗    ██║  ███╗██║   ██║█████╗     ██║   ███████║█████╗
╚════╝██╔╝    ██║   ██║██║   ██║██╔══╝     ██║   ██╔══██║██╔══╝
     ██╔╝     ╚██████╔╝╚██████╔╝███████╗   ██║   ██║  ██║███████╗
     ╚═╝       ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
BANNER
}

main() {
  # Verify prerequisites and the signed CDN, install everything under GOETHE_HOME; goethed stays down unless upgraded.
  HOME="${HOME:-$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)}"
  export HOME
  show_banner
  CDN="${GOETHE_PKG_URL:-${DEFAULT_PKG_URL}}"
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
  STAGES=5
  PROGRESS=0
  [[ -t 2 ]] && PROGRESS=1
  stage 1 "Checking this machine"
  check_glibc
  ensure_prerequisites
  require_tool_version curl 7.50 'curl --version'
  require_tool_version zstd 1.3 'zstd --version'
  require_tool_version openssl 1.1.1 'openssl version'
  GOETHE_HOME="${GOETHE_HOME:-$(default_home)}"
  export GOETHE_HOME
  mkdir -p "${GOETHE_HOME}/bin" "${GOETHE_HOME}/logs" "${GOETHE_HOME}/models" "${GOETHE_HOME}/nginx/certs" \
    "${GOETHE_HOME}/goethe"
  check_disk
  check_nvidia_driver
  check_compiler
  stage 2 "Verifying the signed package"
  verify_cdn
  if [[ "${SECURE}" -eq 1 ]]; then
    done_line "SHA256SUMS signed by the Goethe release key; every download is checked against it"
  fi
  stage 3 "Setting up the Python environment"
  install_pyenv
  stage 4 "Installing goethe and goethed"
  install_ternary
  install_binaries
  local staged="${GOETHE_HOME}/bin/.goethe.new"
  [[ -x "${staged}" ]] || staged="${GOETHE_HOME}/bin/goethe"
  done_line "$("${staged}" --version 2>/dev/null || echo goethe)"
  export PATH="${GOETHE_HOME}/bin:${PATH}"
  NGINX_BIN="${NGINX_BIN:-}"
  stage 5 "Setting up the gateway"
  resolve_nginx
  export NGINX_BIN
  done_line "nginx: ${NGINX_BIN}"
  ensure_certs
  done_line "TLS certificate: ${GOETHE_HOME}/nginx/certs/server.crt"
  install_notices
  write_goethe_env
  apply_binaries
  if [[ "${RESTART}" -eq 1 ]]; then
    restart_goethed
    wait_goethed
    wait_gateway
  fi
  if [[ -n "${UPGRADE_NOTE}" ]]; then
    echo "${UPGRADE_NOTE}"
  fi
  local profile
  if ! profile="$(goethe hw profile 2>&1)"; then
    echo "note: no GPU profile yet (${profile##*$'\n'}); goethe doctor shows the state once the GPU is usable" >&2
  fi
  summary
  get_started
}

main "$@"
