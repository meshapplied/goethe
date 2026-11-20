#!/usr/bin/env bash
# goethe doctor --fix runs this with one action after the user agreed: load, open or install the NVIDIA driver (as root), or compat (as the user).

set -euo pipefail

DRIVER_MIN=580
COMPAT_VERSION=580.196.07
COMPAT_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204"
COMPAT_SHA256_x86_64=b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035
COMPAT_SHA256_aarch64=aa3fdea008f678ba2dd865e371f4127d3cd8bf56aaa118b0a438853a50e73257
ARCH="$(uname -m)"

os_field() {
  sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" "${GOETHE_OS_RELEASE:-/etc/os-release}" 2>/dev/null | head -n 1
}

reboot_note() {
  echo "REBOOT: reboot the machine so the NVIDIA driver loads, then run goethe doctor"
}

load() {
  local err
  if err="$(modprobe nvidia 2>&1)"; then
    echo "the NVIDIA kernel module is loaded"
    return 0
  fi
  if [[ "${err}" == *"Key was rejected"* ]]; then
    echo "Secure Boot blocks the NVIDIA module: enroll its key (mokutil --import /var/lib/shim-signed/mok/MOK.der, then reboot and confirm) or turn Secure Boot off" >&2
  else
    echo "modprobe nvidia failed: ${err}" >&2
    reboot_note
  fi
  return 1
}

distro_tag() {
  # NVIDIA's CUDA repository name for this release (ubuntu2404, debian12, rhel9, fedora41); nothing when there is no automatic install.
  local id like ver
  id="$(os_field ID)"
  like="$(os_field ID_LIKE)"
  ver="$(os_field VERSION_ID)"
  [[ -n "${ver}" ]] || return 0
  case " ${id} ${like} " in
    *" ubuntu "*) echo "ubuntu${ver//./}" ;;
    *" debian "*) echo "debian${ver%%.*}" ;;
    *" rhel "* | *" centos "* | *" fedora "*) if [[ "${id}" == fedora ]]; then echo "fedora${ver%%.*}"; else echo "rhel${ver%%.*}"; fi ;;
  esac
}

nvidia_repo() {
  local repo_arch=x86_64
  [[ "${ARCH}" == aarch64 ]] && repo_arch=sbsa
  echo "https://developer.download.nvidia.com/compute/cuda/repos/$1/${repo_arch}"
}

reachable() {
  # Whether the driver packages can be fetched now: Ubuntu's own 580 package, else the NVIDIA repository file the install uses.
  local tag="$1"
  if [[ "${tag}" == ubuntu* ]] && apt-get update -qq >/dev/null 2>&1 &&
    apt-cache policy "nvidia-driver-${DRIVER_MIN}-server-open" 2>/dev/null | grep -q 'Candidate: [0-9]'; then
    return 0
  fi
  case "${tag}" in
    ubuntu* | debian*) curl -fsSIL -o /dev/null "$(nvidia_repo "${tag}")/cuda-keyring_1.1-1_all.deb" ;;
    *) curl -fsSIL -o /dev/null "$(nvidia_repo "${tag}")/cuda-${tag}.repo" ;;
  esac
}

install_steps() {
  # The open NVIDIA driver 580+ with the distribution's tools, else NVIDIA's CUDA repository for this release.
  local tag="$1" keyring
  case "${tag}" in
    ubuntu*)
      apt-get update -qq || true
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common "linux-headers-$(uname -r)" || true
      if timeout 900 ubuntu-drivers install --gpgpu "nvidia:${DRIVER_MIN}-server-open"; then
        return 0
      fi
      if DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "nvidia-driver-${DRIVER_MIN}-server-open"; then
        return 0
      fi
      ;;
    debian*)
      apt-get update -qq || true
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-headers-$(uname -r)" || true
      ;;
    *)
      dnf -y install dnf-plugins-core "kernel-devel-$(uname -r)" kernel-headers || true
      [[ "${tag}" == fedora* ]] || dnf -y install epel-release || true
      dnf config-manager --add-repo "$(nvidia_repo "${tag}")/cuda-${tag}.repo"
      dnf -y install nvidia-open || dnf -y module install nvidia-driver:open-dkms
      return
      ;;
  esac
  keyring="$(mktemp)"
  curl -fsSL -o "${keyring}" "$(nvidia_repo "${tag}")/cuda-keyring_1.1-1_all.deb"
  dpkg -i "${keyring}"
  rm -f "${keyring}"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-open
}

install() {
  # Nothing is removed until this release has an automatic install and its packages answer, so a refused install keeps the driver it had.
  local tag
  tag="$(distro_tag)"
  if [[ -z "${tag}" ]]; then
    echo "no automatic driver install for $(os_field ID || true); install the NVIDIA driver ${DRIVER_MIN} or newer from https://www.nvidia.com/drivers" >&2
    return 1
  fi
  if ! reachable "${tag}"; then
    echo "the NVIDIA driver packages for ${tag} cannot be reached right now (network or repository); nothing was changed, run goethe doctor --fix again later" >&2
    return 1
  fi
  if mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    echo "Secure Boot is on: if the driver module is built locally, set the MOK password it asks for and enroll it at the next boot"
  fi
  if command -v nvidia-uninstall >/dev/null 2>&1; then
    echo "removing the driver installed from NVIDIA's .run file first, so the packaged driver can replace it"
    nvidia-uninstall --silent || true
  fi
  install_steps "${tag}"
  modprobe nvidia 2>/dev/null || true
  reboot_note
}

compat() {
  # NVIDIA's cuda-compat-13-0 into GOETHE_HOME/cuda-compat (sha256 pinned), kept only when cuInit works through it.
  local dir="${GOETHE_HOME:?}/cuda-compat" pkg want tmp code
  if [[ "${ARCH}" == aarch64 ]]; then
    pkg="sbsa/cuda-compat-13-0_${COMPAT_VERSION}-1ubuntu1_arm64.deb"
    want="${COMPAT_SHA256_aarch64}"
  else
    pkg="x86_64/cuda-compat-13-0_${COMPAT_VERSION}-1ubuntu1_amd64.deb"
    want="${COMPAT_SHA256_x86_64}"
  fi
  tmp="$(mktemp -d "${GOETHE_HOME}/.cuda-compat.XXXXXX")"
  trap 'rm -rf "${tmp}"' EXIT
  echo "downloading NVIDIA's CUDA 13 forward-compatibility libraries (about 60 MB)"
  curl -fsSL "${COMPAT_URL}/${pkg}" -o "${tmp}/compat.deb"
  [[ "$(sha256sum "${tmp}/compat.deb" | awk '{print $1}')" == "${want}" ]] || { echo "the forward-compatibility package's checksum did not match" >&2; return 1; }
  mkdir "${tmp}/lib"
  "${PYTHON:?}" -I - "${tmp}/compat.deb" "${tmp}/lib" <<'PY'
import io, os, sys, tarfile
deb, dest = sys.argv[1], sys.argv[2]
data = open(deb, "rb").read()
assert data[:8] == b"!<arch>\n", "not a .deb archive"
pos, payload = 8, None
while pos + 60 <= len(data):
    name, size = data[pos:pos + 16].decode().strip().rstrip("/"), int(data[pos + 48:pos + 58])
    if name.startswith("data.tar"):
        payload = data[pos + 60:pos + 60 + size]
        break
    pos += 60 + size + (size & 1)
assert payload, "no data member in the .deb"
prefix = "./usr/local/cuda-13.0/compat/"
with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
    for m in tar.getmembers():
        base = m.name[len(prefix):] if m.name.startswith(prefix) else ""
        if not base or "/" in base or base.startswith("."):
            continue
        target = os.path.join(dest, base)
        if m.issym() and "/" not in m.linkname:
            os.symlink(m.linkname, target)
        elif m.isfile():
            with open(target, "wb") as f:
                f.write(tar.extractfile(m).read())
PY
  code="$(LD_LIBRARY_PATH="${tmp}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${PYTHON}" -I -c "import ctypes; print(ctypes.CDLL('libcuda.so.1').cuInit(0))" 2>/dev/null || echo 999)"
  case "${code}" in
    0)
      rm -rf "${dir}"
      mv "${tmp}/lib" "${dir}"
      echo "CUDA 13 forward compatibility ${COMPAT_VERSION}: the GPU starts with this driver"
      ;;
    804) echo "NVIDIA's forward compatibility does not serve this GPU (data-center GPUs only, error 804); update the driver instead" >&2; return 1 ;;
    803) echo "NVIDIA's forward compatibility does not work with this driver (error 803); update the driver instead" >&2; return 1 ;;
    *) echo "CUDA did not start through NVIDIA's forward-compatibility libraries (error ${code}); update the driver instead" >&2; return 1 ;;
  esac
}

case "${1:-}" in
  load | install | compat) "$1" ;;
  open) install ;;
  *) echo "usage: fix_driver.sh load|open|install|compat" >&2; exit 2 ;;
esac
