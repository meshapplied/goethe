#!/usr/bin/env bash
# Stage 6 installer checks without network: version floors, truncation safety, no kills, nginx or goethed starts, unit owner.

set -euo pipefail

GOETHE_PKG="${GOETHE_PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INSTALL="${GOETHE_PKG}/cdn/install.sh"
GATE_ID="${GOETHE_GATE_RUN_ID:-$$}"
WORK="$(mktemp -d "/tmp/goethe-test-${GATE_ID}-installer-XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

tool_path() {
  # Absolute path of a host tool, or exit.
  local name="$1" d
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return
  fi
  for d in /usr/bin /bin /usr/sbin /sbin; do
    if [[ -x "${d}/${name}" ]]; then
      echo "${d}/${name}"
      return
    fi
  done
  echo "missing tool ${name}" >&2
  exit 1
}

make_shim_dir() {
  # A PATH directory with the basic tools install.sh needs plus the real curl, zstd and openssl.
  local dir="$1" t
  mkdir -p "${dir}"
  for t in bash env uname sed awk tr sha256sum mv mkdir printf id head getconf ldd curl zstd openssl grep cut df; do
    ln -sf "$(tool_path "${t}")" "${dir}/${t}"
  done
  echo "${dir}"
}

fake() {
  # Replace one shim tool with a script that prints a fixed version line.
  rm -f "$1/$2"
  printf '#!/bin/sh\necho "%s"\nexit 0\n' "$3" >"$1/$2"
  chmod +x "$1/$2"
}

run_install() {
  # Run install.sh with PATH=<shim>[:<tail>] against an unreachable CDN; print rc then stderr.
  local shim_dir="$1" path_tail="$2" script="${3:-${INSTALL}}" home
  home="$(mktemp -d "${WORK}/home-XXXXXX")"
  set +e
  env -i HOME="${WORK}" GOETHE_HOME="${home}/goethe" GOETHE_PKG_URL="file:///nonexistent" GOETHE_NO_LINK=1 \
    GOETHE_REQUIRE_GPU="${GOETHE_REQUIRE_GPU:-}" GOETHE_NVIDIA_VERSION_FILE="${GOETHE_NVIDIA_VERSION_FILE:-/nonexistent}" \
    GOETHE_PCI_DIR="${GOETHE_PCI_DIR:-${WORK}/no-pci}" GOETHE_OS_RELEASE="${GOETHE_OS_RELEASE:-/nonexistent}" \
    GOETHE_INSTALL_DRIVER="${GOETHE_INSTALL_DRIVER:-}" GOETHE_INSTALL_DEPS="${GOETHE_INSTALL_DEPS:-}" container="${container:-}" PATH="${shim_dir}${path_tail:+:${path_tail}}" \
    bash "${script}" 2>"${home}/stderr.txt" >/dev/null
  local rc=$?
  set -e
  echo "${rc}"
  /bin/cat "${home}/stderr.txt"
  [[ -e "${home}/goethe" ]] && echo "HOME_CREATED"
  return 0
}

expect_fail() {
  # Run a case that must fail and print pattern on stderr.
  local name="$1" pattern="$2" shim="$3" tail="$4" out rc
  out="$(run_install "${shim}" "${tail}")"
  rc="${out%%$'\n'*}"
  if [[ "${rc}" -eq 0 ]] || ! grep -q "${pattern}" <<<"${out}"; then
    echo "${name}: expected failure with '${pattern}', got rc=${rc}: ${out}"
    exit 1
  fi
}

if grep -nE '(^|[^_a-z])(kill|pkill|killall|fuser)([^_a-z]|$)' "${INSTALL}"; then
  echo "install.sh must never signal processes"
  exit 1
fi
if grep -nE '(nginx|NGINX_BIN\}?"?) +-[cps]( |$)|nohup[^#]*nginx' "${INSTALL}"; then
  echo "install.sh must not start or signal nginx (goethed manages it)"
  exit 1
fi

MAIN="$(sed -n '/^main() {/,/^}/p' "${INSTALL}")"
if grep -nE 'goethe (up|recommend)|systemctl|(^|[^_a-z])start_goethed' <<<"${MAIN}" \
  || [[ "$(grep -c 'restart_goethed' <<<"${MAIN}")" -ne 1 ]] \
  || ! grep -A1 'if \[\[ "${RESTART}" -eq 1 \]\]; then' <<<"${MAIN}" | grep -q 'restart_goethed'; then
  echo "install.sh main must not start goethed except to bring back one an upgrade stopped"
  exit 1
fi
if [[ "$(grep -v '^}' <<<"${MAIN}" | tail -n 1)" != "  get_started" ]]; then
  echo "install.sh must end with the Get started block"
  exit 1
fi
GOT="$(LC_ALL=C bash -c "$(sed -n '/^get_started() {/,/^}/p' "${INSTALL}")"$'\n'get_started)"
if [[ "${GOT}" != $'\nGet started:\n\n\t→ goethe start' ]]; then
  echo "install.sh Get started block is wrong: ${GOT}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-curl")"
fake "${SHIM}" curl "curl 7.40.0"
expect_fail curl "curl 7.40.0 is below minimum 7.50" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-zstd")"
fake "${SHIM}" zstd "*** Zstandard CLI (64-bit) v1.2.0"
expect_fail zstd "below minimum 1.3" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-openssl")"
fake "${SHIM}" openssl "OpenSSL 1.0.2k"
expect_fail openssl "below minimum 1.1.1" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-glibc")"
fake "${SHIM}" getconf "glibc 2.28"
expect_fail glibc "glibc 2.28 is below minimum 2.30" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-driver")"
fake "${SHIM}" nvidia-smi "575.57.08"
expect_fail driver "NVIDIA driver 575.57.08 is older than 580" "${SHIM}" "/usr/bin:/bin"

SHIM="$(make_shim_dir "${WORK}/shim-nogpu")"
GOETHE_REQUIRE_GPU=1 expect_fail require-gpu "no NVIDIA GPU found on this machine.*(GOETHE_REQUIRE_GPU=1)" "${SHIM}" ""

OUT="$(run_install "${SHIM}" "")"
if ! grep -q "WARNING: no NVIDIA GPU found" <<<"${OUT}" || grep -q "older than" <<<"${OUT}"; then
  echo "no-GPU warning case: ${OUT}"
  exit 1
fi

SHIM="$(make_shim_dir "${WORK}/shim-nodevices")"
fake "${SHIM}" nvidia-smi "No devices were found"
OUT="$(run_install "${SHIM}" "")"
if grep -q "unbound variable" <<<"${OUT}" || ! grep -q "WARNING: no NVIDIA GPU found" <<<"${OUT}"; then
  echo "nvidia-smi without devices must read as no driver: ${OUT}"
  exit 1
fi
printf 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  575.57.08  Release Build\n' >"${WORK}/nvrm"
GOETHE_NVIDIA_VERSION_FILE="${WORK}/nvrm" expect_fail driver-module "NVIDIA driver 575.57.08 is older than 580" "${SHIM}" ""

SHIM="$(make_shim_dir "${WORK}/shim-trunc")"
LINES="$(wc -l <"${INSTALL}")"
for keep in $((LINES / 3)) $((LINES / 2)) $((LINES - 1)); do
  head -n "${keep}" "${INSTALL}" >"${WORK}/install-trunc.sh"
  OUT="$(run_install "${SHIM}" "/usr/bin:/bin" "${WORK}/install-trunc.sh")"
  if grep -q "HOME_CREATED" <<<"${OUT}"; then
    echo "install.sh truncated to ${keep} lines still created GOETHE_HOME: ${OUT}"
    exit 1
  fi
done

UNITS="${WORK}/units"
mkdir -p "${UNITS}"
unit_home() {
  # install.sh's system_unit_home run against the unit dir under WORK.
  GOETHE_SYSTEMD_UNIT_DIR="${UNITS}" bash -c "$(sed -n '/^system_unit_home() {/,/^}/p' "${INSTALL}")"$'\n'system_unit_home
}
if [[ -n "$(unit_home)" ]]; then
  echo "system_unit_home must be empty without a unit"
  exit 1
fi
printf '[Service]\nEnvironment=GOETHE_HOME=/srv/a\nExecStart=/srv/a/bin/goethed\n' >"${UNITS}/goethed.service"
if [[ "$(unit_home)" != "/srv/a" ]]; then
  echo "system_unit_home misread the installer's unit: $(unit_home)"
  exit 1
fi
printf '[Service]\nEnvironment="GOETHE_HOME=/srv/my home/.goethe"\nEnvironmentFile=-/srv/x/goethe.env\n' \
  >"${UNITS}/goethed.service"
if [[ "$(unit_home)" != "/srv/my home/.goethe" ]]; then
  echo "system_unit_home misread goethe service enable's quoted unit: $(unit_home)"
  exit 1
fi

recorder() {
  # A shim tool that appends its name and arguments to calls.log; sudo runs the rest of its command line.
  if [[ "$2" == "sudo" ]]; then
    printf '#!/bin/sh\necho "sudo $*" >>"%s"\nexec "$@"\n' "${WORK}/calls.log" >"$1/sudo"
  else
    printf '#!/bin/sh\necho "%s $*" >>"%s"\nexit 0\n' "$2" "${WORK}/calls.log" >"$1/$2"
  fi
  chmod +x "$1/$2"
}

driver_case() {
  # Run the installer on a machine with NVIDIA GPUs (DEVICES, default one Ampere); $1 os-release text, then env assignments.
  local release="$1" shim t d n=0
  shift
  shim="$(make_shim_dir "${WORK}/shim-drv-$RANDOM")"
  for t in sudo apt-get ubuntu-drivers dpkg dnf modprobe; do recorder "${shim}" "${t}"; done
  ln -sf "$(tool_path timeout)" "${shim}/timeout"
  for t in "${EXTRA_TOOLS[@]}"; do rm -f "${shim}/${t}" && cp "${WORK}/tools/${t}" "${shim}/${t}"; done
  rm -rf "${WORK}/pci"
  for d in ${DEVICES:-0x2204}; do
    n=$((n + 1))
    mkdir -p "${WORK}/pci/0000:0${n}:00.0"
    echo 0x10de >"${WORK}/pci/0000:0${n}:00.0/vendor"
    echo 0x030200 >"${WORK}/pci/0000:0${n}:00.0/class"
    echo "${d}" >"${WORK}/pci/0000:0${n}:00.0/device"
  done
  printf '%s\n' "${release}" >"${WORK}/os-release"
  : >"${WORK}/calls.log"
  (
    export GOETHE_PCI_DIR="${WORK}/pci" GOETHE_OS_RELEASE="${WORK}/os-release" "$@"
    run_install "${shim}" ""
  )
}

UBUNTU=$'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"'
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1)"
if ! grep -q "sudo timeout 900 ubuntu-drivers install --gpgpu nvidia:580-server-open" "${WORK}/calls.log" \
  || ! grep -q "sudo modprobe nvidia" "${WORK}/calls.log" || grep -q "Goethe installs anyway" <<<"${OUT}"; then
  echo "ubuntu driver install case: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=0)"
if [[ -s "${WORK}/calls.log" ]] || ! grep -q "skipped (GOETHE_INSTALL_DRIVER=0)" <<<"${OUT}"; then
  echo "declined driver install case: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1 container=docker)"
if [[ -s "${WORK}/calls.log" ]] || ! grep -q "this is a container, so fix the NVIDIA driver" <<<"${OUT}"; then
  echo "container driver case: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi
OUT="$(driver_case "${UBUNTU}")"
if [[ -s "${WORK}/calls.log" ]] || ! grep -q "rerun with GOETHE_INSTALL_DRIVER=1" <<<"${OUT}"; then
  echo "no-terminal driver case: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi
ROCKY=$'ID="rocky"\nID_LIKE="rhel centos fedora"\nVERSION_ID="9.4"'
OUT="$(driver_case "${ROCKY}" GOETHE_INSTALL_DRIVER=1)"
if ! grep -q "dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo" \
  "${WORK}/calls.log" || ! grep -q "dnf -y install nvidia-open" "${WORK}/calls.log"; then
  echo "rocky driver install case: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi

mkdir -p "${WORK}/tools"
tool() {
  # A fake tool under WORK/tools: $1 name, $2 shell body.
  printf '#!/bin/sh\n%s\n' "$2" >"${WORK}/tools/$1"
  chmod +x "${WORK}/tools/$1"
}
smi() {
  # A fake nvidia-smi: $1 what -L lists, $2 the driver version line, $3 the index,name,compute_cap lines.
  tool nvidia-smi "case \"\$*\" in
  -L) printf '%b' '$1' ;;
  *driver_version*) printf '%b' '$2' ;;
  *compute_cap*) printf '%b' '$3' ;;
esac"
}
expect_out() {
  # $1 case name, $2 grep pattern that must be in OUT, $3 optional pattern that must not.
  if ! grep -q -- "$2" <<<"${OUT}" || { [[ -n "${3:-}" ]] && grep -q -- "$3" <<<"${OUT}"; }; then
    echo "$1: expected '$2'${3:+ and no '$3'}: $(cat "${WORK}/calls.log") ${OUT}"
    exit 1
  fi
}
EXTRA_TOOLS=()

tool modinfo "exit 0"
EXTRA_TOOLS=(modinfo)
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1)"
expect_out module-not-loaded "kernel module is not loaded" "Goethe installs anyway"
if ! grep -q "sudo modprobe nvidia" "${WORK}/calls.log" || grep -q ubuntu-drivers "${WORK}/calls.log"; then
  echo "module-not-loaded must load the module, not reinstall: $(cat "${WORK}/calls.log")"
  exit 1
fi
tool modprobe "echo \"modprobe \$*\" >>${WORK}/calls.log; echo \"modprobe: ERROR: could not insert 'nvidia': Key was rejected by service\" >&2; exit 1"
EXTRA_TOOLS=(modinfo modprobe)
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1)"
expect_out secure-boot "Secure Boot blocks the NVIDIA module"

printf 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  Release Build\n' >"${WORK}/nvrm-closed"
printf 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.82.07  Release Build\n' >"${WORK}/nvrm-open"
smi "" "No devices were found\n" ""
EXTRA_TOOLS=(nvidia-smi)
OUT="$(DEVICES=0x2b85 driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1 GOETHE_NVIDIA_VERSION_FILE="${WORK}/nvrm-closed")"
expect_out closed-blackwell "proprietary kernel module, which cannot start Blackwell"
if ! grep -q "ubuntu-drivers install --gpgpu nvidia:580-server-open" "${WORK}/calls.log"; then
  echo "closed module on Blackwell must switch to the open driver: $(cat "${WORK}/calls.log")"
  exit 1
fi
OUT="$(DEVICES=0x2b85 driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=1 GOETHE_NVIDIA_VERSION_FILE="${WORK}/nvrm-open")"
expect_out failed-init "loaded but the GPU did not start; the card failed to initialise" "proprietary"
if [[ -s "${WORK}/calls.log" ]]; then
  echo "a GPU that failed to initialise must not trigger a driver install: $(cat "${WORK}/calls.log")"
  exit 1
fi
smi "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-x)\n" "580.82.07\n" "0, NVIDIA GeForce RTX 5090, 12.0\n"
OUT="$(DEVICES="0x2b85 0x2b85" driver_case "${UBUNTU}")"
expect_out partial "1 of 2 NVIDIA GPUs did not start and will be skipped"
smi "GPU 0: Tesla T4 (UUID: GPU-x)\n" "580.82.07\n" "0, Tesla T4, 7.5\n"
OUT="$(DEVICES=0x1eb8 driver_case "${UBUNTU}")"
expect_out old-arch "no supported GPU: gpu0 Tesla T4 (SM 7.5)"
smi "GPU 0: A (UUID: a)\nGPU 1: B (UUID: b)\n" "580.82.07\n" "0, Tesla T4, 7.5\n1, NVIDIA L40S, 8.9\n"
OUT="$(DEVICES="0x1eb8 0x26b9" driver_case "${UBUNTU}")"
expect_out mixed-arch "gpu0 Tesla T4 (SM 7.5) is older than Goethe supports" "no supported GPU"

EXTRA_TOOLS=()
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=0 GOETHE_INSTALL_DEPS=1)"
if ! grep -q "apt-get install -y -qq build-essential" "${WORK}/calls.log"; then
  echo "a missing compiler must be installed with consent: $(cat "${WORK}/calls.log") ${OUT}"
  exit 1
fi
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=0)"
expect_out no-compiler "rerun with GOETHE_INSTALL_DEPS=1"
tool df "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 100 90 1048576 90%% /\\n'"
EXTRA_TOOLS=(df)
OUT="$(driver_case "${UBUNTU}" GOETHE_INSTALL_DRIVER=0)"
expect_out low-disk "only 1 GB free at"

echo "stage6_installer_checks ok"
