#!/usr/bin/env bash
# Check that this Linux host can run a capped nested Windows 11 VM for install.ps1 iteration (no ISO download, no guest start).

set -euo pipefail

ok=1
say() { printf 'wsl-vm: %s\n' "$*"; }
fail() { say "FAIL: $*"; ok=0; }

if [[ "$(uname -s)" != Linux ]]; then
  fail "host OS is not Linux"
fi

if [[ -r /sys/module/kvm_intel/parameters/nested ]]; then
  nested="$(tr -d '[:space:]' </sys/module/kvm_intel/parameters/nested)"
  [[ "${nested}" == "Y" || "${nested}" == "1" ]] || fail "kvm_intel nested=${nested} (want Y); echo Y | sudo tee /sys/module/kvm_intel/parameters/nested"
  say "kvm_intel nested=${nested}"
elif [[ -r /sys/module/kvm_amd/parameters/nested ]]; then
  nested="$(tr -d '[:space:]' </sys/module/kvm_amd/parameters/nested)"
  [[ "${nested}" == "Y" || "${nested}" == "1" ]] || fail "kvm_amd nested=${nested} (want 1)"
  say "kvm_amd nested=${nested}"
else
  fail "no kvm_intel/kvm_amd nested parameter (is kvm loaded?)"
fi

[[ -e /dev/kvm ]] || fail "/dev/kvm missing"
[[ -r /dev/kvm && -w /dev/kvm ]] || fail "/dev/kvm not read/write for $(id -un) (add user to kvm/libvirt)"

command -v qemu-system-x86_64 >/dev/null || fail "qemu-system-x86_64 not installed"
command -v virsh >/dev/null 2>&1 && say "virsh present" || say "virsh optional (virt-manager/libvirt)"

mem_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
mem_gb=$((mem_kb / 1024 / 1024))
say "MemAvailable ~${mem_gb} GiB (want >=12 GiB free before an 8 GiB Windows guest)"
if (( mem_gb < 12 )); then
  say "WARN: free RAM is tight; close apps or use a remote nested Windows VM so this laptop does not thrash"
fi

say "recommended guest caps: 4 vCPU, 8 GiB RAM, 64 GiB sparse disk, snapshot pre-wsl"
say "see tests/wsl/README.md"

[[ "${ok}" -eq 1 ]] || exit 1
say "host looks ready for a nested Windows VM"
