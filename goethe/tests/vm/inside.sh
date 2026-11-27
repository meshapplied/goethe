#!/usr/bin/env bash
# Inside the throwaway container: serve the test packages, boot each distro's cloud image under KVM, run scenarios.sh in it over ssh.

set -uo pipefail

export DEBIAN_FRONTEND=noninteractive
mkdir -p /work/apt/partial
if ! { compgen -G "/work/apt/*.deb" >/dev/null && dpkg -i /work/apt/*.deb >/dev/null 2>&1 && command -v qemu-system-x86_64 cloud-localds >/dev/null; }; then
  apt-get update -qq >/dev/null && apt-get install -y -qq -f -o Dir::Cache::archives=/work/apt qemu-system-x86 qemu-utils cloud-image-utils openssh-client python3 curl >/dev/null
fi
chmod -R a+rX /work/apt /work/images
echo "qemu ready after ${SECONDS} s"
ssh-keygen -q -t ed25519 -N "" -f /root/k
(cd /work/cdn && exec python3 -m http.server 8999 --bind 0.0.0.0 >/dev/null 2>&1) &

declare -A IMAGE=(
  [ubuntu]=https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
  [debian]=https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2
  [rocky]=https://dl.rockylinux.org/pub/rocky/9/images/x86_64/Rocky-9-GenericCloud-Base.latest.x86_64.qcow2
)
SSH=(ssh -i /root/k -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5 -p 2222 tester@127.0.0.1)
failed=0
for d in ${DISTROS}; do
  base="/work/images/${d}.qcow2"
  [[ -s "${base}" ]] || curl -fsSL -o "${base}" "${IMAGE[${d}]}" || { echo "FAIL ${d}: image download"; failed=1; continue; }
  chmod a+r "${base}"
  echo "${d} image ready after ${SECONDS} s"
  qemu-img create -q -f qcow2 -b "${base}" -F qcow2 "/tmp/${d}.qcow2" 20G
  printf '%s\n' "#cloud-config" >/tmp/user-data
  cat >>/tmp/user-data <<UD
users:
  - name: tester
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys: [$(cat /root/k.pub)]
ssh_pwauth: false
UD
  printf 'instance-id: %s\nlocal-hostname: goethe-%s\n' "${d}" "${d}" >/tmp/meta-data
  cloud-localds "/tmp/seed-${d}.iso" /tmp/user-data /tmp/meta-data
  qemu-system-x86_64 -enable-kvm -cpu host -m 4096 -smp 4 -display none -serial "file:/work/console-${d}.log" \
    -drive "file=/tmp/${d}.qcow2,if=virtio" -drive "file=/tmp/seed-${d}.iso,if=virtio,format=raw" \
    -netdev user,id=n,hostfwd=tcp:127.0.0.1:2222-:22 -device virtio-net-pci,netdev=n &
  qemu=$!
  up=0
  for _ in $(seq 1 120); do
    if "${SSH[@]}" true 2>/dev/null; then up=1; break; fi
    sleep 3
  done
  if [[ ${up} -eq 0 ]]; then
    echo "FAIL ${d}: the VM did not come up (see console-${d}.log)"
    failed=1
  else
    "${SSH[@]}" "cloud-init status --wait >/dev/null 2>&1 || true"
    echo "${d} booted after ${SECONDS} s"
    scp -q -i /root/k -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P 2222 \
      /vm/scenarios.sh /work/keys/test_pub.pem tester@127.0.0.1:/tmp/
    echo "=== ${d}"
    "${SSH[@]}" "bash /tmp/scenarios.sh ${d} ${VERSION}" 2>&1 | tee "/work/result-${d}.log"
    grep -q "^FAIL" "/work/result-${d}.log" && failed=1
    grep -q "^ALL PASS" "/work/result-${d}.log" || failed=1
  fi
  kill "${qemu}" 2>/dev/null
  wait "${qemu}" 2>/dev/null
  rm -f "/tmp/${d}.qcow2"
done
echo
for d in ${DISTROS}; do
  printf '%-7s %s\n' "${d}" "$(grep -cE '^PASS' "/work/result-${d}.log" 2>/dev/null || echo 0) passed, $(grep -cE '^FAIL' "/work/result-${d}.log" 2>/dev/null || echo '?') failed"
done
exit "${failed}"
