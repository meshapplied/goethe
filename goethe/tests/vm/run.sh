#!/usr/bin/env bash
# Install goethe into fresh Ubuntu 24.04, Debian 12 and Rocky 9 KVM VMs from signed test packages (scenarios.sh); only docker and /dev/kvm on the host; DISTROS, KEEP=1, UV_BIN adds the model install.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="${ROOT}/tests/vm"
WORK="${HERE}/.work"
BIN="${ROOT}/target-portable/x86_64-unknown-linux-gnu/release"
NGINX="${ROOT}/dist/cdn/bin/linux-x86_64/nginx"
UV="${UV_BIN:-}"
[[ -x "${BIN}/goethe" && -x "${NGINX}" ]] || { echo "build first: GOETHE_CARGO_FEATURES=goethe/test-release-key packaging/build_bin.sh and packaging/build_nginx_linux.sh" >&2; exit 1; }
"${BIN}/goethe" --version >/dev/null

rm -rf "${WORK}/cdn" "${WORK}/keys"
mkdir -p "${WORK}/cdn" "${WORK}/keys" "${WORK}/images"
openssl ecparam -genkey -name prime256v1 -noout -out "${WORK}/keys/test.key"
openssl ec -in "${WORK}/keys/test.key" -pubout -out "${WORK}/keys/test_pub.pem" 2>/dev/null
(cd "${ROOT}" && cargo run --locked -q -p goethe-core --example registry -- build registry.src.json "${WORK}/keys/registry.json")
openssl dgst -sha256 -sign "${WORK}/keys/test.key" -out "${WORK}/keys/registry.json.sig" "${WORK}/keys/registry.json"
version="$(grep -E '^version\s*=' "${ROOT}/crates/goethe/Cargo.toml" | head -n1 | sed 's/.*"\(.*\)".*/\1/')"

package() {
  # One signed package at cdn/$1; $2=extra appends a byte to goethe and goethed (another build of the same version); a v/<version>/ copy for pinned installs.
  local dir="${WORK}/cdn/$1" f
  mkdir -p "${dir}/bin/linux-x86_64" "${dir}/pyenv" "${dir}/v1"
  cp "${BIN}/goethe" "${BIN}/goethed" "${NGINX}" ${UV:+"${UV}"} "${dir}/bin/linux-x86_64/"
  if [[ "${2:-}" == extra ]]; then
    for f in goethe goethed; do printf '\n' >>"${dir}/bin/linux-x86_64/${f}"; done
  fi
  cp "${ROOT}"/pyenv/requirements*.lock "${ROOT}/pyenv/bootstrap.sh" "${dir}/pyenv/"
  cp "${WORK}/keys/registry.json" "${WORK}/keys/registry.json.sig" "${dir}/v1/"
  cp "${ROOT}/../LICENSE" "${ROOT}/../NOTICE" "${dir}/"
  echo "${version}" >"${dir}/VERSION"
  bash "${ROOT}/packaging/subst_install_pubkey.sh" "${ROOT}/cdn/install.sh" "${WORK}/keys/test_pub.pem" "${dir}/install.sh"
  sign "${dir}"
  mkdir -p "${dir}/v/${version}"
  (cd "${dir}" && find . -mindepth 1 -maxdepth 1 ! -name v -exec cp -R {} "v/${version}/" \;)
}

sign() {
  (cd "$1" && find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sig ! -path './v/*' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do
    echo "$(sha256sum "${f}" | awk '{print $1}')  ${f}"
  done >SHA256SUMS)
  openssl dgst -sha256 -sign "${WORK}/keys/test.key" -out "$1/SHA256SUMS.sig" "$1/SHA256SUMS"
}

package a
package b extra
cp -R "${WORK}/cdn/a" "${WORK}/cdn/t"
printf 'tampered' >>"${WORK}/cdn/t/bin/linux-x86_64/goethe"
printf 'tampered' >>"${WORK}/cdn/t/v/${version}/bin/linux-x86_64/goethe"

docker run --rm --name goethe-vm-tests --device /dev/kvm \
  -e DISTROS="${DISTROS:-ubuntu debian rocky}" -e VERSION="${version}" \
  -v "${WORK}:/work" -v "${HERE}:/vm:ro" ubuntu:24.04 bash /vm/inside.sh
status=$?
[[ "${KEEP:-0}" == 1 ]] || rm -rf "${WORK}/cdn"
exit "${status}"
