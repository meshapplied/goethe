#!/usr/bin/env bash
# pkg.goethe.sh's tree from goethe-cdn-VERSION.tar.zst into a site dir: that release at the root (what older installers read) and the newest KEEP (default 4) under v/<version>/.

set -euo pipefail

TARBALL="$1"
SITE="$2"
KEEP="${3:-4}"
REPO="${GITHUB_REPOSITORY:-meshapplied/goethe}"
PUB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/release_pub.pem"

signed() {
  # A tree is served only when its SHA256SUMS is signed by the release key and every file it lists matches.
  openssl dgst -sha256 -verify "${PUB}" -signature "$1/SHA256SUMS.sig" "$1/SHA256SUMS" >/dev/null && (cd "$1" && sha256sum -c --quiet SHA256SUMS)
}

mkdir -p "${SITE}"
zstd -dc "${TARBALL}" | tar -C "${SITE}" --strip-components=1 -xf -
test -f "${SITE}/install.sh" && test -f "${SITE}/install.ps1" && test -f "${SITE}/SHA256SUMS.sig" && test -f "${SITE}/v1/registry.json"
signed "${SITE}" || { echo "${TARBALL} is not signed by the release key; nothing is served" >&2; exit 1; }
current="$(cat "${SITE}/VERSION")"
mkdir -p "${SITE}/v/${current}"
(cd "${SITE}" && find . -mindepth 1 -maxdepth 1 ! -name v -exec cp -R {} "v/${current}/" \;)
kept=1
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
while read -r tag; do
  ((kept < KEEP)) || break
  ver="${tag#v}"
  [[ "${ver}" != "${current}" && "${ver}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
  if gh release download "${tag}" --repo "${REPO}" --pattern "goethe-cdn-${ver}.tar.zst" --dir "${work}" >/dev/null 2>&1; then
    mkdir -p "${SITE}/v/${ver}"
    zstd -dc "${work}/goethe-cdn-${ver}.tar.zst" | tar -C "${SITE}/v/${ver}" --strip-components=1 -xf -
    test "$(cat "${SITE}/v/${ver}/VERSION")" = "${ver}"
    if ! signed "${SITE}/v/${ver}"; then
      echo "skipping ${tag}: not signed by the release key" >&2
      rm -rf "${SITE:?}/v/${ver}" "${work:?}/goethe-cdn-${ver}.tar.zst"
      continue
    fi
    rm -f "${work}/goethe-cdn-${ver}.tar.zst"
    kept=$((kept + 1))
  fi
done < <(gh release list --repo "${REPO}" --limit 30 --json tagName --jq '.[].tagName' | sort -rV)
du -sh "${SITE}"
ls "${SITE}/v"
