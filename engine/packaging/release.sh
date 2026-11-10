#!/usr/bin/env bash
# Release in two commands: prepare opens the version bump PR from main, tag starts the release workflow once it is merged.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
CRATES=(core goethe goethed)
PACKAGES=(goethe-core goethe goethed)

usage() {
  cat >&2 <<'EOF'
usage: engine/packaging/release.sh prepare <X.Y.Z> [title]
       engine/packaging/release.sh tag <X.Y.Z>

prepare  On a fresh release-<X.Y.Z> branch from origin/main: set the version in the three crates and
         Cargo.lock, date the "## Unreleased" changelog section, check the build, commit, push, open the PR.
tag      After the PR is merged: tag origin/main as v<X.Y.Z> (its crate version must match) and push the
         tag, which starts the release workflow; then approve "publish" in the Actions tab.
EOF
  exit 2
}

die() {
  echo "release: $*" >&2
  exit 1
}

current_version() {
  sed -n 's/^version = "\(.*\)"/\1/p' "${REPO}/engine/crates/goethe/Cargo.toml" | head -n1
}

newer() {
  [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" == "$1" ]]
}

set_version() {
  # The package version of each crate and its Cargo.lock entry; nothing else in the files changes.
  local old="$1" new="$2" crate pkg
  for crate in "${CRATES[@]}"; do
    sed -i "0,/^version = \"${old}\"$/s//version = \"${new}\"/" "${REPO}/engine/crates/${crate}/Cargo.toml"
  done
  for pkg in "${PACKAGES[@]}"; do
    awk -v pkg="${pkg}" -v old="${old}" -v new="${new}" '
      $0 == "name = \"" pkg "\"" { hit = 1; print; next }
      hit && $0 == "version = \"" old "\"" { print "version = \"" new "\""; hit = 0; next }
      { hit = 0; print }
    ' "${REPO}/engine/Cargo.lock" >"${REPO}/engine/Cargo.lock.tmp"
    mv "${REPO}/engine/Cargo.lock.tmp" "${REPO}/engine/Cargo.lock"
  done
}

unreleased_notes() {
  # The bullets under "## Unreleased", up to the next heading.
  awk '/^## Unreleased$/ { on = 1; next } /^## / { on = 0 } on && NF' "${REPO}/CHANGELOG.md"
}

prepare() {
  local version="$1" title="${2:-}" old branch notes
  [[ -z "$(git -C "${REPO}" status --porcelain)" ]] || die "the working tree has changes; commit or stash them first"
  git -C "${REPO}" fetch -q origin main --tags
  git -C "${REPO}" rev-parse -q --verify "refs/tags/v${version}" >/dev/null && die "v${version} is already tagged"
  branch="release-${version}"
  git -C "${REPO}" switch -q -c "${branch}" origin/main || die "could not create ${branch} from origin/main"
  old="$(current_version)"
  newer "${version}" "${old}" || die "${version} is not newer than ${old}"
  notes="$(unreleased_notes)"
  [[ -n "${notes}" ]] || die "CHANGELOG.md has no entries under ## Unreleased"
  set_version "${old}" "${version}"
  sed -i "0,/^## Unreleased$/s//## ${version} ($(date +%F))/" "${REPO}/CHANGELOG.md"
  [[ "$(current_version)" == "${version}" ]] || die "the version did not change in engine/crates/goethe/Cargo.toml"
  (cd "${REPO}/engine" && cargo check --locked -q) || die "cargo check --locked failed after the bump"
  git -C "${REPO}" add CHANGELOG.md engine/Cargo.lock engine/crates/*/Cargo.toml
  git -C "${REPO}" commit -q -m "v${version}${title:+: ${title}}"
  git -C "${REPO}" push -q -u origin "${branch}"
  gh pr create --base main --head "${branch}" --title "v${version}" \
    --body "$(printf 'Version bump to %s (from %s) and changelog.\n\n%s\n' "${version}" "${old}" "${notes}")"
  echo "Merge the PR, then run: engine/packaging/release.sh tag ${version}"
}

tag() {
  local version="$1" on_main run
  git -C "${REPO}" fetch -q origin main --tags
  git -C "${REPO}" rev-parse -q --verify "refs/tags/v${version}" >/dev/null && die "v${version} is already tagged"
  on_main="$(git -C "${REPO}" show origin/main:engine/crates/goethe/Cargo.toml | sed -n 's/^version = "\(.*\)"/\1/p' | head -n1)"
  [[ "${on_main}" == "${version}" ]] || die "origin/main is at ${on_main}, not ${version}; merge the release PR first"
  git -C "${REPO}" tag "v${version}" origin/main
  git -C "${REPO}" push -q origin "v${version}"
  echo "Tagged $(git -C "${REPO}" rev-parse --short origin/main) as v${version}; the release workflow starts now."
  for _ in 1 2 3 4 5 6; do
    run="$(gh run list --workflow release --branch "v${version}" --limit 1 --json url --jq '.[0].url' 2>/dev/null || true)"
    if [[ -n "${run}" ]]; then
      echo "Run: ${run}"
      break
    fi
    sleep 5
  done
  echo "Approve \"publish\" there when it waits. If only the Pages step fails, run the pages workflow on main with tag v${version}."
}

main() {
  [[ $# -ge 2 ]] || usage
  [[ "$2" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "the version must look like 0.1.10, got $2"
  case "$1" in
    prepare) prepare "$2" "${3:-}" ;;
    tag) tag "$2" ;;
    *) usage ;;
  esac
}

main "$@"
