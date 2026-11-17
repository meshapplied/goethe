#!/usr/bin/env bash
# Mutation check of the end-to-end cells: each NAME.patch re-introduces one bug we shipped, and the cells must then fail.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${HERE}" rev-parse --show-toplevel)"
ONLY="${1:-}"
[[ "${ONLY}" == --help ]] && { echo "usage: run.sh [NAME]; NAME.kill holds the command that must fail (default: python3 tests/e2e_cli.py)"; exit 0; }
TARGET="${MUTANT_TARGET:-${ROOT}/engine/target}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
died=() survived=() broken=()

tree() {
  # A fresh copy of the committed tree with every source touched, so cargo rebuilds what the last mutant left behind.
  rm -rf "${WORK}/t"
  mkdir "${WORK}/t"
  git -C "${ROOT}" archive HEAD | tar -x -C "${WORK}/t"
  find "${WORK}/t/engine/crates" -name '*.rs' -exec touch {} +
}

build() {
  (cd "${WORK}/t/engine" && CARGO_TARGET_DIR="${TARGET}" cargo build --locked --features goethe/test-release-key >"${WORK}/build.log" 2>&1)
}

check() {
  # The cells (or the named command) against the tree as it stands: exit 0 means they all pass.
  local kill="$1"
  if [[ -f "${kill}" ]]; then
    (cd "${WORK}/t" && bash "${kill}" >"${WORK}/kill.log" 2>&1)
  else
    (cd "${WORK}/t/engine" && E2E_BIN="${TARGET}/debug" python3 tests/e2e_cli.py >"${WORK}/kill.log" 2>&1)
  fi
}

tree
if ! build || ! check "none"; then
  echo "the unmutated tree does not pass its own cells; fix that first"
  tail -30 "${WORK}/build.log" "${WORK}/kill.log"
  exit 1
fi
for patch in "${HERE}"/*.patch; do
  name="$(basename "${patch}" .patch)"
  [[ -n "${ONLY}" && "${name}" != *"${ONLY}"* ]] && continue
  tree
  if ! (cd "${WORK}/t" && git apply "${patch}") 2>"${WORK}/apply.log"; then
    broken+=("${name} (does not apply: $(head -1 "${WORK}/apply.log"))")
    continue
  fi
  if ! build; then
    broken+=("${name} (does not compile: $(grep -m1 '^error' "${WORK}/build.log"))")
    continue
  fi
  kill="${HERE}/${name}.kill"
  [[ -f "${kill}" ]] && cp "${kill}" "${WORK}/kill.sh" && kill="${WORK}/kill.sh" || kill="none"
  if check "${kill}"; then
    survived+=("${name}")
    echo "SURVIVED  ${name}"
  else
    died+=("${name}")
    echo "died      ${name}: $(grep -E '^(FAIL|ERROR)' "${WORK}/kill.log" | head -2 | tr '\n' ' ')"
  fi
done
echo
echo "died ${#died[@]}, survived ${#survived[@]}, broken ${#broken[@]}"
printf 'BROKEN    %s\n' "${broken[@]}"
exit $(( ${#survived[@]} + ${#broken[@]} > 0 ))
