#!/usr/bin/env bash
# Equivalence gates for move-only module splits: baseline | check [NEW_RS...] | focused PARENT [NEW...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${HERE}" rev-parse --show-toplevel)"
GOETHE="${ROOT}/goethe"
BASE_DIR="${EQUIV_BASE:-${ROOT}/.equiv-split-base}"
MAX_NEW_PROD_LOC="${MAX_NEW_PROD_LOC:-500}"
export CC="${CC:-gcc}" CXX="${CXX:-g++}"

die() { echo "equiv-gate FAIL: $*" >&2; exit 1; }
ok() { echo "equiv-gate OK: $*"; }

need_cargo() {
  command -v cargo >/dev/null || die "cargo not found"
  [[ -d "${GOETHE}" ]] || die "missing ${GOETHE}"
}

cargo_in() {
  # Run cargo in the goethe workspace; keep CC/CXX without a broken ccache wrapper.
  (cd "${GOETHE}" && env -u CC -u CXX CC="${CC}" CXX="${CXX}" cargo "$@")
}

prod_loc() {
  awk '/^#\[cfg\(test\)\]/{exit} {n++} END{print n+0}' "$1"
}

write_bare_tests() {
  local out="$1"
  cargo_in test -p goethe-core --lib -- --list >"${BASE_DIR}/list.raw"
  sed -n 's/: test$//p' "${BASE_DIR}/list.raw" | sed 's/.*:://' | sort -u >"${out}"
  [[ -s "${out}" ]] || die "empty bare test list"
}

cmd_baseline() {
  need_cargo
  mkdir -p "${BASE_DIR}"
  cargo_in test -p goethe-core --lib -- --test-threads=8 >"${BASE_DIR}/test.out"
  tail -5 "${BASE_DIR}/test.out"
  grep -E '^test result: ok\.' "${BASE_DIR}/test.out" | tail -1 >"${BASE_DIR}/result.txt" \
    || die "goethe-core lib tests did not pass"
  sed -n 's/.*ok\. \([0-9][0-9]*\) passed.*/\1/p' "${BASE_DIR}/result.txt" >"${BASE_DIR}/passed.txt"
  [[ -s "${BASE_DIR}/passed.txt" ]] || die "could not parse pass count"
  write_bare_tests "${BASE_DIR}/bare_tests.txt"
  ok "baseline passed=$(cat "${BASE_DIR}/passed.txt") tests=$(wc -l <"${BASE_DIR}/bare_tests.txt")"
}

cmd_check() {
  need_cargo
  [[ -f "${BASE_DIR}/passed.txt" ]] || die "run baseline first (${BASE_DIR})"
  local new loc
  for new in "$@"; do
    [[ -f "${new}" ]] || new="${GOETHE}/crates/core/src/${new}"
    [[ -f "${new}" ]] || die "new module not found"
    loc="$(prod_loc "${new}")"
    (( loc <= MAX_NEW_PROD_LOC )) || die "new module ${new} production LOC ${loc} > ${MAX_NEW_PROD_LOC}"
    ok "LOC ${new} prod=${loc}"
  done

  cargo_in check --workspace --locked -q || die "cargo check --workspace"
  ok "cargo check --workspace"

  cargo_in clippy -p goethe-core --locked --all-targets -- -D warnings || die "clippy"
  ok "clippy goethe-core"

  python3 "${GOETHE}/tests/lint_style.py" || die "lint_style"
  ok "lint_style"

  cargo_in check -p goethe -p goethed --locked -q || die "cargo check goethe goethed"
  ok "cargo check goethe goethed"

  cargo_in test -p goethe-core --lib -- --test-threads=8 >"${BASE_DIR}/test.after.out"
  tail -5 "${BASE_DIR}/test.after.out"
  grep -E '^test result: ok\.' "${BASE_DIR}/test.after.out" | tail -1 | grep -q ok \
    || die "goethe-core lib tests failed"
  local passed
  passed="$(sed -n 's/.*ok\. \([0-9][0-9]*\) passed.*/\1/p' "${BASE_DIR}/test.after.out" | tail -1)"
  [[ "${passed}" == "$(cat "${BASE_DIR}/passed.txt")" ]] \
    || die "pass count diverged: got ${passed} want $(cat "${BASE_DIR}/passed.txt")"
  ok "pass count ${passed}"

  write_bare_tests "${BASE_DIR}/bare_tests.after.txt"
  diff -u "${BASE_DIR}/bare_tests.txt" "${BASE_DIR}/bare_tests.after.txt" >"${BASE_DIR}/bare_tests.diff" \
    || die "bare test names diverged (see ${BASE_DIR}/bare_tests.diff)"
  ok "bare test names identical ($(wc -l <"${BASE_DIR}/bare_tests.txt") names)"
}

cmd_focused() {
  need_cargo
  [[ $# -ge 1 ]] || die "focused PARENT [NEW...]"
  local filter
  for filter in "$@"; do
    cargo_in test -p goethe-core --lib "${filter}::" -- --test-threads=8 \
      || die "focused tests failed: ${filter}::"
    ok "focused ${filter}::"
  done
}

case "${1:-}" in
  baseline) shift; cmd_baseline "$@" ;;
  check) shift; cmd_check "$@" ;;
  focused) shift; cmd_focused "$@" ;;
  *)
    echo "usage: $0 baseline | check [NEW_RS...] | focused PARENT [NEW...]" >&2
    exit 2
    ;;
esac
