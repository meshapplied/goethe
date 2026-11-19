#!/usr/bin/env bash
# Equivalence gates for in-place module splits (move-only bites).
# Usage:
#   equiv_split_gate.sh baseline                 # record goethe-core --lib test bare names + pass count
#   equiv_split_gate.sh check [NEW_RS ...]       # gates B–E; NEW_RS files checked for production LOC ≤ 500
#   equiv_split_gate.sh focused PARENT [NEW...]  # cargo test filters for parent:: and new::
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${HERE}" rev-parse --show-toplevel)"
GOETHE="${ROOT}/goethe"
BASE_DIR="${EQUIV_BASE:-${ROOT}/.equiv-split-base}"
MAX_NEW_PROD_LOC="${MAX_NEW_PROD_LOC:-500}"

die() { echo "equiv-gate FAIL: $*" >&2; exit 1; }
ok() { echo "equiv-gate OK: $*"; }

need_cargo() {
  command -v cargo >/dev/null || die "cargo not found"
  [[ -d "${GOETHE}" ]] || die "missing ${GOETHE}"
}

prod_loc() {
  # Lines before #[cfg(test)] (or whole file if none).
  local f="$1" n
  n="$(awk '/^#\[cfg\(test\)\]/{exit} {n++} END{print n+0}' "$f")"
  echo "$n"
}

bare_test_list() {
  # Strip module path; keep final ::name so moves across modules still match.
  cargo test -p goethe-core --lib -- --list 2>/dev/null \
    | sed -n 's/: test$//p' \
    | sed 's/.*:://' \
    | sort -u
}

cmd_baseline() {
  need_cargo
  mkdir -p "${BASE_DIR}"
  (
    cd "${GOETHE}"
    env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" \
      cargo test -p goethe-core --lib -- --test-threads=8 2>"${BASE_DIR}/test.err" \
      | tee "${BASE_DIR}/test.out" | tail -5
  )
  bare_test_list >"${BASE_DIR}/bare_tests.txt"
  grep -E '^test result:' "${BASE_DIR}/test.out" | tail -1 >"${BASE_DIR}/result.txt" || die "no test result line"
  # "test result: ok. N passed;"
  sed -n 's/.*ok\. \([0-9][0-9]*\) passed.*/\1/p' "${BASE_DIR}/result.txt" >"${BASE_DIR}/passed.txt"
  [[ -s "${BASE_DIR}/passed.txt" ]] || die "could not parse pass count from ${BASE_DIR}/result.txt"
  ok "baseline passed=$(cat "${BASE_DIR}/passed.txt") tests=$(wc -l <"${BASE_DIR}/bare_tests.txt") dir=${BASE_DIR}"
}

cmd_check() {
  need_cargo
  [[ -f "${BASE_DIR}/passed.txt" ]] || die "run baseline first (${BASE_DIR})"
  local new
  for new in "$@"; do
    [[ -f "${new}" ]] || new="${GOETHE}/crates/core/src/${new}"
    [[ -f "${new}" ]] || die "new module not found: $1"
    local loc
    loc="$(prod_loc "${new}")"
    if (( loc > MAX_NEW_PROD_LOC )); then
      die "new module ${new} production LOC ${loc} > ${MAX_NEW_PROD_LOC}"
    fi
    ok "LOC ${new} prod=${loc} (<=${MAX_NEW_PROD_LOC})"
  done

  (
    cd "${GOETHE}"
    env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" cargo check --workspace --locked -q
  ) || die "cargo check --workspace"
  ok "cargo check --workspace"

  (
    cd "${GOETHE}"
    env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" \
      cargo clippy -p goethe-core --locked --all-targets -- -D warnings
  ) || die "clippy -D warnings"
  ok "clippy goethe-core"

  python3 "${GOETHE}/tests/lint_style.py" || die "lint_style"
  ok "lint_style"

  (
    cd "${GOETHE}"
    env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" cargo check -p goethe -p goethed --locked -q
  ) || die "cargo check goethe goethed"
  ok "cargo check goethe goethed"

  local out passed
  out="$(mktemp)"
  (
    cd "${GOETHE}"
    env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" \
      cargo test -p goethe-core --lib -- --test-threads=8
  ) | tee "${out}" | tail -8
  grep -E '^test result: ok\.' "${out}" | tail -1 | grep -q ok || die "goethe-core lib tests failed"
  passed="$(sed -n 's/.*ok\. \([0-9][0-9]*\) passed.*/\1/p' "${out}" | tail -1)"
  [[ "${passed}" == "$(cat "${BASE_DIR}/passed.txt")" ]] \
    || die "pass count diverged: got ${passed} want $(cat "${BASE_DIR}/passed.txt")"
  ok "pass count ${passed}"

  bare_test_list >"${BASE_DIR}/bare_tests.after.txt"
  if ! diff -u "${BASE_DIR}/bare_tests.txt" "${BASE_DIR}/bare_tests.after.txt" >"${BASE_DIR}/bare_tests.diff"; then
    die "bare test names diverged (see ${BASE_DIR}/bare_tests.diff)"
  fi
  ok "bare test names identical ($(wc -l <"${BASE_DIR}/bare_tests.txt") names)"
  rm -f "${out}"
}

cmd_focused() {
  need_cargo
  local filter
  [[ $# -ge 1 ]] || die "focused PARENT [NEW...]"
  (
    cd "${GOETHE}"
    for filter in "$@"; do
      env -u CC -u CXX CC="${CC:-gcc}" CXX="${CXX:-g++}" \
        cargo test -p goethe-core --lib "${filter}::" -- --test-threads=8 \
        || die "focused tests failed: ${filter}::"
      ok "focused ${filter}::"
    done
  )
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
