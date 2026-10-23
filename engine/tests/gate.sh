#!/usr/bin/env bash
# Build goethe fresh, then run stage-gate tests 0-6, each in its own session with a timeout and owned-group reaping.

set -euo pipefail

USAGE="usage: gate.sh <0..6|all> [--strict] [--only <file>]"
STAGE="${1:?${USAGE}}"
shift || true
STRICT=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1 ;;
    --only) ONLY="${2:?--only needs a test file name}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
case "${STAGE}" in
  all) STAGES=(0 1 2 3 4 5 6) ;;
  [0-6]) STAGES=("${STAGE}") ;;
  *) echo "${USAGE}" >&2; exit 2 ;;
esac

GOETHE_PKG="$(cd "${GOETHE_PKG:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
export GOETHE_PKG
export GOETHE_DEV_HOME="${GOETHE_DEV_HOME:-${GOETHE_HOME:-/scratch/ternary-flash/goethe-home}}"
export GOETHE_DEV_PY="${GOETHE_DEV_PY:-${GOETHE_DEV_HOME}/py-env/bin/python}"
PY="${GOETHE_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${GOETHE_DEV_PY}" ]]; then PY="${GOETHE_DEV_PY}"; else PY="$(command -v python3)"; fi
fi
export GOETHE_PYTHON="${PY}"
unset GOETHE_SOCK GOETHE_MUX_PORT GOETHE_PUBLIC_PORT GOETHE_ENGINE_PORTS GOETHE_REGISTRY_URL GOETHE_PKG_URL GOETHE_PKG_TOKEN
export GOETHE_GATE_RUN_ID="$$"
export GOETHE_GATE_PARENT="$$"
export PYTHONPATH="${GOETHE_PKG}/tests${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export GOETHE_TEST_GPUS="${GOETHE_TEST_GPUS:-2,3}"
TEST_TIMEOUT="${GOETHE_TEST_TIMEOUT:-2400}"

GATE_TMP="$(mktemp -d "/tmp/goethe-test-${GOETHE_GATE_RUN_ID}-gate-XXXXXX")"
export GOETHE_HOME="${GATE_TMP}/no-home"
CURRENT_SESSION=""

gate_cleanup() {
  rm -rf "${GATE_TMP}" 2>/dev/null || true
}

gate_abort() {
  # Stop the running test session (ours by construction), clean up, and exit with the signal status.
  if [[ -n "${CURRENT_SESSION}" ]]; then
    kill -TERM -- "-${CURRENT_SESSION}" 2>/dev/null || true
    wait "${CURRENT_SESSION}" 2>/dev/null || true
    "${PY}" -c "from lib.goethe_test import reap_run_artifacts; reap_run_artifacts(${CURRENT_SESSION})" || true
  fi
  gate_cleanup
  exit "$1"
}

trap gate_cleanup EXIT
trap 'gate_abort 143' TERM
trap 'gate_abort 130' INT

reap_stale_tmp() {
  # Delete temp paths older than an hour, but only those of runs whose owning process is gone.
  local now p rest id
  now=$(date +%s)
  shopt -s nullglob
  for p in /tmp/goethe-test-*; do
    rest="${p#/tmp/goethe-test-}"
    id="${rest%%-*}"
    [[ "${id}" =~ ^[0-9]+$ ]] || continue
    kill -0 "${id}" 2>/dev/null && continue
    [[ "$(stat -c %Y "${p}" 2>/dev/null || echo "${now}")" -lt $((now - 3600)) ]] && rm -rf "${p}"
  done
  shopt -u nullglob
}
reap_stale_tmp

FAIL=0
SKIP=0
PASS=0
RESULT_LINES=()
SKIP_LINES=()

record() {
  # Append one result row and count it.
  local name="$1" status="$2" secs="$3" note="$4"
  RESULT_LINES+=("$(printf '%-30s %-8s %7ss %s' "${name}" "${status}" "${secs}" "${note}")")
  case "${status}" in
    PASS) PASS=$((PASS + 1)) ;;
    SKIP) SKIP=$((SKIP + 1)); SKIP_LINES+=("${name}: ${note}") ;;
    *) FAIL=$((FAIL + 1)) ;;
  esac
}

cargo_step() {
  # Run one cargo command in GOETHE_PKG and record PASS or FAIL with the log tail.
  local name="$1"
  shift
  local log="${GATE_TMP}/${name// /_}.txt" start rc
  start=$(date +%s)
  set +e
  (cd "${GOETHE_PKG}" && nice -n 10 "$@") >"${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    record "${name}" PASS "$(($(date +%s) - start))" ""
  else
    record "${name}" FAIL "$(($(date +%s) - start))" "$(grep -E '^(error|warning)' "${log}" | head -3 | tr '\n' ' ')"
    tail -40 "${log}" >&2
  fi
  return ${rc}
}

build_step() {
  # Step 0: build, unit-test and clippy the workspace, then pin tests to binaries newer than every source.
  if [[ ! -f "${GOETHE_PKG}/Cargo.toml" ]]; then
    record "step0 sources" FAIL 0 "no Cargo.toml in ${GOETHE_PKG}"
    return 1
  fi
  cargo_step "cargo build" cargo build -j 12 --release --locked || return 1
  cargo_step "cargo test" cargo test -j 12 --workspace --release --locked -- --test-threads 4 || true
  cargo_step "cargo clippy" cargo clippy -j 12 --workspace --release --all-targets --locked -- -D warnings || true
  export GOETHE_BIN_DIR="${GOETHE_BIN_DIR:-${GOETHE_PKG}/target/release}"
  local b stale
  for b in goethe goethed; do
    if [[ ! -x "${GOETHE_BIN_DIR}/${b}" ]]; then
      record "step0 binaries" FAIL 0 "missing ${GOETHE_BIN_DIR}/${b}"
      return 1
    fi
    stale="$(find "${GOETHE_PKG}/crates/core" "${GOETHE_PKG}/crates/${b}" -type f \
      \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "${GOETHE_BIN_DIR}/${b}" | head -3 || true)"
    if [[ -n "${stale}" ]]; then
      record "step0 binaries" FAIL 0 "${GOETHE_BIN_DIR}/${b} is older than: $(echo "${stale}" | tr '\n' ' ')"
      return 1
    fi
  done
  record "step0 binaries" PASS 0 "${GOETHE_BIN_DIR}"
}

collect_tests() {
  # Print the test files of one stage, honoring the stage-1 allow list and --only.
  local n="$1" f base
  shopt -s nullglob
  for f in "${GOETHE_PKG}/tests/stage${n}_"*.py "${GOETHE_PKG}/tests/stage${n}_"*.sh; do
    base="$(basename "${f}")"
    if [[ "${n}" == "1" ]]; then
      case "${base}" in
        stage1_pyenv.py | stage1_layer.py | stage1_moe.py | stage1_e2e_model.py) ;;
        *) continue ;;
      esac
    fi
    if [[ -n "${ONLY}" && "${base}" != "${ONLY}" && "${f}" != *"${ONLY}"* ]]; then
      continue
    fi
    echo "${f}"
  done | sort
  shopt -u nullglob
}

run_one() {
  # Run one test in a new session under timeout, then reap the groups it owned and check hygiene.
  local testpath="$1" base pre_json out err start rc status note secs
  base="$(basename "${testpath}")"
  pre_json="$("${PY}" -c "import json; from lib.goethe_test import gpu_mem_snapshot, test_gpus; print(json.dumps(gpu_mem_snapshot(test_gpus())))")" \
    || pre_json="{}"
  out="${GATE_TMP}/out-${base}.txt"
  err="${GATE_TMP}/err-${base}.txt"
  local runner=("${PY}" "${testpath}")
  if [[ "${testpath}" == *.sh ]]; then runner=(bash "${testpath}"); fi
  start=$(date +%s.%N)
  set +e
  setsid timeout --kill-after=30 "${TEST_TIMEOUT}" "${runner[@]}" >"${out}" 2>"${err}" </dev/null &
  CURRENT_SESSION=$!
  wait "${CURRENT_SESSION}"
  rc=$?
  set -e
  secs=$("${PY}" -c "print(round($(date +%s.%N) - ${start}, 1))")
  note=""
  case ${rc} in
    0) status=PASS ;;
    75) status=SKIP; note="$(head -1 "${out}")" ;;
    77) status=FAIL; note="PENDING (missing binary or subcommand): $(head -1 "${out}")" ;;
    124 | 137) status=FAIL; note="TIMEOUT after ${TEST_TIMEOUT}s: $(tail -3 "${err}" | tr '\n' ' ')" ;;
    *) status=FAIL; note="$(tail -5 "${err}" | tr '\n' ' ')" ;;
  esac
  set +e
  "${PY}" -c "from lib.goethe_test import gate_after_test; gate_after_test('''${pre_json}''', ${CURRENT_SESSION})" \
    >"${GATE_TMP}/hygiene-${base}.txt" 2>&1
  local hrc=$?
  set -e
  CURRENT_SESSION=""
  if [[ ${hrc} -ne 0 ]]; then
    status=FAIL
    note="hygiene: $(tail -3 "${GATE_TMP}/hygiene-${base}.txt" | tr '\n' ' ') ${note}"
  fi
  record "${base}" "${status}" "${secs}" "${note}"
}

if build_step; then
  TESTS=()
  for st in "${STAGES[@]}"; do
    while IFS= read -r t; do
      [[ -n "${t}" ]] && TESTS+=("${t}")
    done < <(collect_tests "${st}")
  done
  if [[ ${#TESTS[@]} -eq 0 ]]; then
    record "collect" FAIL 0 "no tests collected for stage ${STAGE}${ONLY:+ --only ${ONLY}}"
  fi
  for t in "${TESTS[@]}"; do
    run_one "${t}"
  done
fi

set +e
"${PY}" "${GOETHE_PKG}/tests/lint_style.py" "${GOETHE_PKG}" >"${GATE_TMP}/lint.txt" 2>&1
lint_rc=$?
set -e
if [[ ${lint_rc} -eq 0 ]]; then
  record "lint_style" PASS 0 ""
else
  record "lint_style" FAIL 0 "$(head -3 "${GATE_TMP}/lint.txt" | tr '\n' ' ')"
  head -20 "${GATE_TMP}/lint.txt"
fi

printf '%-30s %-8s %8s %s\n' "TEST" "STATUS" "SEC" "NOTE"
printf '%s\n' "${RESULT_LINES[@]}"
if [[ ${SKIP} -gt 0 ]]; then
  echo "==== ${SKIP} TEST(S) SKIPPED: NOT VERIFIED BY THIS RUN ===="
  printf '  %s\n' "${SKIP_LINES[@]}"
fi
echo "gate stage ${STAGE}: PASS=${PASS} FAIL=${FAIL} SKIP=${SKIP}"
if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
if [[ ${STRICT} -eq 1 && ${SKIP} -gt 0 ]]; then
  exit 1
fi
exit 0
