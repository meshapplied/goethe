#!/usr/bin/env bash
# Run goethe stage-gate tests for stages 0 through 6.

set -euo pipefail

STAGE="${1:?usage: gate.sh <0..6|all> [--strict] [--only <file>]}"
shift || true
STRICT=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1 ;;
    --only) ONLY="${2:?}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

export GOETHE_HOME="${GOETHE_HOME:-/scratch/ternary-flash/goethe-home}"
export GOETHE_PKG="${GOETHE_PKG:-/scratch/ternary-flash/goethe-release/engine}"
export GOETHE_GATE_RUN_ID="$$"
export GOETHE_GATE_PARENT="$$"
export PYTHONPATH="${GOETHE_PKG}/tests:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PY="${GOETHE_PYTHON:-${GOETHE_HOME}/py-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
export GOETHE_PYTHON="$PY"
GOETHE_TEST_GPUS="${GOETHE_TEST_GPUS:-2,3}"
export GOETHE_TEST_GPUS

GATE_TMP="$(mktemp -d "/tmp/goethe-test-${GOETHE_GATE_RUN_ID}-gate-XXXXXX")"

gate_cleanup() {
  rm -rf "$GATE_TMP" 2>/dev/null || true
}

trap gate_cleanup EXIT
trap 'gate_cleanup; exit 143' TERM
trap 'gate_cleanup; exit 130' INT

reap_stale_tmp() {
  local now cutoff mt
  now=$(date +%s)
  cutoff=$((now - 3600))
  shopt -s nullglob
  for p in /tmp/goethe-test-* /tmp/goethe-gate-artifacts-*; do
    [[ -e "$p" ]] || continue
    mt=$(stat -c %Y "$p" 2>/dev/null || echo "$now")
    if [[ "$mt" -lt "$cutoff" ]]; then
      rm -rf "$p"
    fi
  done
  shopt -u nullglob
}
reap_stale_tmp

stages=()
if [[ "$STAGE" == "all" ]]; then
  stages=(0 1 2 3 4 5 6)
else
  stages=("$STAGE")
fi

collect_tests() {
  local n="$1"
  local pat="${GOETHE_PKG}/tests/stage${n}_*.py"
  local shpat="${GOETHE_PKG}/tests/stage${n}_*.sh"
  local items=()
  shopt -s nullglob
  for f in $pat $shpat; do
    if [[ "$n" == "1" ]]; then
      case "$(basename "$f")" in
        stage1_pyenv.py|stage1_layer.py|stage1_moe.py|stage1_e2e_model.py) ;;
        *) continue ;;
      esac
    fi
    if [[ -n "$ONLY" ]]; then
      local base
      base="$(basename "$f")"
      if [[ "$base" != "$ONLY" && "$f" != *"$ONLY"* ]]; then
        continue
      fi
    fi
    items+=("$f")
  done
  shopt -u nullglob
  if [[ ${#items[@]} -eq 0 ]]; then
    return
  fi
  printf '%s\n' "${items[@]}" | sort
}

FAIL=0
PENDING=0
SKIP=0
LINT_FAIL=0
RESULT_LINES=()

run_one() {
  local testpath="$1"
  local base
  base="$(basename "$testpath")"
  local pre_json
  pre_json="$("$PY" -c "import json; from lib.goethe_test import gpu_mem_snapshot, test_gpus; print(json.dumps(gpu_mem_snapshot(test_gpus())))")"
  local out="$GATE_TMP/out-${base}.txt"
  local err="$GATE_TMP/err-${base}.txt"
  local start end rc status reason
  start=$(date +%s.%N)
  rc=0
  status=PASS
  reason=""
  set +e
  if [[ "$testpath" == *.py ]]; then
    "$PY" "$testpath" >"$out" 2>"$err"
  else
    bash "$testpath" >"$out" 2>"$err"
  fi
  rc=$?
  set -e
  end=$(date +%s.%N)
  if [[ $rc -eq 77 ]]; then
    status=PENDING
    reason="$(head -1 "$out" 2>/dev/null || true)"
    PENDING=$((PENDING + 1))
  elif [[ $rc -eq 75 ]]; then
    status=SKIP
    reason="$(head -1 "$out" 2>/dev/null || true)"
    SKIP=$((SKIP + 1))
  elif [[ $rc -ne 0 ]]; then
    status=FAIL
    reason="$(tail -5 "$err" 2>/dev/null | tr '\n' ' ')"
    FAIL=$((FAIL + 1))
  fi
  set +e
  "$PY" -c "
import json, os
from lib.goethe_test import post_test_hygiene, test_gpus, cleanup_test_procs, leaked_temp_paths
cleanup_test_procs()
pre = json.loads('''$pre_json''')
post_test_hygiene(test_gpus(), pre, gate_pid=int(os.environ.get('GOETHE_GATE_PARENT','0')), test_pids=[os.getpid()])
if leaked_temp_paths():
    raise SystemExit('leaked temp: ' + ','.join(leaked_temp_paths()))
" 2>"$GATE_TMP/hygiene-${base}.txt"
  local hrc=$?
  set -e
  if [[ $hrc -ne 0 && "$status" == "PASS" ]]; then
    status=FAIL
    reason="hygiene: $(cat "$GATE_TMP/hygiene-${base}.txt" 2>/dev/null || true)"
    FAIL=$((FAIL + 1))
  fi
  local secs
  secs=$("$PY" -c "print(round(float('$end')-float('$start'), 2))")
  RESULT_LINES+=("$(printf '%-28s %-8s %6ss %s' "$base" "$status" "$secs" "$reason")")
}

for st in "${stages[@]}"; do
  while IFS= read -r t; do
    [[ -z "$t" ]] && continue
    run_one "$t"
  done < <(collect_tests "$st")
done

printf '%s\n' "TEST                         STATUS     SEC  NOTE"
printf '%s\n' "----------------------------- -------- ------ ----"
for line in "${RESULT_LINES[@]}"; do
  printf '%s\n' "$line"
done

set +e
"$PY" "${GOETHE_PKG}/tests/lint_style.py" "${GOETHE_PKG}" >"$GATE_TMP/lint.txt" 2>&1
lint_rc=$?
set -e
if [[ $lint_rc -ne 0 ]]; then
  echo "lint_style:"
  head -20 "$GATE_TMP/lint.txt"
  LINT_FAIL=1
  FAIL=$((FAIL + 1))
fi

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
if [[ $STRICT -eq 1 && $((PENDING + SKIP)) -gt 0 ]]; then
  exit 1
fi
exit 0
