#!/usr/bin/env bash
# Replay the recorded agent turns against several server configs and print throughput versus decode batch.
# usage: run_matrix.sh "SPEC:K:LANES:CONTEXT ..." "WORKERS ..."      (profile values from RESULTS.md)
#   run_matrix.sh "mtp:3:8:131072 mtp:4:8:131072 dflash2:7:4:100000" "1 2 4 7"
# Needs: pip install httpx. Writes logs to ${OUT:-./out}.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-./out}"; PORT="${PORT:-18081}"
mkdir -p "$OUT"
for cfg in $1; do
  IFS=: read -r spec k lanes ctx <<<"$cfg"
  tag="${spec}_k${k}"
  SPEC="$spec" K="$k" MAX_CONCURRENCY="$lanes" MAX_CONTEXT="$ctx" PORT="$PORT" REQUEST_LOG="$OUT/$tag.jsonl" \
    "$HERE/../serve.sh" >"$OUT/$tag.server.log" 2>&1 &
  server=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 "$server" 2>/dev/null || { echo "server for $tag exited, see $OUT/$tag.server.log" >&2; exit 1; }
    sleep 2
  done
  for w in $2; do
    echo "== $tag workers=$w"
    python3 "$HERE/agent_replay.py" "$HERE/traj" "$w" --port "$PORT"
  done
  kill "$server"; wait "$server" 2>/dev/null || true
  echo "== $tag decode batch curve"
  python3 "$HERE/batch_curve.py" "$OUT/$tag.jsonl" | tee "$OUT/$tag.curve.txt"
done
