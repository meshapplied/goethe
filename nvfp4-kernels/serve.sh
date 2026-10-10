#!/usr/bin/env bash
# Start ninfer-serve (OpenAI and Anthropic compatible HTTP) with a named profile.
#   PROFILE=default  DFlash2, 7 draft tokens, 4 lanes, 100k context   (measured, see RESULTS.md)
#   PROFILE=long     DFlash2, 7 draft tokens, 1 lane, 131072 context  (boots, not benchmarked)
#   PROFILE=batch    MTP, 4 draft tokens, 8 lanes, 131072 context     (measured)
# Override any value: SPEC=mtp|dflash2  K=<draft tokens>  MAX_CONCURRENCY=<1..8>  MAX_CONTEXT=<tokens>
# Extra arguments are passed to ninfer-serve. REQUEST_LOG=<file> writes a per-request JSONL log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/NINFER_PIN"
NVFP4_HOME="${NVFP4_HOME:-$HOME/.cache/goethe/nvfp4}"
BIN="${NINFER_SERVE:-$NVFP4_HOME/ninfer/build/apps/ninfer-serve}"
MODEL="${MODEL:-$NVFP4_HOME/models/$MODEL_FILE}"

case "${PROFILE:-default}" in
  default) p_spec=dflash2; p_k=7; p_c=4; p_ctx=100000 ;;
  long)    p_spec=dflash2; p_k=7; p_c=1; p_ctx=131072 ;;
  batch)   p_spec=mtp;     p_k=4; p_c=8; p_ctx=131072 ;;
  *) echo "error: unknown PROFILE '${PROFILE}'" >&2; exit 1 ;;
esac
[ -x "$BIN" ] || { echo "error: $BIN not found, run install.sh" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "error: $MODEL not found, run fetch_model.sh" >&2; exit 1; }

args=(--host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" --model-id qwen3.8-27b
  --max-context "${MAX_CONTEXT:-$p_ctx}" --kv-capacity auto
  --max-concurrency "${MAX_CONCURRENCY:-$p_c}" --kv-dtype fp8
  --spec "${SPEC:-$p_spec}" --draft-tokens "${K:-$p_k}" --lm-head-draft
  --prefill-chunk 1024)
[ -n "${REQUEST_LOG:-}" ] && args+=(--request-log-jsonl "$REQUEST_LOG")
exec "$BIN" "$MODEL" "${args[@]}" "$@"
