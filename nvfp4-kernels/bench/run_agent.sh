#!/usr/bin/env bash
# Record agent trajectories: mini-swe-agent (bash tool) on SWE-bench Lite against a running ninfer-serve.
# Needs: pip install mini-swe-agent, podman (SWE-bench images are pulled from docker.io, about 3 GB each).
# usage: run_agent.sh OUTDIR [WORKERS=4] [SLICE=0:8] [PORT=8080]
set -euo pipefail
OUT="${1:?output dir}"; WORKERS="${2:-4}"; SLICE="${3:-0:8}"; PORT="${4:-8080}"
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=x
export MSWEA_COST_TRACKING=ignore_errors MSWEA_DOCKER_EXECUTABLE=podman MSWEA_SILENT_STARTUP=1
exec mini-extra swebench --subset lite --split test --shuffle --slice "$SLICE" -w "$WORKERS" -o "$OUT" \
  -m openai/qwen3.8-27b -c swebench.yaml -c agent.step_limit=80 \
  -c environment.pull_timeout=1800 -c environment.timeout=120
