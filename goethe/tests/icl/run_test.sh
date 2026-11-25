#!/bin/bash
# Session memory test through note_proxy.py (setup in memory.md): run_test.sh <off|mem> <scenarios,csv> <turns> <total>
set -u
HERE=$(cd "$(dirname "$0")" && pwd); W=$HERE/.work; S=$W/stamina
COND=$1; EPISODES=$2; TURNS=$3; TOTAL=$4
export STAMINA_CHAT_RULES=$HERE/chat_rules.json
N=$(echo "$EPISODES" | tr ',' '\n' | wc -l)
cd "$S" && .venv/bin/python -m staminabench.run_eval --values agent=OPENCODE benchmark=RESTFUL_SERVER_GEN_ITERATIVE \
  use_docker=true batch_size="$N" restful_server_gen_iterative.n_scenarios="$TOTAL" \
  restful_server_gen_iterative.n_turns="$TURNS" restful_server_gen_iterative.execute_episodes="$EPISODES" \
  restful_server_gen_iterative.attempt_limit=3 restful_server_gen_iterative.base_seed=42 \
  docker.image_name=agent-benchmarking:opencode output_dir="$W/stamina-runs/note-$COND" \
  "docker.extra_docker_args=--init --env-file $W/stamina-$COND.env -u 0:0 -e HOME=/root -e XDG_CONFIG_HOME=/root/.config -e XDG_DATA_HOME=/root/.local/share"
