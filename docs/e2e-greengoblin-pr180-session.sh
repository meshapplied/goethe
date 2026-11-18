#!/usr/bin/env bash
# Real-user E2E on greengoblin for PR180 (isolated GOETHE_HOME). Leaves TRACE.md.
# Tailnet expose only for positive path; public/share probes must refuse or be reverted.
# Restores production goethe after. Never aborts mid-run on a single failed step.
set -u
set -o pipefail

ROOT=/home/v/proj/petya
BUILD=$ROOT/goethe-pr180/engine/target/release
SESSION=$ROOT/goethe-e2e-180
HOME_E2E=$SESSION/home
TRACE=$SESSION/TRACE.md
LOG=$SESSION/session.log
G=$BUILD/goethe
PROD_G=$(command -v goethe || true)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FAILS=0
PASS=0
FINDINGS=()

mkdir -p "$SESSION" "$HOME_E2E"
: >"$LOG"

log() { printf '%s\n' "$*" | tee -a "$LOG"; }
md() { printf '%s\n' "$*" >>"$TRACE"; }
finding() { FINDINGS+=("$1"); log "FINDING: $1"; }

# run never enables set -e for the caller (v1 aborted the session after first soft fail)
run() {
  local title=$1; shift
  md ""
  md "### $title"
  md ""
  md '```'
  md "+ $*"
  md '```'
  log ""
  log "==> $title"
  log "+ $*"
  local out ec
  set +e
  out=$(GOETHE_HOME="$HOME_E2E" PATH="$BUILD:$PATH" "$@" 2>&1)
  ec=$?
  set +e
  printf '%s\n' "$out" | tee -a "$LOG" | sed 's/^/    /'
  md ""
  md '```'
  printf '%s\n' "$out" | head -c 14000 >>"$TRACE"
  if [ "${#out}" -gt 14000 ]; then md "...(truncated)"; fi
  md '```'
  md ""
  md "exit: \`$ec\`"
  return "$ec"
}

expect_ok() {
  local title=$1; shift
  if run "$title" "$@"; then
    PASS=$((PASS + 1))
    md "**result:** pass"
    return 0
  fi
  FAILS=$((FAILS + 1))
  md "**result:** FAIL (expected success)"
  finding "FAIL ok: $title"
  return 1
}

expect_fail() {
  local title=$1; shift
  if run "$title" "$@"; then
    FAILS=$((FAILS + 1))
    md "**result:** FAIL (expected refusal — command succeeded)"
    finding "FAIL refuse: $title (command succeeded when it should refuse)"
    return 1
  fi
  PASS=$((PASS + 1))
  md "**result:** pass (refused as expected)"
  return 0
}

note() { md ""; md "> $*"; log "NOTE: $*"; }

# --- TRACE header ---
cat >"$TRACE" <<EOF
# Goethe PR180 greengoblin E2E session

- **When (UTC):** $TS
- **Host:** $(hostname)
- **Binary:** \`$G\` (\`$($G --version 2>/dev/null | head -1)\`)
- **GOETHE_HOME (isolated):** \`$HOME_E2E\`
- **Prod goethe (restored after):** \`${PROD_G:-none}\`
- **Scope:** install/scan local path, fit, load, run (chat + embed), CLI, tailnet expose — **no lasting public bind**
- **GPU:** $(nvidia-smi -L 2>/dev/null | head -1)

## Method

Isolated \`GOETHE_HOME\` so the live 0.3.x install is not mutated. Production \`goethe\` is stopped for the session (port/GPU), then started again. Findings are written as observed — not softened. Script continues after failures (no mid-run abort).

## Preflight
EOF

{
  echo "disk:"; df -h / /home/v | sed 's/^/  /'
  echo "gpu mem:"; nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv | sed 's/^/  /'
  echo "prod status:"; goethe status 2>&1 | sed 's/^/  /' | head -25
  echo "prod listeners:"; ss -ltn | grep -E '8443|8444' | sed 's/^/  /' || echo "  (none)"
} >>"$TRACE"

note "Stopping production Goethe for isolated E2E (will restore)."
goethe down 2>&1 | tee -a "$LOG" || true
sleep 2

# fresh isolated home each full run (keep TRACE history by rotating)
rm -rf "$HOME_E2E"
mkdir -p "$HOME_E2E/bin" "$HOME_E2E/models"
cp -a "$BUILD/goethe" "$BUILD/goethed" "$HOME_E2E/bin/"
if [ -f /home/v/.goethe/hardware_state.json ]; then
  cp -a /home/v/.goethe/hardware_state.json "$HOME_E2E/"
fi
# local-path model: symlink existing embedding GGUF tree (no re-download)
if [ -d /home/v/.goethe/models/ggml-org__embeddinggemma-2-GGUF ]; then
  ln -sfn /home/v/.goethe/models/ggml-org__embeddinggemma-2-GGUF \
    "$HOME_E2E/models/ggml-org__embeddinggemma-2-GGUF"
fi
if [ -d /home/v/.goethe/models/nvidia__Llama-3.1-8B-Instruct-NVFP4 ]; then
  ln -sfn /home/v/.goethe/models/nvidia__Llama-3.1-8B-Instruct-NVFP4 \
    "$HOME_E2E/models/nvidia__Llama-3.1-8B-Instruct-NVFP4"
fi

export GOETHE_HOME="$HOME_E2E"
export PATH="$HOME_E2E/bin:$BUILD:$PATH"

md ""
md "## 1. Start (fresh home)"
expect_ok "goethe up" goethe up || true
sleep 3
expect_ok "goethe status" goethe status || true
expect_ok "goethe doctor" goethe doctor || true
expect_ok "goethe doctor --hardware" goethe doctor --hardware || true

md ""
md "## 2. CLI surface / help consistency"
expect_ok "goethe --help" goethe --help || true
expect_ok "goethe api share --help" goethe api share --help || true
expect_ok "goethe api expose --help" goethe api expose --help || true
expect_ok "goethe models --help" goethe models --help || true
expect_ok "goethe run --help" goethe run --help || true
# -p must not exist on run (clap would accept unknown only with allow)
if goethe run --help 2>&1 | grep -qE -- '--prompt|-p '; then
  finding "goethe run still documents -p/--prompt (unexpected vs current help)"
else
  note "goethe run has no -p/--prompt; INPUT is positional. CI should not teach -p."
fi

md ""
md "## 3. What fits"
expect_ok "goethe models recommend" goethe models recommend || true
expect_ok "goethe models" goethe models || true
note "Compare models vs recommend per-GPU + ctx wording for the same id."

md ""
md "## 4. Local path model (scan + install from folder)"
# Correct scan arg is folder name under models/, not HF id (footgun)
run "models scan HF-id form (footgun check)" goethe models scan "ggml-org/embeddinggemma-2-GGUF"
if [ $? -ne 0 ]; then
  PASS=$((PASS + 1)); md "**result:** pass (HF-id form rejected)"
  finding "UX footgun: users typing HF id into models scan get os error 2; hint should say use folder name org__name or goethe install <path>"
else
  FAILS=$((FAILS + 1)); finding "models scan accepted org/name HF id as folder"
fi

expect_ok "models scan folder name --yes" \
  goethe models scan "ggml-org__embeddinggemma-2-GGUF" --yes || true
expect_ok "models --installed after scan" goethe models --installed || true

# alternate: install from absolute path (local path based model)
expect_ok "install from absolute path --name" \
  goethe install "$HOME_E2E/models/ggml-org__embeddinggemma-2-GGUF" \
    --name ggml-org/embeddinggemma-2-GGUF --yes || true
expect_ok "models --installed after install path" goethe models --installed || true

md ""
md "## 5. API key + tailnet expose (NO lasting public)"
expect_ok "create key e2e-tailnet" \
  goethe api keys create e2e-tailnet --models ggml-org/embeddinggemma-2-GGUF || true

RAW=$(grep -oE 'goethe_sk_[0-9a-f]+' "$LOG" | head -1 || true)
md ""
md "### captured bearer (redact in shared copies)"
md '```'
md "${RAW:-NONE}"
md '```'

expect_ok "expose vpn on tailscale0 with CGNAT allow" \
  goethe api expose vpn --interface tailscale0 --allow 100.64.0.0/10 --yes || true
expect_ok "goethe api expose (status)" goethe api expose || true
expect_ok "allowlist shows tailnet" goethe api allow || true
run "ss after vpn expose" bash -c 'ss -ltn | grep -E "8443|8444" || true'

# failure modes
# Must refuse: --yes alone must NOT bind 0.0.0.0 (fixed after greengoblin incident)
expect_fail "public expose --yes without --allow must refuse" \
  goethe api expose public --yes || true
expect_fail "raw expose 0.0.0.0 must refuse" \
  goethe api expose 0.0.0.0 || true
run "ss after public refuse probes" bash -c 'ss -ltn | grep -E "8443|8444" || true'
if ss -ltn | grep -q '0.0.0.0:8443'; then
  finding "CRITICAL: 0.0.0.0:8443 listening after refused public probes"
  FAILS=$((FAILS + 1))
else
  PASS=$((PASS + 1))
  md "**result:** pass (no 0.0.0.0:8443 after refuse probes)"
fi
# stay on vpn; re-assert tailnet bind
run "reaffirm vpn after public refuse" \
  goethe api expose vpn --interface tailscale0 --allow 100.64.0.0/10 --yes || true

expect_fail "share without --allow" goethe api share 5m --key e2e-tailnet || true
expect_fail "share with this-machine public IP as --allow" bash -c '
  IP=$(curl -4 -fsS --max-time 3 ifconfig.me || true)
  if [ -z "$IP" ]; then echo "skip: no public ip"; exit 1; fi
  goethe api share 5m --key e2e-tailnet --allow "$IP/32"
' || true
expect_ok "share status closed" goethe api share || true
run "ss :8444 must be absent" bash -c 'ss -ltn | grep 8444 || echo no_8444'
if ss -ltn | grep -q ':8444'; then
  finding "public share port 8444 is listening"
  FAILS=$((FAILS + 1))
else
  PASS=$((PASS + 1))
fi

md ""
md "## 6. Load + run embedding model"
expect_ok "load dry-run embedding" goethe load ggml-org/embeddinggemma-2-GGUF --dry-run || true
expect_ok "load embedding" goethe load ggml-org/embeddinggemma-2-GGUF || true
sleep 8
expect_ok "goethe ps" goethe ps || true
expect_ok "goethe status after load" goethe status || true

# correct run form: positional INPUT, optional --plain
run "goethe run embed positional INPUT" \
  goethe run --plain ggml-org/embeddinggemma-2-GGUF "hello greengoblin e2e"
ec_run=$?
if [ "$ec_run" -eq 0 ]; then
  PASS=$((PASS + 1)); md "**result:** pass"
else
  FAILS=$((FAILS + 1)); md "**result:** FAIL"; finding "goethe run --plain embed INPUT failed (exit $ec_run)"
fi

# wrong forms (document footguns)
run "goethe run with -- separator (footgun)" \
  goethe run ggml-org/embeddinggemma-2-GGUF -- embed "hello"
if [ $? -ne 0 ]; then
  PASS=$((PASS + 1)); md "**result:** pass (bad argv rejected)"; finding "UX: goethe run does not take -- extra args; clap error is opaque for users copying engine-flag patterns"
else
  FAILS=$((FAILS + 1)); md "**result:** unexpected success"
fi

API_TAIL="https://100.90.97.111:8443/v1"
API_LOCAL="https://127.0.0.1:8443/v1"
CACERT="$HOME_E2E/nginx/certs/server.crt"
if [ -n "$RAW" ] && [ -f "$CACERT" ]; then
  expect_ok "curl /v1/models via 127.0.0.1 + cacert" \
    curl -sS --max-time 20 --cacert "$CACERT" "$API_LOCAL/models" \
      -H "Authorization: Bearer $RAW" || true
  expect_ok "curl /v1/models via tailnet IP + cacert" \
    curl -sS --max-time 20 --cacert "$CACERT" "$API_TAIL/models" \
      -H "Authorization: Bearer $RAW" || true
  expect_ok "curl /v1/embeddings via tailnet" \
    curl -sS --max-time 90 --cacert "$CACERT" "$API_TAIL/embeddings" \
      -H "Authorization: Bearer $RAW" -H "Content-Type: application/json" \
      -d '{"model":"ggml-org/embeddinggemma-2-GGUF","input":"greengoblin e2e"}' || true
  # negative: no auth
  set +e
  http=$(curl -sS -o /tmp/e2e-noauth.body -w "%{http_code}" --max-time 20 --cacert "$CACERT" \
    "$API_TAIL/embeddings" -H "Content-Type: application/json" \
    -d '{"model":"ggml-org/embeddinggemma-2-GGUF","input":"x"}' 2>/tmp/e2e-noauth.err)
  set +e
  md ""
  md "### curl embeddings without key"
  md "HTTP \`$http\`"
  md '```'
  head -c 2000 /tmp/e2e-noauth.body >>"$TRACE" 2>/dev/null || true
  md '```'
  log "==> curl embeddings without key -> HTTP $http"
  if [ "$http" = "401" ] || [ "$http" = "403" ]; then
    PASS=$((PASS + 1)); md "**result:** pass (auth rejected)"
  else
    FAILS=$((FAILS + 1)); md "**result:** FAIL (expected 401/403, got $http)"
    finding "embeddings without bearer returned HTTP $http (expected 401/403)"
  fi
else
  finding "Could not curl API — missing bearer or cacert"
  FAILS=$((FAILS + 1))
fi

md ""
md "## 7. Chat model (if linked)"
if [ -e "$HOME_E2E/models/nvidia__Llama-3.1-8B-Instruct-NVFP4" ]; then
  expect_ok "models scan llama folder --yes" \
    goethe models scan "nvidia__Llama-3.1-8B-Instruct-NVFP4" --yes || true
  run "unload embedding before chat" goethe unload --all || true
  sleep 2
  expect_ok "load dry-run llama" goethe load nvidia/Llama-3.1-8B-Instruct-NVFP4 --dry-run || true
  expect_ok "load llama" goethe load nvidia/Llama-3.1-8B-Instruct-NVFP4 || true
  sleep 12
  run "goethe run llama one-shot" \
    goethe run --plain nvidia/Llama-3.1-8B-Instruct-NVFP4 "Say hi in five words."
  if [ $? -eq 0 ]; then PASS=$((PASS + 1)); md "**result:** pass"; else FAILS=$((FAILS + 1)); md "**result:** FAIL"; finding "llama run failed"; fi
else
  note "Llama NVFP4 tree not linked; skipped chat load."
fi

md ""
md "## 8. More failure / UX probes"
expect_fail "load missing model" goethe load definitely-missing-model-xyz || true
expect_fail "run missing model" goethe run --plain definitely-missing-model-xyz hi || true
run "load --dry-run missing" goethe load definitely-missing-model-xyz --dry-run || true
expect_fail "share dangerously without remembered key" \
  goethe api share 5m --dangerously-allow-all-inbound || true
run "gpu table" goethe gpu || true
run "logs goethed (tail)" bash -c 'goethe logs goethed 2>&1 | tail -50' || true

# consistency: status URL vs expose URL vs api URL
md ""
md "## 8b. URL consistency"
run "goethe status (URLs)" goethe status || true
run "goethe api (URLs)" goethe api || true
run "goethe api expose" goethe api expose || true

md ""
md "## 9. Cleanup on box"
run "unload all" goethe unload --all || true
run "expose local" goethe api expose local --yes || true
run "share stop" goethe api share stop || true
run "goethe down (e2e)" goethe down || true
# ensure e2e left no listeners
sleep 1
run "ss after e2e down" bash -c 'ss -ltn | grep -E "8443|8444" || echo none'

# restore production
note "Restoring production Goethe."
unset GOETHE_HOME
export PATH="/home/v/.cargo/bin:/home/v/.local/bin:$PATH"
goethe up 2>&1 | tee -a "$LOG" || true
sleep 3
goethe status 2>&1 | tee -a "$LOG" | head -30 || true
ss -ltn | grep -E '8443|8444' | tee -a "$LOG" || true
# prod must not be on 0.0.0.0
if ss -ltn | grep -q '0.0.0.0:8443'; then
  finding "PROD restored but listening on 0.0.0.0:8443"
  FAILS=$((FAILS + 1))
else
  PASS=$((PASS + 1))
  note "Prod not on 0.0.0.0:8443"
fi

md ""
md "## Scoreboard"
md ""
md "- Passes: **$PASS**"
md "- Failures / unexpected: **$FAILS**"
md "- Full command log: \`$LOG\`"
md "- Isolated home: \`$HOME_E2E\` — delete when done: \`rm -rf $SESSION\`"
md ""
md "## Findings (observed, not softened)"
md ""
if [ ${#FINDINGS[@]} -eq 0 ]; then
  md "_No automated findings recorded; review TRACE manually._"
else
  i=1
  for f in "${FINDINGS[@]}"; do
    md "$i. $f"
    i=$((i + 1))
  done
fi
md ""
md "## CI / follow-up test ideas"
md ""
md "1. Assert \`goethe models scan org/name\` fails with a hint to use \`org__name\` folder form."
md "2. Assert \`goethe api expose public --yes\` either refuses without a non-VPN client allow, or never binds \`0.0.0.0\` unless \`--dangerously-allow-all-inbound\` (product decision)."
md "3. After any public probe in E2E, assert \`ss\` has no \`0.0.0.0:8443\` / \`:8444\`."
md "4. \`goethe run --plain <model> <input>\` happy path for embed + chat; reject teaching \`run -- -p\`."
md "5. Curl \`/v1/models\` and \`/v1/embeddings\` with cacert over tailnet IP; 401 without bearer."
md "6. Share: machine public IP as \`--allow\` must refuse; no \`--allow\` must refuse."
md "7. Script must not \`set -e\` across soft failures (v1 aborted mid-session)."
md ""

echo "PASS=$PASS FAILS=$FAILS TRACE=$TRACE" | tee -a "$LOG"
# exit 0 so ssh wrappers still fetch TRACE; scoreboard holds the truth
exit 0
