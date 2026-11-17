#!/usr/bin/env bash
# Acceptance on a GPU machine: install, load, run and unload each model (and a local folder) with the goethe commands, in this home or a fresh one, with first-run UX limits and failure cells under --ux; one table at the end, or --json; not run in CI.

set -uo pipefail

USAGE="usage: accept.sh [--json] [--fresh] [--ux] [MODEL[:QUANT]...]   (default: the models with a kernel pack for this machine; --ux times every step, fails on a traceback, a silent gap over ACCEPT_UX_GAP seconds (10), an unannounced download over ACCEPT_UX_BIG_GB (0.1) or more than ACCEPT_UX_STDERR (10) stderr lines, and runs the failure cells (unknown id, gated id, offline, interrupted download, not logged in); --fresh runs each cell in a throwaway home with no engine environment, logged in by cloning ACCEPT_LOGIN_HOME; ACCEPT_LOCAL_DIR [ACCEPT_LOCAL_NAME] adds a local-folder install that must end on the pack runtime; GOETHE_BIN picks the goethe command; ACCEPT_CLIP and ACCEPT_REF give a speech clip and its reference text for a WER)"
PACKED=(
  ggml-org/embeddinggemma-2-GGUF:Q8_0
  mradermacher/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
  nvidia/Qwen3.8-27B-NVFP4
  orcarouter/OrcaCyber-Nano-1.0
)
PROMPT="Say hello in one short sentence."
BAD='Illegal instruction|cannot open shared object|Traceback'
REFUSED='is not supported|larger than the allowed GPUs|not supported here|gated or private|no placement fits'
PACKFAIL='does not run on|a host library is missing|crashed at open|does not fit this GPU|failed to open|runs only with its kernel pack'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOETHE="${GOETHE_BIN:-goethe}"
UXBAD='Traceback|os error|panicked at|No such file or directory'
UXGAP="${ACCEPT_UX_GAP:-10}"
UXERR="${ACCEPT_UX_STDERR:-10}"
UXLINES="${ACCEPT_UX_INSTALL_LINES:-12}"
UXBIG="${ACCEPT_UX_BIG_GB:-0.1}"
UXSMALL="${ACCEPT_UX_SMALL:-ggml-org/embeddinggemma-2-GGUF:Q8_0}"
RESERVE_GB="${ACCEPT_RESERVE_GB:-35}"
LIMIT="${ACCEPT_TIMEOUT:-3600}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
PASS=0
FAIL=0
SKIP=0
JSON=0
FRESH=0
UX=0
HOMEKIND=warm
ROWS="${WORK}/rows.tsv"
: >"${ROWS}"

while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -h | --help) echo "${USAGE}"; exit 0 ;;
    --json) JSON=1 ;;
    --fresh) FRESH=1; HOMEKIND=fresh ;;
    --ux) UX=1 ;;
    *) echo "${USAGE}" >&2; exit 2 ;;
  esac
  shift
done

g() {
  if command -v timeout >/dev/null 2>&1; then timeout "${LIMIT}" "${GOETHE}" "$@"; else "${GOETHE}" "$@"; fi
}

xg() {
  # xg STEP ARGS: goethe ARGS with stdout in ${WORK}/xg.out and stderr appended to ${WORK}/err; under --ux also timed by ux_run.py.
  local step="$1"
  shift
  if [[ ${UX} -eq 1 ]]; then
    eval "$(python3 "${HERE}/ux_run.py" --timeout "${LIMIT}" "${WORK}/xg.out" "${WORK}/xg.err" -- "${GOETHE}" "$@")"
    cat "${WORK}/xg.err" >>"${WORK}/err"
    ux_note "${step}"
    return "${U_RC}"
  fi
  g "$@" >"${WORK}/xg.out" 2>>"${WORK}/err"
}

ux_note() {
  # Add one step's time, longest silence and stderr lines to this cell's totals, and flag a traceback or an unannounced download.
  UX_WALL="$(awk -v a="${UX_WALL}" -v b="${U_WALL}" 'BEGIN { printf "%.1f", a + b }')"
  UX_GAP="$(awk -v a="${UX_GAP}" -v b="${U_GAP}" 'BEGIN { printf "%.1f", (b > a ? b : a) }')"
  UX_ERRN=$((UX_ERRN + U_ERR))
  [[ "$1" == install ]] && UX_LINES=$((U_OUT + U_ERR))
  grep -qE "${UXBAD}" "${WORK}/xg.out" "${WORK}/xg.err" && UX_TRACE=1
  if [[ "$1" == install ]] && awk -v n="${need:-0}" -v b="${UXBIG}" 'BEGIN { exit !(n > b) }' && ! cat "${WORK}/xg.out" "${WORK}/xg.err" | grep -qE '[0-9]+(\.[0-9]+)? ?(GB|MB)'; then
    UX_UNANN=1
  fi
}

ux_verdict() {
  # A passing cell must also keep the first-run limits.
  UXCELL="ttfa ${UX_WALL}s gap ${UX_GAP}s err ${UX_ERRN} lines ${UX_LINES}"
  local bad=()
  [[ ${UX_TRACE} -eq 1 ]] && bad+=("a traceback or raw OS error")
  awk -v g="${UX_GAP}" -v m="${UXGAP}" 'BEGIN { exit !(g > m) }' && bad+=("silent for ${UX_GAP}s (limit ${UXGAP}s)")
  [[ ${UX_UNANN} -eq 1 ]] && bad+=("a download over ${UXBIG} GB not announced with its size")
  [[ ${UX_LINES} -gt ${UXLINES} ]] && bad+=("install printed ${UX_LINES} lines (limit ${UXLINES})")
  [[ ${UX_ERRN} -gt ${UXERR} ]] && bad+=("${UX_ERRN} stderr lines on a success path (limit ${UXERR})")
  if [[ ${#bad[@]} -gt 0 ]]; then
    VERDICT=FAIL
    REASON="ux: $(IFS=';'; echo "${bad[*]}")"
  fi
}

gpu_name() {
  # The first GPU's name and compute capability, as nvidia-smi gives them.
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ',' | sed 's/  */ /g'
}

pack_sha() {
  # The manifest sha256 the installed pack was verified under, shortened; empty without a pack.
  sed -n 's/.*"manifest_sha256": *"\([0-9a-f]\{16\}\).*/\1/p' "$(dirname "$(card_of "$1")")/pack.verify.json" 2>/dev/null | head -n 1
}

wer_of() {
  # Word error rate of text $1 against the reference text $2 (edit distance over words).
  python3 - "$1" "$2" <<'PY'
import sys
h, r = sys.argv[1].lower().split(), sys.argv[2].lower().split()
d = list(range(len(h) + 1))
for i, rw in enumerate(r, 1):
    prev, d[0] = d[0], i
    for j, hw in enumerate(h, 1):
        prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
print(f"{d[len(h)] / max(len(r), 1):.3f}")
PY
}

clip_seconds() {
  python3 -c 'import sys, wave
w = wave.open(sys.argv[1])
print(w.getnframes() / w.getframerate())' "$1" 2>/dev/null
}

free_gb() {
  # Whole GB free where goethe keeps its models.
  df -Pk "${GOETHE_HOME:-${HOME}/.goethe}/models/." 2>/dev/null | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

model_row() {
  # One field (gb, or installed as 1 or 0) of a model's row in goethe models --json.
  g models --json 2>/dev/null | python3 -c 'import json, sys
want, field = sys.argv[1], sys.argv[2]
rows = [r for r in json.load(sys.stdin) if r["id"].lower() == want.lower()]
if field == "installed":
    print(int(any(r["status"] != "available" for r in rows)))
else:
    print(max([r.get("gb", 0) for r in rows] or [0]))' "$1" "$2" 2>/dev/null
}

packed_models() {
  # The models goethe marks as having optimised kernels on this machine; the known four when it marks none.
  g models --json 2>/dev/null | python3 -c 'import json, sys
for row in json.load(sys.stdin):
    if row.get("optimised"):
        print(row["id"])' 2>/dev/null
}

card_of() {
  echo "${GOETHE_HOME:-${HOME}/.goethe}/models/${1//\//__}/goethe_card.toml"
}

runtime_of() {
  # The runtime named by an installed model'"'"'s card.
  sed -n 's/^runtime *= *"\(.*\)"$/\1/p' "$(card_of "$1")" 2>/dev/null | head -n 1
}

clip() {
  # The given clip, else one second of silence at 16 kHz made once with the standard library.
  if [[ -n "${ACCEPT_CLIP:-}" && -f "${ACCEPT_CLIP}" ]]; then echo "${ACCEPT_CLIP}"; return; fi
  [[ -f "${WORK}/clip.wav" ]] ||
    python3 -c 'import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(16000)
w.writeframes(bytes(32000))
w.close()' "${WORK}/clip.wav" || return 1
  echo "${WORK}/clip.wav"
}

shape() {
  # Sets SHAPE (chat, embedding or transcript) and DETAIL for a run's stdout in $1; returns 1 when it is not that shape.
  local out="$1" dims norm
  if [[ "${out}" =~ ^dims=([0-9]+)\ norm=([0-9.]+) ]]; then
    dims="${BASH_REMATCH[1]}"
    norm="${BASH_REMATCH[2]}"
    SHAPE=embedding
    DETAIL="dims=${dims} norm=${norm}"
    awk -v d="${dims}" -v n="${norm}" 'BEGIN { exit !(d > 0 && n > 0) }'
    return
  fi
  if [[ "${SHAPE}" == transcript ]]; then
    DETAIL="${#out} chars"
    return 0
  fi
  SHAPE=chat
  DETAIL="${#out} chars"
  [[ "${out}" =~ [[:alnum:]] ]]
}

accept_one() {
  # Returns the verdict in VERDICT (PASS, FAIL or SKIP), REASON, and RUNTIME, PSHA, METRIC for the table; goethe's stderr for this model is kept in ${WORK}/err.
  local spec="$1" out rc audio t0 secs clip_s expect=0
  local id="${1%%:*}" install=(install "$1")
  if [[ "${spec}" == @local ]]; then
    id="${ACCEPT_LOCAL_NAME:-local/accept}"
    install=(install "${ACCEPT_LOCAL_DIR}" --name "${id}")
    expect=1
  fi
  : >"${WORK}/err"
  SHAPE=chat
  DETAIL=""
  REASON=""
  RUNTIME=""
  PSHA=""
  METRIC=""
  UX_WALL=0 UX_GAP=0 UX_ERRN=0 UX_LINES=0 UX_TRACE=0 UX_UNANN=0 UXCELL=""
  xg install "${install[@]}" --yes --load
  rc=$?
  cp "${WORK}/xg.out" "${WORK}/inst"
  if [[ ${rc} -ne 0 ]]; then
    if grep -qE "${PACKFAIL}" "${WORK}/inst" "${WORK}/err"; then
      VERDICT=FAIL
      REASON="the kernel pack does not work here ($(grep -ohE "${PACKFAIL}" "${WORK}/inst" "${WORK}/err" | head -n 1))"
    elif grep -qE "${REFUSED}" "${WORK}/err" && ! grep -qE "${BAD}" "${WORK}/err"; then
      VERDICT=SKIP
      REASON="refused on this machine (not supported, or no GPU memory free)"
    else
      VERDICT=FAIL
      REASON="install exit ${rc}"
    fi
    return
  fi
  RUNTIME="$(runtime_of "${id}")"
  PSHA="$(pack_sha "${id}")"
  t0="$(date +%s.%N)"
  xg run run "${id}" "${PROMPT}"
  rc=$?
  out="$(cat "${WORK}/xg.out")"
  if [[ ${rc} -ne 0 ]] && grep -q 'is a speech model' "${WORK}/err"; then
    SHAPE=transcript
    audio="$(clip)" || { VERDICT=FAIL; REASON="no python3 to make the audio clip"; return; }
    t0="$(date +%s.%N)"
    xg run run "${id}" "${audio}"
    rc=$?
    out="$(cat "${WORK}/xg.out")"
  fi
  secs="$(awk -v a="${t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')"
  if [[ ${rc} -ne 0 ]]; then
    VERDICT=FAIL
    REASON="run exit ${rc}"
    return
  fi
  if ! shape "${out}"; then
    VERDICT=FAIL
    REASON="run output is not a ${SHAPE} (${DETAIL})"
    return
  fi
  METRIC="${DETAIL}"
  if [[ "${SHAPE}" == transcript ]]; then
    clip_s="$(clip_seconds "${audio}")"
    METRIC="rtf $(awk -v s="${secs}" -v c="${clip_s:-1}" 'BEGIN { printf "%.3f", s / c }')"
    [[ -n "${ACCEPT_REF:-}" ]] && METRIC="${METRIC} wer $(wer_of "${out}" "${ACCEPT_REF}")"
  fi
  VERDICT=PASS
  REASON="${SHAPE} ${METRIC}"
  if [[ "${RUNTIME}" != goethe-engine ]] && { [[ ${expect} -eq 1 ]] || grep -qiE "optimised kernels|${PACKFAIL}" "${WORK}/inst" "${WORK}/err" "$(card_of "${id}")"; }; then
    VERDICT=FAIL
    REASON="has a kernel pack for this card but was served by ${RUNTIME} (${SHAPE} ${METRIC})"
  fi
  [[ ${UX} -eq 1 && "${VERDICT}" == PASS ]] && ux_verdict
}

table() {
  # The rows of this run as one aligned table.
  { printf 'MODEL\tHOME\tGPU\tRUNTIME\tPACK\tANSWER\tRESULT\tUX\n'; awk -F'\t' -v gpu="$(gpu_name)" 'BEGIN { OFS = "\t" } { print $1, $2, gpu, $3, ($4 == "" ? "-" : $4), ($5 == "" ? "-" : $5), $6, ($8 == "" ? "-" : $8) }' "${ROWS}"; } | column -t -s "$(printf '\t')"
}

json() {
  # The rows of this run as one JSON document.
  python3 - "${ROWS}" "$(gpu_name)" "${PASS}" "${FAIL}" "${SKIP}" <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
keys = ["model", "home", "runtime", "pack", "answer", "result", "reason", "ux"]
out = {"gpu": sys.argv[2], "pass": int(sys.argv[3]), "fail": int(sys.argv[4]), "skip": int(sys.argv[5]),
       "models": [dict(zip(keys, (r + [""] * 8)[:8])) for r in rows]}
print(json.dumps(out, indent=2))
PY
}

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <(packed_models)
  [[ ${#MODELS[@]} -gt 0 ]] || MODELS=("${PACKED[@]}")
fi

record() {
  # One finished cell: its row for the table and the JSON, its progress line, its tally.
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "${HOMEKIND}" "${RUNTIME}" "${PSHA}" "${METRIC}" "${VERDICT}" "${REASON}" "${UXCELL}" >>"${ROWS}"
  local line="${VERDICT} $1 ${HOMEKIND} ${REASON} ($2)"
  if [[ ${JSON} -eq 1 ]]; then echo "${line}" >&2; else echo "${line}"; fi
  case "${VERDICT}" in
    PASS) PASS=$((PASS + 1)) ;;
    SKIP) SKIP=$((SKIP + 1)) ;;
    *) FAIL=$((FAIL + 1)) ;;
  esac
}

ux_fail() {
  # A failure must end in one short message with a next step: exit not 0, no traceback or raw OS error, at most 3 lines, the expected wording and something to try.
  local name="$1" want="$2" text lines
  shift 2
  eval "$(python3 "${HERE}/ux_run.py" --timeout 300 "${WORK}/f.out" "${WORK}/f.err" -- "$@")"
  text="$(cat "${WORK}/f.out" "${WORK}/f.err")"
  lines="$(printf '%s\n' "${text}" | grep -c .)"
  RUNTIME="-" PSHA="" METRIC="${lines} lines" UXCELL="" VERDICT=PASS REASON="one message"
  if grep -qi 'already installed' <<<"${text}"; then
    VERDICT=SKIP REASON="already installed here, nothing to refuse"
  elif [[ ${U_RC} -eq 0 ]]; then
    VERDICT=FAIL REASON="succeeded, a failure was expected"
  elif grep -qE "${UXBAD}" <<<"${text}"; then
    VERDICT=FAIL REASON="a traceback or raw OS error"
  elif [[ ${lines} -gt 3 ]]; then
    VERDICT=FAIL REASON="${lines} lines, one message and a next step expected"
  elif ! grep -qiE "${want}" <<<"${text}"; then
    VERDICT=FAIL REASON="the message does not say what happened (wanted ${want})"
  elif ! grep -qiE 'goethe |--[a-z]|re-run|try again|check' <<<"${text}"; then
    VERDICT=FAIL REASON="no next step"
  fi
  record "${name}" "exit ${U_RC}, $(printf '%s' "${U_WALL}")s"
}

ux_cells() {
  # The failure modes of a first run, each in this home and against nothing it must keep.
  ux_fail ux/unknown-id 'not found' "${GOETHE}" install org/does-not-exist-xyz-123 --yes
  ux_fail ux/gated-no-token 'hf-token|gated' "${GOETHE}" install "${ACCEPT_UX_GATED:-orcarouter/OrcaCyber-Nano-1.0}" --yes
  ux_fail ux/offline 'reach|offline|network|proxy|connect' env HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 "${GOETHE}" install org/offline-probe-xyz --yes
  local had id="${UXSMALL%%:*}"
  had="$(model_row "${id}" installed)"
  eval "$(python3 "${HERE}/ux_run.py" --interrupt 3 --timeout 300 "${WORK}/i.out" "${WORK}/i.err" -- "${GOETHE}" install "${UXSMALL}" --yes --force)"
  local cut="exit ${U_RC}"
  eval "$(python3 "${HERE}/ux_run.py" --timeout "${LIMIT}" "${WORK}/f.out" "${WORK}/f.err" -- "${GOETHE}" install "${UXSMALL}" --yes)"
  RUNTIME="-" PSHA="" METRIC="rerun exit ${U_RC}" UXCELL="" VERDICT=PASS REASON="interrupted (${cut}), then installed"
  if [[ ${U_RC} -ne 0 ]]; then
    VERDICT=FAIL REASON="the rerun after an interrupt failed ($(tail -n 1 "${WORK}/f.err" | cut -c1-80))"
  elif grep -qE "${UXBAD}" "${WORK}/i.out" "${WORK}/i.err" "${WORK}/f.out" "${WORK}/f.err"; then
    VERDICT=FAIL REASON="a traceback or raw OS error around the interrupt"
  fi
  [[ "${had:-0}" == 1 ]] || g remove "${id}" </dev/null >/dev/null 2>&1
  record ux/interrupt-rerun "exit ${U_RC}"
  if [[ -n "${ACCEPT_UX_PACK_ONLY:-nvidia/Qwen3.8-27B-NVFP4}" ]] && NOLOGIN=1 fresh_open; then
    ux_fail ux/not-logged-in 'goethe login' "${GOETHE}" install "${ACCEPT_UX_PACK_ONLY:-nvidia/Qwen3.8-27B-NVFP4}" --yes
    fresh_close
  fi
}

fresh_open() {
  # A throwaway home with the login cloned from a linked one and no engine environment; goethe is started in it on its own ports.
  FRESH="$(mktemp -d "${ACCEPT_FRESH_ROOT:-${TMPDIR:-/tmp}}/goethe-accept-XXXXXX")" || return 1
  mkdir -p "${FRESH}/models" "${FRESH}/nginx"
  [[ -z "${NOLOGIN:-}" ]] && cp -a "${ACCEPT_LOGIN_HOME}/platform" "${FRESH}/platform"
  python3 -c 'import socket
for _ in range(2):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])' | { read -r mux; read -r pub; echo "${mux}" >"${FRESH}/nginx/mux_port"; echo "${pub}" >"${FRESH}/nginx/public_port"; }
  OUTER_HOME="${GOETHE_HOME-}"
  export GOETHE_HOME="${FRESH}" GOETHE_NO_AUTOSTART=1 GOETHE_SYSTEMD_USER_UNIT_DIR="${FRESH}/systemd-user" GOETHE_SYSTEMD_UNIT_DIR="${FRESH}/systemd-system"
  g up --no-boot >/dev/null 2>>"${WORK}/err"
}

fresh_close() {
  g down >/dev/null 2>&1
  [[ "${FRESH}" == *goethe-accept-* ]] && rm -rf "${FRESH}"
  if [[ -n "${OUTER_HOME}" ]]; then export GOETHE_HOME="${OUTER_HOME}"; else unset GOETHE_HOME; fi
}

cell() {
  # One install path in the current home: size guard, the run, unload, removal of what it installed.
  local spec="$1" id="${1%%:*}"
  [[ "${spec}" == @local ]] && id="${ACCEPT_LOCAL_NAME:-local/accept}"
  VERDICT=FAIL
  REASON="not run"
  RUNTIME=""
  PSHA=""
  METRIC=""
  UXCELL=""
  before="$(free_gb)"
  had="$(model_row "${id}" installed)"
  need="$(model_row "${id}" gb)"
  if [[ "${had:-0}" != 1 ]] && awk -v n="${need:-0}" -v f="${before:-0}" -v r="${RESERVE_GB}" 'BEGIN { exit !(n > f - r) }'; then
    VERDICT=SKIP
    REASON="needs ${need} GB, ${before} GB free with a ${RESERVE_GB} GB reserve"
    : >"${WORK}/err"
  else
    accept_one "${spec}"
    g unload "${id}" >/dev/null 2>>"${WORK}/err"
    if [[ "${VERDICT}" != SKIP ]] && grep -qE "${BAD}" "${WORK}/err"; then
      VERDICT=FAIL
      REASON="stderr has: $(grep -oE "${BAD}" "${WORK}/err" | head -n 1)"
    fi
    [[ "${had:-0}" == 1 ]] || g remove "${id}" </dev/null >/dev/null 2>&1
  fi
}

if [[ -n "${ACCEPT_LOCAL_DIR:-}" ]]; then
  MODELS+=("@local")
fi

for spec in "${MODELS[@]}"; do
  id="${spec%%:*}"
  [[ "${spec}" == @local ]] && id="${ACCEPT_LOCAL_NAME:-local/accept}"
  if [[ ${FRESH} -eq 0 ]]; then
    cell "${spec}"
  elif [[ ! -d "${ACCEPT_LOGIN_HOME:-/nonexistent}/platform" ]]; then
    VERDICT=SKIP
    REASON="a fresh home needs ACCEPT_LOGIN_HOME, a linked home to clone the login from"
    RUNTIME="" PSHA="" METRIC="" UXCELL="" before=""
  elif fresh_open; then
    cell "${spec}"
    fresh_close
  else
    VERDICT=FAIL
    REASON="goethe did not start in a fresh home"
    RUNTIME="" PSHA="" METRIC="" UXCELL="" before=""
    fresh_close
  fi
  record "${id}" "disk free ${before} GB -> $(free_gb) GB"
done
[[ ${UX} -eq 1 && ${FRESH} -eq 0 ]] && ux_cells
if [[ ${JSON} -eq 1 ]]; then
  json
else
  echo
  table
  echo "accept: ${PASS} pass, ${FAIL} fail, ${SKIP} skip"
fi
[[ ${FAIL} -eq 0 ]]
