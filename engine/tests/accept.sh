#!/usr/bin/env bash
# Acceptance on a GPU machine: install, load, run and unload each model (and a local folder) with the goethe commands, in this home or a fresh one; one table at the end, or --json; not run in CI.

set -uo pipefail

USAGE="usage: accept.sh [--json] [--fresh] [MODEL[:QUANT]...]   (default: the models with a kernel pack for this machine; --fresh runs each cell in a throwaway home with no engine environment, logged in by cloning ACCEPT_LOGIN_HOME; ACCEPT_LOCAL_DIR [ACCEPT_LOCAL_NAME] adds a local-folder install that must end on the pack runtime; GOETHE_BIN picks the goethe command; ACCEPT_CLIP and ACCEPT_REF give a speech clip and its reference text for a WER)"
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
GOETHE="${GOETHE_BIN:-goethe}"
RESERVE_GB="${ACCEPT_RESERVE_GB:-35}"
LIMIT="${ACCEPT_TIMEOUT:-3600}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
PASS=0
FAIL=0
SKIP=0
JSON=0
FRESH=0
HOMEKIND=warm
ROWS="${WORK}/rows.tsv"
: >"${ROWS}"

while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -h | --help) echo "${USAGE}"; exit 0 ;;
    --json) JSON=1 ;;
    --fresh) FRESH=1; HOMEKIND=fresh ;;
    *) echo "${USAGE}" >&2; exit 2 ;;
  esac
  shift
done

g() {
  if command -v timeout >/dev/null 2>&1; then timeout "${LIMIT}" "${GOETHE}" "$@"; else "${GOETHE}" "$@"; fi
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
  g "${install[@]}" --yes --load >"${WORK}/inst" 2>>"${WORK}/err"
  rc=$?
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
  out="$(g run "${id}" "${PROMPT}" 2>>"${WORK}/err")"
  rc=$?
  if [[ ${rc} -ne 0 ]] && grep -q 'is a speech model' "${WORK}/err"; then
    SHAPE=transcript
    audio="$(clip)" || { VERDICT=FAIL; REASON="no python3 to make the audio clip"; return; }
    t0="$(date +%s.%N)"
    out="$(g run "${id}" "${audio}" 2>>"${WORK}/err")"
    rc=$?
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
}

table() {
  # The rows of this run as one aligned table.
  { printf 'MODEL\tHOME\tGPU\tRUNTIME\tPACK\tANSWER\tRESULT\n'; awk -F'\t' -v gpu="$(gpu_name)" 'BEGIN { OFS = "\t" } { print $1, $2, gpu, $3, ($4 == "" ? "-" : $4), ($5 == "" ? "-" : $5), $6 }' "${ROWS}"; } | column -t -s "$(printf '\t')"
}

json() {
  # The rows of this run as one JSON document.
  python3 - "${ROWS}" "$(gpu_name)" "${PASS}" "${FAIL}" "${SKIP}" <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
keys = ["model", "home", "runtime", "pack", "answer", "result", "reason"]
out = {"gpu": sys.argv[2], "pass": int(sys.argv[3]), "fail": int(sys.argv[4]), "skip": int(sys.argv[5]),
       "models": [dict(zip(keys, (r + [""] * 7)[:7])) for r in rows]}
print(json.dumps(out, indent=2))
PY
}

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <(packed_models)
  [[ ${#MODELS[@]} -gt 0 ]] || MODELS=("${PACKED[@]}")
fi

fresh_open() {
  # A throwaway home with the login cloned from a linked one and no engine environment; goethe is started in it on its own ports.
  FRESH="$(mktemp -d "${ACCEPT_FRESH_ROOT:-${TMPDIR:-/tmp}}/goethe-accept-XXXXXX")" || return 1
  mkdir -p "${FRESH}/models" "${FRESH}/nginx"
  cp -a "${ACCEPT_LOGIN_HOME}/platform" "${FRESH}/platform"
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
    RUNTIME="" PSHA="" METRIC="" before=""
  elif fresh_open; then
    cell "${spec}"
    fresh_close
  else
    VERDICT=FAIL
    REASON="goethe did not start in a fresh home"
    RUNTIME="" PSHA="" METRIC="" before=""
    fresh_close
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${id}" "${HOMEKIND}" "${RUNTIME}" "${PSHA}" "${METRIC}" "${VERDICT}" "${REASON}" >>"${ROWS}"
  line="${VERDICT} ${id} ${HOMEKIND} ${REASON} (disk free ${before} GB -> $(free_gb) GB)"
  if [[ ${JSON} -eq 1 ]]; then echo "${line}" >&2; else echo "${line}"; fi
  case "${VERDICT}" in
    PASS) PASS=$((PASS + 1)) ;;
    SKIP) SKIP=$((SKIP + 1)) ;;
    *) FAIL=$((FAIL + 1)) ;;
  esac
done
if [[ ${JSON} -eq 1 ]]; then
  json
else
  echo
  table
  echo "accept: ${PASS} pass, ${FAIL} fail, ${SKIP} skip"
fi
[[ ${FAIL} -eq 0 ]]
