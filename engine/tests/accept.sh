#!/usr/bin/env bash
# Acceptance on a GPU machine: install, load, run and unload each kernel-pack model with the goethe commands, one PASS, FAIL or SKIP line each; not run in CI.

set -uo pipefail

USAGE="usage: accept.sh [MODEL[:QUANT]...]   (default: the models with a kernel pack for this machine; GOETHE_BIN picks the goethe command)"
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

case "${1:-}" in
  -h | --help) echo "${USAGE}"; exit 0 ;;
esac

g() {
  if command -v timeout >/dev/null 2>&1; then timeout "${LIMIT}" "${GOETHE}" "$@"; else "${GOETHE}" "$@"; fi
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
    print(int(any(r["status"] == "installed" for r in rows)))
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
  # One second of silence at 16 kHz, made once with the standard library.
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
  # Returns the verdict in VERDICT (PASS, FAIL or SKIP) and REASON; goethe's stderr for this model is kept in ${WORK}/err.
  local spec="$1" id="${1%%:*}" out rc audio
  : >"${WORK}/err"
  SHAPE=chat
  DETAIL=""
  REASON=""
  g install "${spec}" --yes --load >"${WORK}/inst" 2>>"${WORK}/err"
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
  out="$(g run "${id}" "${PROMPT}" 2>>"${WORK}/err")"
  rc=$?
  if [[ ${rc} -ne 0 ]] && grep -q 'is a speech model' "${WORK}/err"; then
    SHAPE=transcript
    audio="$(clip)" || { VERDICT=FAIL; REASON="no python3 to make the audio clip"; return; }
    out="$(g run "${id}" "${audio}" 2>>"${WORK}/err")"
    rc=$?
  fi
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
  VERDICT=PASS
  REASON="${SHAPE} ${DETAIL}, runtime $(runtime_of "${id}")"
  if [[ "$(runtime_of "${id}")" != goethe-engine ]]; then
    VERDICT=FAIL
    REASON="served by $(runtime_of "${id}"), not by its kernel pack (${SHAPE} ${DETAIL})"
  fi
}

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <(packed_models)
  [[ ${#MODELS[@]} -gt 0 ]] || MODELS=("${PACKED[@]}")
fi

for spec in "${MODELS[@]}"; do
  id="${spec%%:*}"
  VERDICT=FAIL
  REASON="not run"
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
  echo "${VERDICT} ${id} ${REASON} (disk free ${before} GB -> $(free_gb) GB)"
  case "${VERDICT}" in
    PASS) PASS=$((PASS + 1)) ;;
    SKIP) SKIP=$((SKIP + 1)) ;;
    *) FAIL=$((FAIL + 1)) ;;
  esac
done
echo "accept: ${PASS} pass, ${FAIL} fail, ${SKIP} skip"
[[ ${FAIL} -eq 0 ]]
