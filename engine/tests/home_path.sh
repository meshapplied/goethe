#!/bin/bash
# GOETHE_HOME written into a systemd unit must be one absolute path.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
eval "$(awk '/^goethe_home_ok\(\)/,/^}/' "${root}/cdn/install.sh")"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

write() {
  local home="$1" dest="$2"
  if ! goethe_home_ok "${home}"; then
    return 1
  fi
  printf '%s\n' 'ExecStart=__GOETHE_HOME__/bin/goethed' | sed "s|__GOETHE_HOME__|${home}|g" >"${dest}"
}

write /var/lib/goethe "${tmp}/ok"
grep -qx 'ExecStart=/var/lib/goethe/bin/goethed' "${tmp}/ok"

bad_nl=$'/var/lib/goethe\nExecStartPost=/tmp/pwned'
if write "${bad_nl}" "${tmp}/bad"; then
  echo "raw newline was accepted" >&2
  exit 1
fi
test ! -e "${tmp}/bad"

bad_bs=$'/var/lib/goethe\\\nExecStartPost=/tmp/pwned'
if write "${bad_bs}" "${tmp}/bad2"; then
  echo "backslash newline was accepted" >&2
  exit 1
fi
test ! -e "${tmp}/bad2"
echo "home path ok"
