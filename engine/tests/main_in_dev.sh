#!/usr/bin/env bash
# Fails when the main branch holds a commit the dev branch lacks: merge main into dev first.

set -euo pipefail

MAIN="${1:-origin/main}"
DEV="${2:-origin/dev}"
missing="$(git rev-list --count "${DEV}..${MAIN}")"
if [[ "${missing}" -gt 0 ]]; then
  echo "::error::${MAIN} has ${missing} commit(s) that ${DEV} lacks: merge main into dev first" >&2
  git log --format='  %h %s' "${DEV}..${MAIN}" >&2
  exit 1
fi
echo "every commit of ${MAIN} is in ${DEV}"
