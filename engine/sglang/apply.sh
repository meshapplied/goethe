#!/usr/bin/env bash
# Apply goethe-marlin.patch to an SGLang fork directory; a no-op when every hunk is already applied.

set -euo pipefail

FORK_DIR="${1:?usage: apply.sh <fork_dir>}"
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/goethe-marlin.patch"
cd "${FORK_DIR}"
if patch -p1 -R -f --dry-run --silent <"${PATCH}" >/dev/null 2>&1; then
  echo "Goethe Marlin patch already applied under ${FORK_DIR}"
  exit 0
fi
if ! patch -p1 --forward --dry-run --batch --silent <"${PATCH}"; then
  echo "ERROR: goethe-marlin.patch does not apply cleanly to ${FORK_DIR} (nothing was changed)" >&2
  exit 1
fi
patch -p1 --forward --batch --silent <"${PATCH}"
echo "Goethe Marlin patch applied under ${FORK_DIR}"
