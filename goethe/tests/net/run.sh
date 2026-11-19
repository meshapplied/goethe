#!/usr/bin/env bash
# Run vpn_interface_e2e.py in a throwaway container with its own network, using the goethe and goethed in target/release.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="${GOETHE_BIN_DIR:-$here/../../target/release}"
docker build -q -t goethe-net-test "$here" >/dev/null
docker run --rm --cap-add NET_ADMIN --device /dev/net/tun \
  -v "$bin/goethe:/usr/local/bin/goethe:ro" -v "$bin/goethed:/usr/local/bin/goethed:ro" \
  -v "$here/vpn_interface_e2e.py:/test.py:ro" \
  goethe-net-test python3 /test.py
