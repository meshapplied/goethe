#!/usr/bin/env python3
"""goethe configure api without a terminal: local, public with named clients, public refusing 0.0.0.0/0, VPN."""

from __future__ import annotations

import json

from lib.goethe_test import home_goethe_env, main_exit, require_cli, run, temp_home


def _status(goethe: str, env: dict[str, str]) -> dict:
    """gateway status --json as a dict."""
    return json.loads(run([goethe, "gateway", "status", "--json"], env=env).stdout)


def main() -> None:
    """Each mode sets the bind and allowlist it promises; unsafe or impossible requests change nothing."""
    goethe = str(require_cli("goethe"))
    with temp_home(stub_python=True, tag="cfgapi") as home:
        env = home_goethe_env(home)
        rr = run([goethe, "configure", "api"], env=env)
        if rr.returncode == 0 or "--mode" not in rr.stderr:
            raise RuntimeError(f"without a terminal configure api must ask for --mode: {rr.stdout} {rr.stderr}")
        rr = run([goethe, "configure", "api", "--mode", "local", "--yes"], env=env)
        if rr.returncode != 0 or "API key (default): goethe_sk_" not in rr.stdout:
            raise RuntimeError(f"local mode with --yes must bind loopback and create a key: {rr.stdout} {rr.stderr}")
        if _status(goethe, env)["bind"] != "127.0.0.1":
            raise RuntimeError("local mode must bind 127.0.0.1")
        rr = run([goethe, "configure", "api", "--mode", "public", "--yes", "--allow", "0.0.0.0/0"], env=env)
        if rr.returncode == 0 or _status(goethe, env)["bind"] != "127.0.0.1":
            raise RuntimeError(f"public mode must refuse 0.0.0.0/0 without a terminal and change nothing: {rr.stdout}")
        quiet = {**env, "GOETHE_PUBLIC_IP_LOOKUP": "0"}
        rr = run([goethe, "configure", "api", "--mode", "public", "--yes", "--allow", "203.0.113.7"], env=quiet)
        if "Your public IP: unknown, lookup disabled" not in rr.stdout:
            raise RuntimeError(f"public mode must say when the public IP lookup is disabled: {rr.stdout}")
        if rr.returncode != 0 or "not recommended" not in rr.stdout or _status(goethe, env)["bind"] != "0.0.0.0":
            raise RuntimeError(f"public mode must warn and bind 0.0.0.0: {rr.stdout} {rr.stderr}")
        allowed = run([goethe, "allowlist", "list"], env=env).stdout
        if "203.0.113.7/32" not in allowed or "127.0.0.1/32" not in allowed:
            raise RuntimeError(f"public mode must allow the named client and this machine: {allowed}")
        rr = run([goethe, "configure", "api", "--mode", "vpn", "--yes", "--vpn-ip", "192.0.2.77"], env=env)
        if rr.returncode == 0 or "not an address of this host" not in rr.stderr:
            raise RuntimeError(f"a VPN address this host does not have must be refused: {rr.stdout} {rr.stderr}")
        rr = run([goethe, "configure", "api", "--mode", "local", "--yes"], env=env)
        if rr.returncode != 0 or "API keys: 1 exist" not in rr.stdout:
            raise RuntimeError(f"a second run must keep the existing key: {rr.stdout} {rr.stderr}")
    print("stage5_configure_api ok")


if __name__ == "__main__":
    main_exit(main)
