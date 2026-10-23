#!/usr/bin/env python3
"""goethe get api with goethed down, then goethe up, get api (text and JSON) and goethe down."""

from __future__ import annotations

import json

from lib.goethe_test import ensure_nginx, home_goethe_env, main_exit, require_cli, run, temp_home


def main() -> None:
    """The API line always prints; up starts goethed once; get api lists no models; down stops it."""
    goethe = str(require_cli("goethe"))
    with temp_home(stub_python=True, tag="upapi") as home:
        env = {**home_goethe_env(home), "NGINX_BIN": str(ensure_nginx(home))}
        rr = run([goethe, "get", "api"], env=env)
        if rr.returncode != 0 or "API: https://" not in rr.stdout or "goethed is not running" not in rr.stdout:
            raise RuntimeError(f"get api without goethed: {rr.returncode} {rr.stdout} {rr.stderr}")
        try:
            up = run([goethe, "up"], env=env, timeout=90)
            if up.returncode != 0 or "goethed is up" not in up.stdout or "/v1" not in up.stdout:
                raise RuntimeError(f"goethe up: {up.returncode} {up.stdout} {up.stderr}")
            again = run([goethe, "up"], env=env, timeout=30)
            if again.returncode != 0 or "already running" not in again.stdout:
                raise RuntimeError(f"a second goethe up must say it is already running: {again.stdout} {again.stderr}")
            api = run([goethe, "get", "api"], env=env)
            if "no models loaded" not in api.stdout:
                raise RuntimeError(f"get api with nothing loaded: {api.stdout} {api.stderr}")
            data = json.loads(run([goethe, "get", "api", "--json"], env=env).stdout)
            if not data["goethed_running"] or data["models"] != [] or not data["api"].endswith("/v1"):
                raise RuntimeError(f"get api --json: {data}")
        finally:
            down = run([goethe, "down"], env=env, timeout=60)
        if down.returncode != 0 or "goethed stopped" not in down.stdout:
            raise RuntimeError(f"goethe down: {down.returncode} {down.stdout} {down.stderr}")
    print("stage3_up_get_api ok")


if __name__ == "__main__":
    main_exit(main)
