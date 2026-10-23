#!/usr/bin/env python3
"""Gate: Prometheus /metrics on loopback mux only, never through the public gateway."""

from __future__ import annotations

import re

from lib.goethe_test import (
    ensure_nginx,
    gpu_flock,
    home_goethe_env,
    http_get,
    main_exit,
    make_tls_certs,
    require_cli,
    roomiest_test_gpu,
    run_goethe,
    seed_installed_model,
    start_goethed,
    temp_home,
    test_mux_port,
    test_public_port,
    wait_until,
    write_card,
)


def main() -> None:
    """Scrape mux /metrics, then prove an allowlisted client with a valid key cannot read it via nginx."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        ensure_nginx(home)
        make_tls_certs(home)
        write_card(home / "models" / "org__m", "m", 512, 2)
        seed_installed_model(home, "m", "models/org__m", 5.0, 512, 512, 2, 1)
        goethed = start_goethed(home)
        env = home_goethe_env(home)
        run_goethe([str(goethe), "load", "m", "--gpus", str(gpu), "--yes"], env=env, timeout=120)
        code, raw = http_get(f"http://127.0.0.1:{test_mux_port(home)}/metrics", timeout=10)
        text = raw.decode("utf-8")
        if code != 200 or "# HELP goethe_up" not in text or "# TYPE goethe_up gauge" not in text:
            raise RuntimeError(f"mux /metrics st={code} without goethe_up HELP/TYPE")
        for name in ("goethe_engine_status", "goethe_requests_total", "goethe_gpu_memory_total_bytes"):
            if f"# HELP {name}" not in text:
                raise RuntimeError(f"missing HELP for {name}")
        if not re.search(r"goethe_engine_status\{[^}]+\}\s+[01]", text):
            raise RuntimeError("goethe_engine_status sample missing")
        run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=env)
        out = run_goethe([str(goethe), "key", "create", "--name", "m", "--raw"], env=env).stdout
        token = next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))
        public = f"https://127.0.0.1:{test_public_port(home)}"
        if not wait_until(lambda: http_get(f"{public}/health")[0] == 200, timeout=20):
            raise RuntimeError("public gateway not serving /health, so the /metrics check would prove nothing")
        for headers in ({}, {"Authorization": f"Bearer {token}"}):
            code, body = http_get(f"{public}/metrics", headers=headers)
            if code == 200 or b"goethe_up" in body:
                raise RuntimeError(f"nginx exposed /metrics (st={code}, auth={bool(headers)})")
        run_goethe([str(goethe), "unload", "--all"], env=env, timeout=60)
        goethed.stop()
    print("stage3_metrics ok")


if __name__ == "__main__":
    main_exit(main)
