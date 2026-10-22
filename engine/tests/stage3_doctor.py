#!/usr/bin/env python3
"""goethe doctor speed and JSON shape."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from lib.goethe_test import main_exit, require_cli, run, temp_home, test_public_port


class _CanaryHandler(BaseHTTPRequestHandler):
    last_qs: dict[str, list[str]] = {}

    def log_message(self, _fmt: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        _CanaryHandler.last_qs = parse_qs(parsed.query)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"open")


def _seed_doctor_ok(home) -> None:
    nginx_bin = home / "nginx-root" / "usr" / "sbin" / "nginx"
    nginx_bin.parent.mkdir(parents=True, exist_ok=True)
    nginx_bin.write_text("", encoding="utf-8")
    nginx_bin.chmod(0o755)


def _seed_public_bind(home, bind: str) -> None:
    db = home / "state.db"
    con = sqlite3.connect(str(db))
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS goethe_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        con.execute(
            "INSERT OR REPLACE INTO goethe_settings (key, value) VALUES ('public_bind', ?)",
            (bind,),
        )
        con.commit()
    finally:
        con.close()
    port = test_public_port(home)
    (home / "nginx" / "public_port").write_text(f"{port}\n", encoding="utf-8")


def main() -> None:
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, tag="doctor") as home:
        base_env = {
            "GOETHE_HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "GOETHE_REGISTRY_URL": "http://127.0.0.1:9/nope.json",
        }
        t0 = time.time()
        rr = run([str(goethe), "doctor", "--json"], env=base_env, timeout=30)
        elapsed = time.time() - t0
        if elapsed > 8.0:
            raise RuntimeError(f"doctor too slow: {elapsed:.1f}s")
        rows = json.loads(rr.stdout)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("doctor json not a list")
        for row in rows:
            if not {"name", "status", "hint"} <= set(row):
                raise RuntimeError("doctor row missing fields")
        if rr.returncode not in (0, 1):
            raise RuntimeError(f"unexpected exit {rr.returncode}")
        _seed_doctor_ok(home)
        run([str(goethe), "hw", "profile"], env=base_env, timeout=120)
        _seed_public_bind(home, "0.0.0.0")
        probe_env = {
            **base_env,
            "GOETHE_PUBLIC_IP": "203.0.113.7",
        }
        rr_u = run([str(goethe), "doctor", "--probe-public"], env=probe_env, timeout=60)
        if rr_u.returncode != 0:
            raise RuntimeError(f"probe-public unknown exit {rr_u.returncode}")
        out_u = rr_u.stdout + rr_u.stderr
        if "reachability unknown" not in out_u:
            raise RuntimeError(f"missing reachability unknown: {out_u[-500:]}")
        srv = HTTPServer(("127.0.0.1", 0), _CanaryHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            canary_env = {
                **probe_env,
                "GOETHE_CANARY_URL": f"http://127.0.0.1:{port}/probe",
            }
            rr_o = run([str(goethe), "doctor", "--probe-public"], env=canary_env, timeout=30)
            if rr_o.returncode != 0:
                raise RuntimeError(f"probe-public open exit {rr_o.returncode}")
            if "public probe: OPEN" not in (rr_o.stdout + rr_o.stderr):
                raise RuntimeError(f"missing OPEN: {rr_o.stdout}")
            qs = _CanaryHandler.last_qs
            if qs.get("host") != ["203.0.113.7"]:
                raise RuntimeError(f"canary host qs {qs}")
            pub = str(test_public_port(home))
            if qs.get("port") != [pub]:
                raise RuntimeError(f"canary port qs {qs} expected {pub}")
        finally:
            srv.shutdown()
    print("stage3_doctor ok")


if __name__ == "__main__":
    main_exit(main)
