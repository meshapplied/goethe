#!/usr/bin/env python3
"""Gate: API key rpm, expiry, scope, and usage via loopback mux."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lib.goethe_test import (
    gpu_flock,
    home_goethe_env,
    main_exit,
    require_cli,
    require_gpus,
    run_goethe,
    seed_installed_model,
    sha256_hex,
    sqlite_query,
    start_goethed,
    temp_home,
    test_mux_port,
    write_card,
)

CHAT = {"model": "k", "messages": [{"role": "user", "content": "hi"}]}


def _mux_post(home: Path, body: dict[str, Any], token: str) -> tuple[int, dict[str, str], dict[str, Any]]:
    """POST a chat request to the mux and return status, headers, parsed JSON body."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{test_mux_port(home)}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return e.code, dict(e.headers), payload


def main() -> None:
    """Exercise key limits against a stub engine through the mux."""
    goethe = require_cli("goethe")
    gpu = require_gpus(1)[0]
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        write_card(home / "models" / "org__k", "k", 512, 2)
        seed_installed_model(home, "k", "models/org__k", 5.0, 512, 512, 2, 1)
        goethed = start_goethed(home)
        env = home_goethe_env(home)

        def key(*args: str) -> str:
            out = run_goethe([str(goethe), "key", "create", *args], env=env).stdout
            return next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))

        run_goethe([str(goethe), "load", "k", "--gpus", str(gpu), "--yes"], env=env, timeout=120)
        tok_rpm = key("--name", "rpm", "--raw", "--rpm", "2")
        for _ in range(2):
            st, _, _ = _mux_post(home, CHAT, tok_rpm)
            if st != 200:
                raise RuntimeError(f"rpm pre-limit st={st}")
        st, hdrs, _ = _mux_post(home, CHAT, tok_rpm)
        if st != 429:
            raise RuntimeError(f"expected 429 got {st}")
        if not (hdrs.get("Retry-After") or hdrs.get("retry-after")):
            raise RuntimeError("missing Retry-After")
        st, _, body = _mux_post(home, CHAT, key("--name", "exp", "--raw", "--expires", "2000-01-01"))
        if st != 401 or "expired" not in json.dumps(body):
            raise RuntimeError(f"expiry: st={st} body={body}")
        st, _, body = _mux_post(home, CHAT, key("--name", "sc", "--raw", "--models", "other"))
        if st != 404 or "Model Not Found" not in json.dumps(body):
            raise RuntimeError(f"scope: st={st} body={body}")
        run_goethe([str(goethe), "key", "create", "--name", "adm", "--admin"], env=env)
        lst_adm = run_goethe([str(goethe), "key", "list"], env=env).stdout
        if "adm" not in lst_adm or "ADMIN=yes" not in lst_adm:
            raise RuntimeError(f"admin list missing: {lst_adm}")
        tok_use = key("--name", "use", "--raw")
        st, _, _ = _mux_post(home, {**CHAT, "max_tokens": 2}, tok_use)
        if st != 200:
            raise RuntimeError(f"usage call st={st}")
        ids = sqlite_query(home / "state.db", "SELECT key_id FROM api_keys WHERE token_hash=?", (sha256_hex(tok_use),))
        if not ids:
            raise RuntimeError("created key has no api_keys row with its sha256")
        key_id = ids[0][0]
        rows = json.loads(run_goethe([str(goethe), "key", "list", "--json"], env=env).stdout)
        row = next(r for r in rows if r.get("id") == key_id)
        if row.get("requests_today", 0) < 1 or row.get("tokens_total", 0) < 1:
            raise RuntimeError(f"usage not updated: {row}")
        if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM key_usage WHERE key_id=?", (key_id,))[0][0] < 1:
            raise RuntimeError("key_usage row missing")
        run_goethe([str(goethe), "unload", "--all"], env=env, timeout=60)
        goethed.stop()
    print("stage3_keys ok")


if __name__ == "__main__":
    main_exit(main)
