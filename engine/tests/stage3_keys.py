#!/usr/bin/env python3
"""Gate: API key rpm, expiry, scope, and usage via loopback mux."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from lib.goethe_test import (
    _acquire_gpu_flocks,
    home_goethe_env,
    main_exit,
    require_cli,
    require_gpus,
    run_goethe,
    seed_installed_model,
    sha256_hex,
    spawn,
    temp_home,
    test_mux_port,
)


def _mux_post(home: Path, path: str, body: dict, token: str) -> tuple[int, dict, dict]:
    """POST JSON to the mux and return status, headers, parsed JSON body."""
    port = test_mux_port(home)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return e.code, dict(e.headers), payload


def main() -> None:
    """Exercise key limits against stub engine through mux."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    gpus = require_gpus(1)
    gpu = gpus[0]
    with temp_home(stub_python=True) as home:
        model_dir = home / "models" / "org__k"
        model_dir.mkdir(parents=True)
        (model_dir / "goethe_card.toml").write_text(
            'model_id="k"\ncontext_length=512\nmax_running_requests=2\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "k", str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1)
        goethed = spawn([str(require_cli("goethed"))], env=home_goethe_env(home))
        env = {**home_goethe_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
        release = _acquire_gpu_flocks([gpu])
        try:
            run_goethe([str(goethe), "load", "k", "--gpus", str(gpu)], env=env, timeout=120)
            raw = subprocess.check_output(
                [str(goethe), "key", "create", "--name", "rpm", "--raw", "--rpm", "2"],
                env=env,
                text=True,
            ).strip()
            tok_rpm = raw.splitlines()[-1]
            for _ in range(2):
                st, _, _ = _mux_post(
                    home,
                    "/v1/chat/completions",
                    {"model": "k", "messages": [{"role": "user", "content": "hi"}]},
                    tok_rpm,
                )
                if st != 200:
                    raise RuntimeError(f"rpm pre-limit st={st}")
            st, hdrs, body = _mux_post(
                home,
                "/v1/chat/completions",
                {"model": "k", "messages": [{"role": "user", "content": "hi"}]},
                tok_rpm,
            )
            if st != 429:
                raise RuntimeError(f"expected 429 got {st}")
            ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
            if not ra:
                raise RuntimeError("missing Retry-After")
            exp_raw = subprocess.check_output(
                [
                    str(goethe),
                    "key",
                    "create",
                    "--name",
                    "exp",
                    "--raw",
                    "--expires",
                    "2000-01-01",
                ],
                env=env,
                text=True,
            ).strip()
            tok_exp = exp_raw.splitlines()[-1]
            st, _, body = _mux_post(
                home,
                "/v1/chat/completions",
                {"model": "k", "messages": [{"role": "user", "content": "hi"}]},
                tok_exp,
            )
            if st != 401 or "expired" not in json.dumps(body):
                raise RuntimeError(f"expiry: st={st} body={body}")
            scope_raw = subprocess.check_output(
                [
                    str(goethe),
                    "key",
                    "create",
                    "--name",
                    "sc",
                    "--raw",
                    "--models",
                    "other",
                ],
                env=env,
                text=True,
            ).strip()
            tok_scope = scope_raw.splitlines()[-1]
            st, _, body = _mux_post(
                home,
                "/v1/chat/completions",
                {"model": "k", "messages": [{"role": "user", "content": "hi"}]},
                tok_scope,
            )
            if st != 404 or "Model Not Found" not in json.dumps(body):
                raise RuntimeError(f"scope: st={st} body={body}")
            subprocess.check_output(
                [str(goethe), "key", "create", "--name", "adm", "--admin"],
                env=env,
                text=True,
            )
            lst_adm = subprocess.check_output(
                [str(goethe), "key", "list"],
                env=env,
                text=True,
            )
            if "adm" not in lst_adm or "ADMIN=yes" not in lst_adm:
                raise RuntimeError(f"admin list missing: {lst_adm}")
            use_raw = subprocess.check_output(
                [str(goethe), "key", "create", "--name", "use", "--raw"],
                env=env,
                text=True,
            ).strip()
            tok_use = use_raw.splitlines()[-1]
            st, _, _ = _mux_post(
                home,
                "/v1/chat/completions",
                {"model": "k", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
                tok_use,
            )
            if st != 200:
                raise RuntimeError(f"usage call st={st}")
            key_id = tok_use[8:16]
            lst = subprocess.check_output(
                [str(goethe), "key", "list", "--json"],
                env=env,
                text=True,
            )
            rows = json.loads(lst)
            row = next(r for r in rows if r.get("id") == key_id)
            if row.get("requests_today", 0) < 1:
                raise RuntimeError(f"requests_today not updated: {row}")
            if row.get("tokens_total", 0) < 1:
                raise RuntimeError(f"tokens not updated: {row}")
            h = sha256_hex(tok_use)
            con = sqlite3.connect(home / "state.db")
            n = con.execute("SELECT COUNT(*) FROM key_usage WHERE key_id=?", (key_id,)).fetchone()[0]
            if n < 1:
                raise RuntimeError("key_usage row missing")
        finally:
            release()
            goethed.stop()
    print("stage3_keys ok")


if __name__ == "__main__":
    main_exit(main)
