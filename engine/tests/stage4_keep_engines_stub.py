#!/usr/bin/env python3
"""Stage 4: stopping goethed with keep_engines leaves engines serving; the next goethed adopts them (contract C-D)."""

from __future__ import annotations

import json
from pathlib import Path

from lib.goethe_test import (
    Proc,
    ensure_nginx,
    goethe_rpc,
    gpu_flock,
    home_goethe_env,
    http_get,
    http_post,
    main_exit,
    make_tls_certs,
    pid_alive,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    start_ticks,
    stub_requests,
    temp_home,
    test_mux_port,
    test_public_port,
    wait_until,
    write_card,
)

MODEL = "keep-model"
ROWS = "SELECT instance_id, pid, internal_port, engine_key, status FROM loaded_instances"
LATE: list[str] = []


class Keep:
    """One temp home with a stub engine behind goethed, its mux and its nginx gateway."""

    def __init__(self, goethe: str, home: Path, md: Proc) -> None:
        self.goethe, self.home, self.md = goethe, home, md
        self.env = home_goethe_env(home)
        self.model_dir = home / "models" / "org__keep"
        self.hdr: dict[str, str] = {}
        self.body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
        self.mux = f"http://127.0.0.1:{test_mux_port(home)}/v1/chat/completions"
        self.public = f"https://127.0.0.1:{test_public_port(home)}/v1/chat/completions"

    def rows(self) -> list[tuple]:
        return sqlite_query(self.home / "state.db", ROWS)

    def chat(self, url: str) -> tuple[int, bytes]:
        return http_post(url, self.body.encode(), headers=self.hdr, timeout=10)

    def restart(self, stop: str) -> None:
        """Stop goethed keeping engines (rpc or cli), check the engine and the gap, start a goethed on saved ports."""
        before, ticks = self.rows(), start_ticks(int(self.rows()[0][1]))
        _iid, pid, port, key, _status = before[0]
        mux_port = (self.home / "nginx" / "mux_port").read_text(encoding="utf-8").strip()
        if stop == "rpc":
            goethe_rpc(self.home, "shutdown", {"keep_engines": True})
        else:
            rr = run([self.goethe, "stop", "--keep-engines"], env=self.env, timeout=60)
            if rr.returncode != 0:
                LATE.append(f"goethe stop --keep-engines exit {rr.returncode}: {(rr.stderr or rr.stdout)[-200:]}")
                return
        self.md.popen.wait(timeout=20)
        if self.md.popen.returncode != 0:
            raise RuntimeError(f"goethed must exit 0 on a keep_engines stop, got {self.md.popen.returncode}")
        self.md.stop()
        auth = {"Authorization": f"Bearer {key}"}
        if start_ticks(int(pid)) != ticks or http_get(f"http://127.0.0.1:{port}/v1/models", headers=auth)[0] != 200:
            raise RuntimeError(f"engine {pid} must keep serving on {port} while goethed is down ({stop})")
        code, gap = self.chat(self.public)
        if code != 502 or not _json(gap):
            LATE.append(f"during the gap nginx must answer a JSON 502 ({stop}): {code} {gap[:120]!r}")
        self.md = start_goethed(self.home, env={"GOETHE_MUX_PORT": "", "GOETHE_PUBLIC_PORT": ""}, label="goethed-next")
        adopted = [r[:4] + ("RUNNING",) for r in before]
        if not wait_until(lambda: self.rows() == adopted, timeout=30):
            raise RuntimeError(f"the next goethed must adopt {adopted} unchanged ({stop}), has {self.rows()}")
        if (self.home / "nginx" / "mux_port").read_text(encoding="utf-8").strip() != mux_port:
            raise RuntimeError(f"the next goethed must keep the persisted mux port {mux_port} ({stop})")
        served = len(stub_requests(self.model_dir))
        code, reply = self.chat(self.mux)
        if code != 200 or len(stub_requests(self.model_dir)) <= served:
            raise RuntimeError(f"the adopted stub must serve through the mux ({stop}): {code} {reply[:120]!r}")
        if not wait_until(lambda: self.chat(self.public)[0] == 200, timeout=20):
            raise RuntimeError(f"the gateway must serve again after adoption ({stop})")


def _json(body: bytes) -> bool:
    try:
        return isinstance(json.loads(body.decode("utf-8")), dict)
    except ValueError:
        return False


def main() -> None:
    """Adoption through the shutdown RPC, then through goethe stop --keep-engines, then a clean unload."""
    goethe = str(require_cli("goethe"))
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="keep") as home:
        write_card(home / "models" / "org__keep", MODEL, 512, 2)
        seed_installed_model(home, MODEL, "models/org__keep", 1.0, 512, 512, 2, 1)
        ensure_nginx(home)
        make_tls_certs(home)
        k = Keep(goethe, home, start_goethed(home))
        pid = 0
        try:
            run_goethe([goethe, "load", MODEL, "--gpus", str(gpu), "--yes"], env=k.env, timeout=120)
            out = run_goethe([goethe, "key", "create", "--name", "k", "--raw"], env=k.env).stdout
            k.hdr = {"Authorization": f"Bearer {next(ln for ln in out.splitlines() if ln.startswith('goethe_sk_'))}"}
            run_goethe([goethe, "allowlist", "add", "127.0.0.1/32"], env=k.env)
            if k.chat(k.mux)[0] != 200 or not wait_until(lambda: k.chat(k.public)[0] == 200, timeout=20):
                raise RuntimeError("the stub must serve through the mux and the gateway before the restart")
            pid = int(k.rows()[0][1])
            k.restart("rpc")
            k.restart("cli")
            run_goethe([goethe, "unload", MODEL], env=k.env, timeout=60)
            if not wait_until(lambda: not pid_alive(pid), timeout=10):
                raise RuntimeError(f"unload after adoption left engine {pid} running")
        finally:
            run([goethe, "unload", "--all"], env=k.env, timeout=60)
            k.md.stop()
    if LATE:
        raise RuntimeError(" | ".join(LATE))
    print("stage4_keep_engines_stub ok")


if __name__ == "__main__":
    main_exit(main)
