#!/usr/bin/env python3
"""Stage 4: goethe stop --keep-engines leaves engines serving and the next goethed adopts them (contract C-D)."""

from __future__ import annotations

import json

from lib.goethe_test import (
    gpu_flock,
    home_goethe_env,
    http_get,
    http_post,
    main_exit,
    pid_alive,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    temp_home,
    test_mux_port,
    wait_until,
    write_card,
)

ROW = "SELECT instance_id, pid, internal_port, engine_key, status FROM loaded_instances WHERE model_id='a'"


def main() -> None:
    """Load a stub, stop goethed keeping it, check it still serves alone, then a new goethed adopts and unloads it."""
    goethe = str(require_cli("goethe"))
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="keep") as home:
        env = home_goethe_env(home)
        write_card(home / "models" / "org__a", "a", 512, 2)
        seed_installed_model(home, "a", "models/org__a", 5.0, 512, 512, 2, 1)
        md = start_goethed(home)
        try:
            run_goethe([goethe, "load", "a", "--gpus", str(gpu), "--yes"], env=env, timeout=120)
            out = run_goethe([goethe, "key", "create", "--name", "keep", "--raw"], env=env).stdout
            token = next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))
            inst, pid, port, engine_key, _status = sqlite_query(home / "state.db", ROW)[0]
            rr = run([goethe, "stop", "--keep-engines"], env=env, timeout=60)
            if rr.returncode != 0:
                raise RuntimeError(f"stop --keep-engines exit {rr.returncode}: {(rr.stderr or rr.stdout)[-300:]}")
            md.popen.wait(timeout=30)
            md.stop()
            auth = {"Authorization": f"Bearer {engine_key}"}
            if not pid_alive(int(pid)) or http_get(f"http://127.0.0.1:{port}/v1/models", headers=auth)[0] != 200:
                raise RuntimeError(f"the kept engine {pid} must keep serving on {port} while goethed is down")
            md = start_goethed(home)
            adopted = [(inst, pid, port, engine_key, "RUNNING")]
            if not wait_until(lambda: sqlite_query(home / "state.db", ROW) == adopted, timeout=30):
                raise RuntimeError(f"the new goethed must adopt {adopted}, has {sqlite_query(home / 'state.db', ROW)}")
            body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
            url = f"http://127.0.0.1:{test_mux_port(home)}/v1/chat/completions"
            code, reply = http_post(url, body.encode(), headers={"Authorization": f"Bearer {token}"})
            if code != 200:
                raise RuntimeError(f"chat through the adopting goethed failed: {code} {reply[:200]!r}")
            run_goethe([goethe, "unload", "a"], env=env, timeout=60)
            if not wait_until(lambda: not pid_alive(int(pid)), timeout=10):
                raise RuntimeError(f"unload after adoption left engine {pid} running")
        finally:
            run([goethe, "unload", "--all"], env=env, timeout=60)
            md.stop()
    print("stage4_keep_engines_stub ok")


if __name__ == "__main__":
    main_exit(main)
