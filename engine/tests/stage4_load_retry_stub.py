#!/usr/bin/env python3
"""Stage 4: in-load VRAM retries, cancellation, and run-registry orphan reap."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from lib.goethe_test import (
    Proc,
    assert_no_stub_engines,
    gpu_flock,
    home_goethe_env,
    main_exit,
    pid_alive,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    stub_engine_pids,
    stub_launch,
    temp_home,
    wait_until,
    write_card,
)


def _stubs(home: Path) -> list[int]:
    return stub_engine_pids(home / "models" / "org__retry")


def _index_lines(home: Path) -> int:
    """Number of engine spawns logged in logs/index.jsonl."""
    idx = home / "logs" / "index.jsonl"
    if not idx.is_file():
        return 0
    return sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())


def _run_registry(home: Path) -> list[Path]:
    return sorted((home / "run").glob("*.json")) if (home / "run").is_dir() else []


def main() -> None:
    """Exercise shortfall retries, CLI notices, cancel, and orphan reap."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu()
    md: Proc | None = None
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        model = home / "models" / "org__retry"
        write_card(model, "retry")
        (model / "stub_shortfall_fail.txt").write_text("2", encoding="utf-8")
        (model / "stub_shortfall_gb.txt").write_text("4.7", encoding="utf-8")
        seed_installed_model(home, "retry", "models/org__retry", 5.0, 1000, 4096, 4, 1)
        md = start_goethed(home)
        env = home_goethe_env(home)
        load = [str(goethe), "load", "--model", "retry", "--gpus", str(gpu), "--yes"]
        rr = run(load, env=env, timeout=120)
        if rr.returncode != 0:
            raise RuntimeError(f"load failed: {rr.stderr[-400:]}")
        out = rr.stdout + rr.stderr
        if "engine needed" not in out or out.count("retried") != 2:
            raise RuntimeError(f"expected two retry notices in CLI output: {out[-400:]}")
        n_inst = sqlite_query(home / "state.db", "SELECT COUNT(*) FROM loaded_instances WHERE model_id='retry'")[0][0]
        if n_inst != 1:
            raise RuntimeError(f"expected one instance row got {n_inst}")
        if _index_lines(home) != 3:
            raise RuntimeError(f"expected 3 spawns in index got {_index_lines(home)}")
        run_goethe([str(goethe), "unload", "--model", "retry"], env=env)
        if not wait_until(lambda: not _stubs(home) and not _run_registry(home), timeout=5):
            raise RuntimeError("orphan engine or run registry entry after unload")
        (model / "stub_shortfall_seen.txt").unlink(missing_ok=True)
        if run([*load, "--no-wait"], env=env, timeout=60).returncode != 0:
            raise RuntimeError("no-wait load failed")
        time.sleep(1.0)
        run_goethe([str(goethe), "unload", "--model", "retry"], env=env, timeout=60)
        if not wait_until(lambda: not _stubs(home), timeout=5):
            raise RuntimeError("stub survived unload during retry window")
        if not wait_until(lambda: not _run_registry(home), timeout=5):
            raise RuntimeError("run registry left after cancel unload")
        for name in ("stub_shortfall_seen.txt", "stub_shortfall_fail.txt", "stub_launch_argv.json"):
            (model / name).unlink(missing_ok=True)
        (model / "stub_health_mode.txt").write_text("503", encoding="utf-8")
        load_proc = subprocess.Popen(load, env={**os.environ, **env}, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            launched = model / "stub_launch_argv.json"
            if not wait_until(lambda: launched.is_file() and bool(_stubs(home)), timeout=60):
                raise RuntimeError("expected an (unhealthy) stub during mid-load")
            orphan_pids = _stubs(home)
            os.kill(md.popen.pid, signal.SIGKILL)
            md.popen.wait(timeout=5)
            md.stop()
            time.sleep(0.5)
            if not any(pid_alive(p) for p in orphan_pids):
                raise RuntimeError("stub died with goethed; nothing left to reap")
            md = start_goethed(home, label="reap_goethed")
            if not wait_until(lambda: run([str(goethe), "ps"], env=env, timeout=10).returncode == 0, timeout=30):
                raise RuntimeError("goethed did not come back over the stale socket of the killed daemon")
            if not wait_until(lambda: not any(pid_alive(p) for p in orphan_pids), timeout=10):
                raise RuntimeError("orphan engine not reaped on goethed start")
        finally:
            load_proc.kill()
            load_proc.wait(timeout=5)
        run([str(goethe), "unload", "--all"], env=env, timeout=60)
        (model / "stub_health_mode.txt").unlink(missing_ok=True)
        run_goethe(load, env=env, timeout=120)
        healthy = int(stub_launch(model)["pid"])
        os.kill(md.popen.pid, signal.SIGKILL)
        md.popen.wait(timeout=5)
        md.stop()
        md = start_goethed(home, label="adopt_goethed")
        if not wait_until(lambda: run([str(goethe), "ps"], env=env, timeout=10).returncode == 0, timeout=30):
            raise RuntimeError("goethed did not come back after the second SIGKILL")
        time.sleep(3)
        rows = json.loads(run_goethe([str(goethe), "ps", "--json"], env=env).stdout or "[]")
        adopted = pid_alive(healthy) and any(
            int(r.get("pid", 0)) == healthy and r.get("status") == "RUNNING" for r in rows
        )
        run([str(goethe), "unload", "--all"], env=env, timeout=60)
        md.stop()
        assert_no_stub_engines(home)
        if not adopted:
            raise RuntimeError(f"healthy verified engine {healthy} was not adopted after goethed SIGKILL: {rows}")
        print("stage4_load_retry_stub ok")


if __name__ == "__main__":
    main_exit(main)
