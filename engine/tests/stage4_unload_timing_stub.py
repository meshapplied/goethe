#!/usr/bin/env python3
"""Stage 4: unload/stop finish within spec grace and TERM->KILL reclaims a TERM-ignoring engine (stub engines)."""

from __future__ import annotations

import time
from pathlib import Path

from lib.goethe_test import (
    assert_no_stub_engines,
    assert_no_zombies,
    gpu_flock,
    home_goethe_env,
    main_exit,
    pid_alive,
    require_cli,
    roomiest_test_gpu,
    run_goethe,
    seed_installed_model,
    start_goethed,
    stub_launch,
    stub_signals,
    temp_home,
    write_card,
)


def _load(goethe: Path, env: dict[str, str], model: Path, mid: str, gpu: int, knobs: dict[str, str]) -> int:
    """Load mid with stub knobs in its card [env], prove the engine saw them, and return the engine pid."""
    write_card(model, mid, env=knobs)
    for old in model.glob("stub_*"):
        old.unlink()
    run_goethe([str(goethe), "load", mid, "--gpus", str(gpu), "--yes"], env=env, timeout=120)
    launch = stub_launch(model)
    for key, val in knobs.items():
        if launch["knobs"].get(key) != val:
            raise RuntimeError(f"engine for {mid} did not receive {key}={val}: {launch['knobs']}")
    return int(launch["pid"])


def _timed(goethe: Path, env: dict[str, str], *args: str) -> float:
    """Run a goethe command and return its wall time."""
    t0 = time.time()
    run_goethe([str(goethe), *args], env=env, timeout=60)
    return time.time() - t0


def _term_seen(model: Path, ignored: bool) -> bool:
    return any(s["sig"] == "TERM" and s["ignored"] == ignored for s in stub_signals(model))


def main() -> None:
    """Measure unload latency for TERM-ignoring and slow-TERM stubs and prove the KILL escalation."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        model_a = home / "models" / "org__a"
        model_b = home / "models" / "org__b"
        seed_installed_model(home, "a", "models/org__a", 10.0, 1000, 4096, 4, 1)
        seed_installed_model(home, "b", "models/org__b", 10.0, 1000, 4096, 4, 1)
        md = start_goethed(home)
        env = home_goethe_env(home)
        ignore = {"STUB_IGNORE_SIGTERM": "1"}
        pid = _load(goethe, env, model_a, "a", gpu, ignore)
        dt = _timed(goethe, env, "unload", "a")
        if dt >= 4.0:
            raise RuntimeError(f"ignore-SIGTERM unload took {dt:.2f}s (limit 4.0s)")
        if not _term_seen(model_a, ignored=True):
            raise RuntimeError(f"unload never sent SIGTERM first: {stub_signals(model_a)}")
        if pid_alive(pid):
            raise RuntimeError(f"TERM-ignoring engine {pid} survived unload: no SIGKILL escalation")
        assert_no_zombies(home)
        pid = _load(goethe, env, model_a, "a", gpu, {"STUB_SIGTERM_DELAY": "1"})
        dt1 = _timed(goethe, env, "unload", "a")
        if not 0.9 <= dt1 < 2.0:
            raise RuntimeError(f"slow-TERM unload took {dt1:.2f}s (expected graceful exit in 0.9-2.0s)")
        if not _term_seen(model_a, ignored=False) or pid_alive(pid):
            raise RuntimeError("slow-TERM engine did not exit on SIGTERM")
        pids = [_load(goethe, env, model_a, "a", gpu, ignore), _load(goethe, env, model_b, "b", gpu, ignore)]
        dt2 = _timed(goethe, env, "unload", "--all")
        if dt2 >= 4.0:
            raise RuntimeError(f"two-instance unload --all took {dt2:.2f}s (limit 4.0s)")
        if any(pid_alive(p) for p in pids):
            raise RuntimeError(f"engines alive after unload --all: {[p for p in pids if pid_alive(p)]}")
        assert_no_zombies(home)
        dt3 = _timed(goethe, env, "stop")
        if dt3 >= 4.0:
            raise RuntimeError(f"goethe stop exceeded 4.0s on empty home ({dt3:.2f}s)")
        md.stop()
        assert_no_stub_engines(home)
        print(f"stage4_unload_timing_stub ok ignore={dt:.2f}s slow_term={dt1:.2f}s all={dt2:.2f}s stop={dt3:.2f}s")


if __name__ == "__main__":
    main_exit(main)
