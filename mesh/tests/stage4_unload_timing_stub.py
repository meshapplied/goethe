#!/usr/bin/env python3
"""Stage 4: unload/stop must finish within spec grace (stub engines)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from lib.mesh_test import (
    Skip,
    _acquire_gpu_flocks,
    assert_no_stub_engines,
    home_mesh_env,
    main_exit,
    require_cli,
    require_gpus,
    run_mesh,
    seed_installed_model,
    spawn,
    temp_home,
    wait_meshd_socket,
    wait_until,
)

DEV_PY = Path("/scratch/ternary-flash/mesh-home/py-env/bin/python")


def _no_zombies() -> bool:
    proc = __import__("subprocess").run(
        ["ps", "-eo", "stat,pid,ppid="],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return True
    my = os.getpid()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or "Z" not in parts[0]:
            continue
        if int(parts[2]) == my:
            return False
    return True


def main() -> None:
    """Measure unload latency for SIGTERM-friendly and SIGTERM-ignoring stubs."""
    mesh = require_cli("mesh")
    require_cli("meshd")
    gpu = require_gpus(1)[0]
    release = _acquire_gpu_flocks([gpu])
    dev_py = DEV_PY if DEV_PY.is_file() else None
    try:
        with temp_home(stub_python=True, real_python=dev_py) as home:
            model_a = home / "models" / "org__a"
            model_b = home / "models" / "org__b"
            for model in (model_a, model_b):
                model.mkdir(parents=True)
                (model / "mesh_card.toml").write_text(
                    'model_id="a"\ncontext_length=4096\nmax_running_requests=4\n',
                    encoding="utf-8",
                )
            (model_b / "mesh_card.toml").write_text(
                'model_id="b"\ncontext_length=4096\nmax_running_requests=4\n',
                encoding="utf-8",
            )
            seed_installed_model(home, "a", str(model_a.relative_to(home)), 10.0, 1000, 4096, 4, 1)
            seed_installed_model(home, "b", str(model_b.relative_to(home)), 10.0, 1000, 4096, 4, 1)
            md = spawn([str(require_cli("meshd"))], env=home_mesh_env(home), label="meshd")
            wait_meshd_socket(home)
            env = {**home_mesh_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
            slow_env = {**env, "STUB_IGNORE_SIGTERM": "1"}
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=slow_env, timeout=120)
            t0 = time.time()
            run_mesh([str(mesh), "unload", "a"], env=slow_env, timeout=60)
            dt = time.time() - t0
            if dt >= 4.0:
                raise RuntimeError(f"ignore-SIGTERM unload took {dt:.2f}s (limit 4.0s)")
            if not wait_until(_no_zombies, timeout=2.0):
                raise RuntimeError("zombies after ignore-SIGTERM unload")
            fast_env = {**env, "STUB_SIGTERM_DELAY": "1"}
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=fast_env, timeout=120)
            t1 = time.time()
            run_mesh([str(mesh), "unload", "a"], env=fast_env, timeout=60)
            dt1 = time.time() - t1
            if dt1 >= 2.0:
                raise RuntimeError(f"SIGTERM-fast unload took {dt1:.2f}s (limit 2.0s)")
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=slow_env, timeout=120)
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu)], env=slow_env, timeout=120)
            t2 = time.time()
            run_mesh([str(mesh), "unload", "--all"], env=slow_env, timeout=60)
            dt2 = time.time() - t2
            if dt2 >= 4.0:
                raise RuntimeError(f"two-instance unload --all took {dt2:.2f}s (limit 4.0s)")
            t3 = time.time()
            run_mesh([str(mesh), "stop"], env=env, timeout=60)
            if time.time() - t3 >= 4.0:
                raise RuntimeError("mesh stop exceeded 4.0s on empty home")
            md.stop()
            assert_no_stub_engines(home)
        print("stage4_unload_timing_stub ok")
    finally:
        release()


if __name__ == "__main__":
    main_exit(main)
