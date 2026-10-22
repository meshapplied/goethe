#!/usr/bin/env python3
"""Stage 4: in-load VRAM retries, cancellation, and run-registry orphan reap."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

from lib.mesh_test import (
    Proc,
    _acquire_gpu_flocks,
    assert_no_stub_engines,
    home_mesh_env,
    main_exit,
    require_cli,
    require_gpus,
    run,
    run_mesh,
    seed_installed_model,
    spawn,
    temp_home,
    wait_until,
)


def _pgrep_stub(home: Path, model_rel: str = "models/org__retry") -> list[int]:
    needle = str(home / model_rel)
    proc = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def _index_lines(home: Path) -> int:
    idx = home / "logs" / "index.jsonl"
    if not idx.is_file():
        return 0
    return sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())


def _run_registry(home: Path) -> list[Path]:
    d = home / "run"
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def main() -> None:
    """Exercise shortfall retries, CLI notices, cancel, and orphan reap."""
    mesh = require_cli("mesh")
    require_cli("meshd")
    gpu = require_gpus(1)[0]
    release = _acquire_gpu_flocks([gpu])
    dev_py = Path(os.environ.get("MESH_DEV_HOME", "/scratch/ternary-flash/mesh-home")) / "py-env" / "bin" / "python"
    md: Proc | None = None
    try:
        with temp_home(stub_python=True, real_python=dev_py if dev_py.is_file() else None) as home:
            model = home / "models" / "org__retry"
            model.mkdir(parents=True)
            (model / "mesh_card.toml").write_text(
                'model_id="retry"\ncontext_length=4096\nmax_running_requests=4\n',
                encoding="utf-8",
            )
            (model / "stub_shortfall_fail.txt").write_text("2", encoding="utf-8")
            (model / "stub_shortfall_gb.txt").write_text("4.7", encoding="utf-8")
            seed_installed_model(home, "retry", str(model.relative_to(home)), 5.0, 1000, 4096, 4, 1)
            meshd = spawn([str(require_cli("meshd"))], env=home_mesh_env(home), label="meshd")
            md = meshd
            if not wait_until(lambda: (home / "mesh.sock").is_socket(), timeout=20.0):
                raise RuntimeError("meshd sock")
            env = {**home_mesh_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
            rr = run(
                [str(mesh), "load", "retry", "--gpus", str(gpu)],
                env=env,
                timeout=120,
            )
            if rr.returncode != 0:
                raise RuntimeError(f"load failed: {rr.stderr[-400:]}")
            out = rr.stdout + rr.stderr
            if "engine needed" not in out or "retried" not in out:
                raise RuntimeError("missing retry notice in CLI output")
            if out.count("retried") != 2:
                raise RuntimeError("expected two retry notices")
            n_inst = sqlite3.connect(home / "state.db").execute(
                "SELECT COUNT(*) FROM loaded_instances WHERE model_id='retry'"
            ).fetchone()[0]
            if n_inst != 1:
                raise RuntimeError(f"expected one instance row got {n_inst}")
            idx_n = _index_lines(home)
            if idx_n != 3:
                raise RuntimeError(f"expected 3 spawns in index got {idx_n}")
            run_mesh([str(mesh), "unload", "retry"], env=env)
            time.sleep(2)
            if _pgrep_stub(home) or _run_registry(home):
                raise RuntimeError("orphan after unload")
            (model / "stub_shortfall_seen.txt").unlink(missing_ok=True)
            rr2 = run(
                [str(mesh), "load", "retry", "--gpus", str(gpu), "--no-wait"],
                env=env,
                timeout=60,
            )
            if rr2.returncode != 0:
                raise RuntimeError("no-wait load failed")
            time.sleep(1.0)
            run_mesh([str(mesh), "unload", "retry"], env=env, timeout=60)
            time.sleep(2)
            if _pgrep_stub(home):
                raise RuntimeError("stub survived unload during retry window")
            if _run_registry(home):
                raise RuntimeError("run registry left after cancel unload")
            (model / "stub_shortfall_seen.txt").unlink(missing_ok=True)
            load_proc = subprocess.Popen(
                [str(mesh), "load", "retry", "--gpus", str(gpu)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(2.5)
            orphan_pids = _pgrep_stub(home)
            if not orphan_pids:
                load_proc.kill()
                raise RuntimeError("expected stub during mid-load")
            os.kill(meshd.popen.pid, signal.SIGKILL)
            meshd.popen.wait(timeout=5)
            time.sleep(0.5)
            if not any(_pid_alive(p) for p in orphan_pids):
                load_proc.kill()
                raise RuntimeError("stub died before reap test")
            reap = spawn([str(require_cli("meshd"))], env=home_mesh_env(home), label="reap_meshd")
            md = reap
            if not wait_until(lambda: (home / "mesh.sock").is_socket(), timeout=20.0):
                raise RuntimeError("reap meshd sock")
            time.sleep(2)
            if any(_pid_alive(p) for p in orphan_pids):
                load_proc.kill()
                raise RuntimeError("orphan engine not reaped on meshd start")
            load_proc.kill()
            load_proc.wait(timeout=5)
            reap.stop()
            assert_no_stub_engines(home)
        print("stage4_load_retry_stub ok")
    finally:
        try:
            if md is not None:
                md.stop()
        except Exception:
            pass
        release()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main_exit(main)
