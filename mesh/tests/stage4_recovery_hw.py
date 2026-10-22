#!/usr/bin/env python3
"""Opt-in real-engine recovery on isolated MESH_HOME (MESH_TEST_REAL=1)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from lib.mesh_test import (
    Skip,
    _acquire_gpu_flocks,
    main_exit,
    require_cli,
    run_mesh,
    spawn,
    sqlite_query,
    test_gpus,
    wait_until,
)

MODEL = os.environ.get("MESH_RECOVERY_MODEL", "meshapplied/Qwen3.8-27B-NVFP4-NInfer")
DEV_HOME = Path(os.environ.get("MESH_DEV_HOME", "/scratch/ternary-flash/mesh-home"))


def _isolated_home(work: Path) -> None:
    """Symlink py-env/models from dev home and scrub volatile DB tables."""
    if work.resolve() == DEV_HOME.resolve():
        raise RuntimeError("refusing to run the recovery scenario on the dev home")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("py-env", "models"):
        dst = work / name
        if not dst.exists():
            dst.symlink_to(DEV_HOME / name)
    if (DEV_HOME / "state.db").is_file():
        shutil.copy2(DEV_HOME / "state.db", work / "state.db")
    con = sqlite3.connect(work / "state.db")
    for t in ("desired_loads", "loaded_instances", "watchdog_restarts", "api_keys", "key_usage"):
        con.execute(f"DELETE FROM {t}")
    con.commit()
    con.close()


def _ps_row(env: dict[str, str], *, required: bool = True) -> dict | None:
    mesh = require_cli("mesh")
    ps = json.loads(subprocess.check_output([str(mesh), "ps", "--json"], env=env, text=True))
    for row in ps:
        mid = str(row.get("model_id", ""))
        if mid == MODEL or MODEL.endswith(mid):
            return row
    if required:
        raise RuntimeError(f"no ps row for {MODEL}: {ps}")
    return None


def _chat(env: dict[str, str], token: str, mux: int) -> int:
    body = json.dumps(
        {"model": MODEL, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 8}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{mux}/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def _spawn_lines(home: Path) -> int:
    idx = home / "logs" / "index.jsonl"
    if not idx.is_file():
        return 0
    return sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())


def main() -> None:
    """Run lead recovery scenario on one GPU when MESH_TEST_REAL=1."""
    if os.environ.get("MESH_TEST_REAL") != "1":
        raise Skip("set MESH_TEST_REAL=1 for hardware recovery gate")
    mesh = require_cli("mesh")
    meshd = require_cli("meshd")
    gpu = test_gpus()[0]
    work = Path(tempfile.mkdtemp(prefix=f"mesh-test-{os.environ.get('MESH_GATE_RUN_ID', os.getpid())}-recovery-"))
    _isolated_home(work)
    mux = int(os.environ.get("MESH_MUX_PORT", "0") or 0)
    if mux <= 0:
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        mux = s.getsockname()[1]
        s.close()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MESH_HOME": str(work),
        "MESH_MUX_PORT": str(mux),
    }
    release = _acquire_gpu_flocks([gpu])
    md = spawn([str(meshd)], env=env)
    timings: list[str] = []
    try:
        (work / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run_mesh(
            [str(mesh), "load", MODEL, "--gpus", str(gpu), "--ctx", "65536", "--max-conc", "8"],
            env=env,
            timeout=900,
        )
        timings.append(f"load1={time.time() - t0:.0f}s")
        row = _ps_row(env)
        if row.get("status") != "RUNNING":
            raise RuntimeError(f"not RUNNING after load: {row}")
        token_out = subprocess.check_output(
            [str(mesh), "key", "create", "--name", "real", "--raw"], env=env, text=True
        )
        token = next(t for t in token_out.split() if t.startswith("mesh_sk_"))
        t_obs = time.time()
        time.sleep(60)
        row = _ps_row(env)
        if int(row.get("restarts", 0)) != 0 or row.get("status") != "RUNNING":
            meshd_log = work / "logs" / "meshd.log"
            tail = meshd_log.read_text(encoding="utf-8")[-4000:] if meshd_log.is_file() else ""
            raise RuntimeError(f"observe failed: {row}\nmeshd: {tail}")
        if _chat(env, token, mux) != 200:
            raise RuntimeError("chat after observe failed")
        timings.append(f"observe60={time.time() - t_obs:.0f}s")
        sp0 = _spawn_lines(work)
        pgid = int(row["pgid"])
        os.killpg(pgid, signal.SIGKILL)
        t1 = time.time()
        def _restarted() -> bool:
            row_r = _ps_row(env, required=False)
            if not row_r:
                return False
            return int(row_r.get("restarts", 0)) >= 1 and row_r.get("status") == "RUNNING"

        if not wait_until(_restarted, timeout=120.0):
            raise RuntimeError("watchdog restart failed")
        row = _ps_row(env)
        if _chat(env, token, mux) != 200:
            raise RuntimeError("chat after restart failed")
        sp1 = _spawn_lines(work)
        if sp1 != sp0 + 1:
            raise RuntimeError(f"expected one spawn got {sp0}->{sp1}")
        timings.append(f"kill9_restart={time.time() - t1:.0f}s")
        run_mesh([str(mesh), "stop"], env=env)
        md.stop()
        for name in ("mesh.sock", "meshd.lock"):
            p = work / name
            if p.exists():
                p.unlink()
        md = spawn([str(meshd)], env=env)

        def _restored() -> bool:
            row_r = _ps_row(env, required=False)
            return row_r is not None and row_r.get("status") == "RUNNING"

        if not wait_until(_restored, timeout=120.0):
            raise RuntimeError("restore after stop failed")
        run_mesh([str(mesh), "unload", MODEL], env=env, timeout=120)
        time.sleep(2)
        if any(r.get("status") == "RUNNING" for r in json.loads(subprocess.check_output([str(mesh), "ps", "--json"], env=env, text=True))):
            raise RuntimeError("unload resurrected")
        t_v = time.time()
        used_before = float(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if not wait_until(
            lambda: float(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={gpu}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            < used_before * 0.5 + 500,
            timeout=4.0,
        ):
            raise RuntimeError("VRAM not freed after unload")
        timings.append(f"vram_free={time.time() - t_v:.1f}s")
        run_mesh(
            [str(mesh), "load", MODEL, "--gpus", str(gpu), "--ctx", "16384", "--max-conc", "4"],
            env=env,
            timeout=900,
        )
        row = _ps_row(env)
        if row.get("status") != "RUNNING":
            raise RuntimeError(f"learned load failed: {row}")
        oh = sqlite_query(
            work / "state.db",
            "SELECT extra_gb FROM engine_overhead WHERE model_id LIKE ?1 LIMIT 1",
            (f"%{MODEL.split('/')[-1]}%",),
        )
        if not oh or float(oh[0][0]) <= 0:
            raise RuntimeError(f"expected learned overhead: {oh}")
        timings.append("ctx16384_ok")
    finally:
        try:
            run_mesh([str(mesh), "unload", "--all"], env=env, timeout=120)
        except Exception:
            pass
        try:
            run_mesh([str(mesh), "stop"], env=env, timeout=60)
        except Exception:
            pass
        md.stop()
        release()
        for name in ("py-env", "models"):
            link = work / name
            if link.is_symlink():
                link.unlink()
        shutil.rmtree(work, ignore_errors=True)
    print(f"stage4_recovery_hw ok {' '.join(timings)}")


if __name__ == "__main__":
    main_exit(main)
