#!/usr/bin/env python3
"""Stage 4 real-model hardware supervisor gate."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from lib.mesh_test import (
    Proc,
    Skip,
    _acquire_gpu_flocks,
    _allocate_mux_public_ports,
    gpu_mem_snapshot,
    http_post,
    main_exit,
    post_test_hygiene,
    require_cli,
    require_gpus,
    run,
    run_mesh,
    spawn,
    sqlite_query,
    test_gpus,
    wait_meshd_socket,
    wait_until,
)

DEV_HOME = Path(os.environ.get("MESH_DEV_HOME", "/scratch/ternary-flash/mesh-home"))


def _isolated_home(work: Path) -> None:
    """Symlink py-env/models from dev home and scrub volatile DB tables."""
    if work.resolve() == DEV_HOME.resolve():
        raise RuntimeError("refusing to run stage4_hw on the dev MESH_HOME")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("py-env", "models"):
        dst = work / name
        if not dst.exists():
            dst.symlink_to(DEV_HOME / name)
    if (DEV_HOME / "state.db").is_file():
        shutil.copy2(DEV_HOME / "state.db", work / "state.db")
    con = sqlite3.connect(work / "state.db")
    for t in (
        "desired_loads",
        "loaded_instances",
        "watchdog_restarts",
        "api_keys",
        "key_usage",
        "engine_overhead",
    ):
        con.execute(f"DELETE FROM {t}")
    con.commit()
    con.close()


def _mesh_env(home: Path) -> dict[str, str]:
    """Private mux/public ports for a real-engine run against isolated MESH_HOME."""
    mux, public = _allocate_mux_public_ports()
    return {
        "MESH_HOME": str(home),
        "MESH_MUX_PORT": str(mux),
        "MESH_PUBLIC_PORT": str(public),
        "MESH_SOCK": str(home / "mesh.sock"),
    }


def _model_slug(model_id: str) -> str:
    return model_id.strip().replace("/", "__")


def _installed(home: Path, model_id: str) -> bool:
    return (home / "models" / _model_slug(model_id) / "mesh_card.toml").is_file()


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _ensure_installed(mesh: Path, env: dict[str, str], model_id: str) -> None:
    home = Path(env["MESH_HOME"])
    if _installed(home, model_id):
        return
    run_mesh([str(mesh), "install", model_id.strip(), "--no-load"], env=env, timeout=600)


def _ps_json(mesh: Path, env: dict[str, str]) -> list[dict]:
    rr = run_mesh([str(mesh), "ps", "--json"], env=env, timeout=60)
    return json.loads(rr.stdout or "[]")


def _make_key(mesh: Path, home: Path, env: dict[str, str]) -> tuple[str, str]:
    """Create a throwaway API key and return (token, key_id)."""
    rr = run_mesh([str(mesh), "key", "create", "--name", "hw", "--raw"], env=env, timeout=60)
    token = next(ln for ln in rr.stdout.splitlines() if ln.startswith("mesh_sk_"))
    sql = "SELECT key_id FROM api_keys WHERE name = ? ORDER BY created_at DESC LIMIT 1"
    return token, str(sqlite_query(home / "state.db", sql, ("hw",))[0][0])


def _chat_mux(mux_port: int, model: str, token: str) -> bytes:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode()
    url = f"http://127.0.0.1:{mux_port}/v1/chat/completions"
    code, out = http_post(url, body, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    if code != 200 or len(out) < 10:
        raise RuntimeError(f"mux chat failed code={code} body={out[:200]!r}")
    return out


def main() -> None:
    """Run spec Stage 4 hardware tests with real engines on isolated MESH_HOME."""
    if os.environ.get("MESH_TEST_REAL", "0") != "1":
        raise Skip("set MESH_TEST_REAL=1 for hardware supervisor gate")
    mesh = require_cli("mesh")
    require_cli("meshd")
    models = [m.strip() for m in os.environ.get("MESH_TEST_REAL_MODELS", "meshapplied/Qwen3.8-27B-NVFP4-NInfer").split(",") if m.strip()]
    primary = models[0]
    gpus = test_gpus()
    gpu0 = gpus[0]
    work = Path(tempfile.mkdtemp(prefix=f"mesh-test-{os.environ.get('MESH_GATE_RUN_ID', os.getpid())}-hw-"))
    _isolated_home(work)
    env = _mesh_env(work)
    mux = int(env["MESH_MUX_PORT"])
    meshd: Proc | None = None
    token = key_id = ""
    release_gpu: Callable[[], None] = lambda: None
    pre_mem = gpu_mem_snapshot(gpus)
    try:
        for mid in models:
            _ensure_installed(mesh, env, mid)
        meshd = spawn([str(require_cli("meshd"))], env=env, label="meshd-real")
        wait_meshd_socket(work, timeout=30.0)
        release_gpu = _acquire_gpu_flocks(gpus)
        token, key_id = _make_key(mesh, work, env)
        t_load1 = time.time()
        run_mesh(
            [str(mesh), "load", primary, "--gpus", str(gpu0), "--ctx", "65536", "--max-conc", "8"],
            env=env,
            timeout=900,
        )
        load1_sec = time.time() - t_load1
        t_load2 = time.time()
        rr2 = run(
            [str(mesh), "load", primary, "--gpus", str(gpu0), "--ctx", "65536", "--max-conc", "8"],
            env=env,
            timeout=900,
        )
        load2_sec = time.time() - t_load2
        print(f"stage4_hw load1_sec={load1_sec:.1f} load2_sec={load2_sec:.1f}")
        if rr2.returncode != 0:
            msg = (rr2.stderr or rr2.stdout or "").strip()
            raise RuntimeError(f"co-residency finding: mesh rejected second instance: {msg[:800]}")
        rows = _ps_json(mesh, env)
        if len(rows) < 2:
            raise RuntimeError(f"expected two RUNNING instances, mesh ps={rows!r}")
        for row in rows:
            if row.get("status") != "RUNNING":
                raise RuntimeError(f"instance not RUNNING: {row!r}")
            p = int(row.get("internal_port", 0))
            if not (30001 <= p <= 30099):
                raise RuntimeError(f"engine port out of range: {row!r}")
        mid_load = gpu_mem_snapshot([gpu0])[gpu0]
        if mid_load - pre_mem[gpu0] < 28000:
            raise RuntimeError(f"gpu {gpu0} VRAM too low after co-residency: {mid_load - pre_mem[gpu0]} MiB")
        _chat_mux(mux, primary, token)
        run_mesh([str(mesh), "unload", primary], env=env, timeout=300)
        if len(gpus) < 2:
            raise Skip("need 2 GPUs in MESH_TEST_GPUS for TP=2 case")
        require_gpus(2)
        tp_model = models[-1]
        run_mesh(
            [str(mesh), "load", tp_model, "--gpus", f"{gpus[0]},{gpus[1]}", "--tp", "2", "--ctx", "131072"],
            env=env,
            timeout=900,
        )
        tp_rows = _ps_json(mesh, env)
        if not tp_rows or tp_rows[-1].get("status") != "RUNNING":
            raise RuntimeError(f"TP load not RUNNING: {tp_rows!r}")
        _chat_mux(mux, tp_model, token)
        pgids = [int(r["pgid"]) for r in _ps_json(mesh, env) if r.get("pgid")]
        t0 = time.time()
        run_mesh([str(mesh), "unload", "--all"], env=env, timeout=300)

        def reclaimed() -> bool:
            snap = gpu_mem_snapshot(gpus)
            return all(not _pgid_alive(p) for p in pgids) and all(snap[i] - pre_mem[i] <= 300 for i in gpus)

        if not wait_until(reclaimed, timeout=max(0.2, 4.0 - (time.time() - t0)), interval=0.1):
            raise RuntimeError(f"unload --all: processes or VRAM not reclaimed within 4s ({time.time() - t0:.1f}s)")
    finally:
        release_gpu()
        try:
            run_mesh([str(mesh), "unload", "--all"], env=env, timeout=120)
        except Exception:
            pass
        if meshd is not None:
            if key_id:
                run([str(mesh), "key", "revoke", key_id], env=env, timeout=30)
            meshd.stop()
        for name in ("py-env", "models"):
            link = work / name
            if link.is_symlink():
                link.unlink()
        shutil.rmtree(work, ignore_errors=True)
    post_test_hygiene(gpus, pre_mem, gate_pid=os.getpid(), test_pids=[os.getpid()])
    print("stage4_hw ok")


if __name__ == "__main__":
    main_exit(main)
