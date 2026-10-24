#!/usr/bin/env python3
"""Stage 4 real-model hardware supervisor gate (GOETHE_TEST_REAL=1)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from lib.goethe_test import (
    Proc,
    Skip,
    _acquire_gpu_flocks,
    dev_home,
    gpu_free_gb,
    gpu_mem_snapshot,
    http_post,
    isolated_real_home,
    main_exit,
    post_test_hygiene,
    require_cli,
    run,
    run_goethe,
    run_temp_prefix,
    spawn,
    sqlite_query,
    test_gpus,
    tree_fingerprint,
    wait_goethed_socket,
    wait_until,
)

SCRUB = ("desired_loads", "loaded_instances", "watchdog_restarts", "api_keys", "key_usage", "engine_overhead")
SAFETY_MARGIN_GB = 1.5
NOT_RUN: list[str] = []


def _model_slug(model_id: str) -> str:
    return model_id.strip().replace("/", "__")


def _pgid_alive(pgid: int) -> bool:
    """True while any process is in the group."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _ensure_installed(goethe: Path, env: dict[str, str], model_id: str) -> None:
    """Install model_id into the isolated home unless the mirrored card is already there."""
    if (Path(env["GOETHE_HOME"]) / "models" / _model_slug(model_id) / "goethe_card.toml").is_file():
        return
    run_goethe([str(goethe), "install", "--model", model_id.strip(), "--no-load"], env=env, timeout=600)


def _ps_json(goethe: Path, env: dict[str, str]) -> list[dict]:
    return json.loads(run_goethe([str(goethe), "ps", "--json"], env=env, timeout=60).stdout or "[]")


def _make_key(goethe: Path, home: Path, env: dict[str, str]) -> tuple[str, str]:
    """Create a throwaway API key and return (token, key_id)."""
    rr = run_goethe([str(goethe), "key", "--create", "hw", "--raw"], env=env, timeout=60)
    token = next(ln for ln in rr.stdout.splitlines() if ln.startswith("goethe_sk_"))
    sql = "SELECT key_id FROM api_keys WHERE name = ? ORDER BY created_at DESC LIMIT 1"
    return token, str(sqlite_query(home / "state.db", sql, ("hw",))[0][0])


def _chat_mux(mux_port: int, model: str, token: str) -> bytes:
    """One chat completion through the mux; fails on anything but a real answer."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode()
    url = f"http://127.0.0.1:{mux_port}/v1/chat/completions"
    code, out = http_post(url, body, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    if code != 200 or len(out) < 10:
        raise RuntimeError(f"mux chat failed code={code} body={out[:200]!r}")
    return out


def _co_residency(goethe: Path, env: dict[str, str], load: list[str], gpu: int, pre_mb: int) -> None:
    """A --replica of the loaded model on the same GPU, or a recorded skip when that GPU has no room for it."""
    home = Path(env["GOETHE_HOME"])
    sql = "SELECT MAX(allocated_vram_gb_per_gpu) FROM loaded_instances"
    per_gpu = float(sqlite_query(home / "state.db", sql)[0][0])
    free = gpu_free_gb(gpu)
    if free < per_gpu + SAFETY_MARGIN_GB:
        NOT_RUN.append(f"co-residency --replica: GPU {gpu} has {free:.1f} GB free, a copy needs {per_gpu:.1f} GB")
        return
    t0 = time.time()
    rr = run([*load, "--replica"], env=env, timeout=900)
    print(f"stage4_hw load2_sec={time.time() - t0:.1f}")
    if rr.returncode != 0:
        raise RuntimeError(f"co-residency: goethe rejected the second instance: {(rr.stderr or rr.stdout)[:800]}")
    rows = _ps_json(goethe, env)
    if len(rows) < 2 or any(r.get("status") != "RUNNING" for r in rows):
        raise RuntimeError(f"expected two RUNNING instances, goethe ps={rows!r}")
    used = gpu_mem_snapshot([gpu])[gpu] - pre_mb
    if used < 28000:
        raise RuntimeError(f"gpu {gpu} VRAM too low after co-residency: {used} MiB")


def main() -> None:
    """Run spec Stage 4 hardware tests with real engines on an isolated GOETHE_HOME."""
    if os.environ.get("GOETHE_TEST_REAL", "0") != "1":
        raise Skip("set GOETHE_TEST_REAL=1 for hardware supervisor gate")
    goethe = require_cli("goethe")
    require_cli("goethed")
    default_model = "meshapplied/Qwen3.8-27B-NVFP4-NInfer"
    models = [m.strip() for m in os.environ.get("GOETHE_TEST_REAL_MODELS", default_model).split(",") if m.strip()]
    primary = models[0]
    gpus = test_gpus()
    gpu0 = gpus[0]
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("hw"), dir="/tmp"))
    models_before = tree_fingerprint(dev_home() / "models")
    env = isolated_real_home(work, SCRUB)
    mux = int(env["GOETHE_MUX_PORT"])
    goethed: Proc | None = None
    key_id = ""
    release_gpu: Callable[[], None] | None = None
    pre_mem = gpu_mem_snapshot(gpus)
    try:
        for mid in models:
            _ensure_installed(goethe, env, mid)
        goethed = spawn([str(require_cli("goethed"))], env=env, label="goethed-real")
        goethed.term_timeout = 30.0
        wait_goethed_socket(work, timeout=30.0)
        release_gpu = _acquire_gpu_flocks(gpus)
        token, key_id = _make_key(goethe, work, env)
        load = [str(goethe), "load", "--model", primary, "--gpus", str(gpu0)]
        load += ["--ctx", "65536", "--max-conc", "8", "--yes"]
        t_load1 = time.time()
        run_goethe(load, env=env, timeout=900)
        print(f"stage4_hw load1_sec={time.time() - t_load1:.1f}")
        _co_residency(goethe, env, load, gpu0, pre_mem[gpu0])
        rows = _ps_json(goethe, env)
        if not rows or any(r.get("status") != "RUNNING" for r in rows):
            raise RuntimeError(f"expected RUNNING instances, goethe ps={rows!r}")
        if not all(30001 <= int(r.get("internal_port", 0)) <= 30099 for r in rows):
            raise RuntimeError(f"engine port out of range: {rows!r}")
        _chat_mux(mux, primary, token)
        run_goethe([str(goethe), "unload", "--model", primary], env=env, timeout=300)
        if len(gpus) < 2:
            NOT_RUN.append(f"tp=2 load (one test GPU: GOETHE_TEST_GPUS={','.join(map(str, gpus))})")
        else:
            tp_model = models[-1]
            tp = [str(goethe), "load", "--model", tp_model, "--gpus", f"{gpus[0]},{gpus[1]}"]
            tp += ["--tp", "2", "--ctx", "131072"]
            run_goethe([*tp, "--yes"], env=env, timeout=900)
            tp_rows = _ps_json(goethe, env)
            if not tp_rows or tp_rows[-1].get("status") != "RUNNING":
                raise RuntimeError(f"TP load not RUNNING: {tp_rows!r}")
            _chat_mux(mux, tp_model, token)
        pgids = [int(r["pgid"]) for r in _ps_json(goethe, env) if r.get("pgid")]
        t0 = time.time()
        run_goethe([str(goethe), "unload", "--all"], env=env, timeout=300)

        def reclaimed() -> bool:
            snap = gpu_mem_snapshot(gpus)
            return all(not _pgid_alive(p) for p in pgids) and all(snap[i] - pre_mem[i] <= 300 for i in gpus)

        if not wait_until(reclaimed, timeout=max(0.2, 4.0 - (time.time() - t0)), interval=0.1):
            raise RuntimeError(f"unload --all: processes or VRAM not reclaimed within 4s ({time.time() - t0:.1f}s)")
    finally:
        if release_gpu is not None:
            release_gpu()
        run([str(goethe), "unload", "--all"], env=env, timeout=120)
        if goethed is not None:
            if key_id:
                run([str(goethe), "key", "--revoke", key_id], env=env, timeout=30)
            goethed.stop()
        shutil.rmtree(work, ignore_errors=True)
    if tree_fingerprint(dev_home() / "models") != models_before:
        raise RuntimeError("the real-engine run modified files under the dev models directory")
    post_test_hygiene(gpus, pre_mem, gate_pid=os.getpid(), test_pids=[os.getpid()])
    if NOT_RUN:
        raise Skip(f"every step that ran passed; not run: {NOT_RUN}")
    print("stage4_hw ok")


if __name__ == "__main__":
    main_exit(main)
