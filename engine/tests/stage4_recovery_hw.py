#!/usr/bin/env python3
"""Opt-in real-engine recovery on an isolated GOETHE_HOME (GOETHE_TEST_REAL=1)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from lib.goethe_test import (
    Skip,
    _acquire_gpu_flocks,
    dev_home,
    gpu_mem_used_mb,
    http_post,
    isolated_real_home,
    kill_engine_group,
    main_exit,
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

MODEL = os.environ.get("GOETHE_RECOVERY_MODEL", "meshapplied/Qwen3.8-27B-NVFP4-NInfer")
SCRUB = ("desired_loads", "loaded_instances", "watchdog_restarts", "api_keys", "key_usage")


def _ps_row(goethe: Path, env: dict[str, str], required: bool = True) -> dict[str, Any] | None:
    """The ps row of MODEL, or None (raising when required)."""
    rows = json.loads(run_goethe([str(goethe), "ps", "--json"], env=env, timeout=60).stdout or "[]")
    for row in rows:
        mid = str(row.get("model_id", ""))
        if mid == MODEL or MODEL.endswith(mid):
            return row
    if required:
        raise RuntimeError(f"no ps row for {MODEL}: {rows}")
    return None


def _chat(token: str, mux: int) -> int:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 8}).encode()
    url = f"http://127.0.0.1:{mux}/v1/chat/completions"
    return http_post(url, body, headers={"Authorization": f"Bearer {token}"}, timeout=60)[0]


def _spawn_lines(home: Path) -> int:
    """Engine spawns recorded in logs/index.jsonl."""
    idx = home / "logs" / "index.jsonl"
    if not idx.is_file():
        return 0
    return sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())


def _start(work: Path, env: dict[str, str]) -> Any:
    """Start goethed for the isolated home with a 30 s graceful stop."""
    md = spawn([str(require_cli("goethed"))], env=env, label="goethed-recovery")
    md.term_timeout = 30.0
    wait_goethed_socket(work, timeout=30.0)
    return md


def main() -> None:
    """Run the lead recovery scenario on one GPU when GOETHE_TEST_REAL=1."""
    if os.environ.get("GOETHE_TEST_REAL") != "1":
        raise Skip("set GOETHE_TEST_REAL=1 for hardware recovery gate")
    goethe = require_cli("goethe")
    gpu = test_gpus()[0]
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("recovery"), dir="/tmp"))
    models_before = tree_fingerprint(dev_home() / "models")
    env = {**isolated_real_home(work, SCRUB), "CUDA_VISIBLE_DEVICES": str(gpu)}
    mux = int(env["GOETHE_MUX_PORT"])
    release = _acquire_gpu_flocks([gpu])
    md = _start(work, env)
    timings: list[str] = []
    load = [str(goethe), "load", "--model", MODEL, "--gpus", str(gpu), "--yes"]
    try:
        t0 = time.time()
        run_goethe([*load, "--ctx", "65536", "--max-conc", "8"], env=env, timeout=900)
        timings.append(f"load1={time.time() - t0:.0f}s")
        row = _ps_row(goethe, env)
        if row.get("status") != "RUNNING":
            raise RuntimeError(f"not RUNNING after load: {row}")
        out = run_goethe([str(goethe), "key", "--create", "real", "--raw"], env=env).stdout
        token = next(t for t in out.split() if t.startswith("goethe_sk_"))
        t_obs = time.time()
        time.sleep(60)
        row = _ps_row(goethe, env)
        if int(row.get("restarts", 0)) != 0 or row.get("status") != "RUNNING":
            log = work / "logs" / "goethed.log"
            raise RuntimeError(f"observe failed: {row}\ngoethed: {log.read_text()[-4000:] if log.is_file() else ''}")
        if _chat(token, mux) != 200:
            raise RuntimeError("chat after observe failed")
        timings.append(f"observe60={time.time() - t_obs:.0f}s")
        sp0 = _spawn_lines(work)
        kill_engine_group(work, int(row["pgid"]))
        t1 = time.time()

        def _restarted() -> bool:
            r = _ps_row(goethe, env, required=False)
            return bool(r) and int(r.get("restarts", 0)) >= 1 and r.get("status") == "RUNNING"

        if not wait_until(_restarted, timeout=120.0):
            raise RuntimeError("watchdog restart failed")
        if _chat(token, mux) != 200:
            raise RuntimeError("chat after restart failed")
        if _spawn_lines(work) != sp0 + 1:
            raise RuntimeError(f"expected one spawn got {sp0}->{_spawn_lines(work)}")
        timings.append(f"kill9_restart={time.time() - t1:.0f}s")
        run_goethe([str(goethe), "down"], env=env)
        md.stop()
        md = _start(work, env)
        if not wait_until(lambda: (_ps_row(goethe, env, required=False) or {}).get("status") == "RUNNING", 120.0):
            raise RuntimeError("restore after stop failed")
        used_before = gpu_mem_used_mb(gpu)
        t_v = time.time()
        run_goethe([str(goethe), "unload", "--model", MODEL], env=env, timeout=120)
        if not wait_until(lambda: gpu_mem_used_mb(gpu) < used_before * 0.5 + 500, timeout=4.0):
            raise RuntimeError("VRAM not freed within 4s of unload")
        timings.append(f"vram_free={time.time() - t_v:.1f}s")
        time.sleep(2)
        if (_ps_row(goethe, env, required=False) or {}).get("status") == "RUNNING":
            raise RuntimeError("unload resurrected")
        run_goethe([*load, "--ctx", "16384", "--max-conc", "4"], env=env, timeout=900)
        if _ps_row(goethe, env).get("status") != "RUNNING":
            raise RuntimeError("learned load failed")
        oh = sqlite_query(
            work / "state.db",
            "SELECT extra_gb FROM engine_overhead WHERE model_id LIKE ?1 LIMIT 1",
            (f"%{MODEL.split('/')[-1]}%",),
        )
        if not oh or float(oh[0][0]) <= 0:
            raise RuntimeError(f"expected learned overhead: {oh}")
        timings.append("ctx16384_ok")
    finally:
        run([str(goethe), "unload", "--all"], env=env, timeout=120)
        run([str(goethe), "down"], env=env, timeout=60)
        md.stop()
        release()
        shutil.rmtree(work, ignore_errors=True)
    if tree_fingerprint(dev_home() / "models") != models_before:
        raise RuntimeError("the real-engine run modified files under the dev models directory")
    print(f"stage4_recovery_hw ok {' '.join(timings)}")


if __name__ == "__main__":
    main_exit(main)
