#!/usr/bin/env python3
"""Stage 4 meshd supervisor gate with stub engines."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from lib.mesh_test import (
    Proc,
    Skip,
    _acquire_gpu_flocks,
    assert_no_stub_engines,
    foreign_gpu_process,
    home_mesh_env,
    main_exit,
    require_cli,
    require_gpus,
    run,
    run_mesh,
    seed_installed_model,
    spawn,
    sqlite_query,
    temp_home,
    test_gpus,
    vram_per_gpu_gb,
    wait_until,
)

ASSERTS = 0


def check(cond: bool, msg: str) -> None:
    """Record one assertion."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def _hw_fixture(home: Path, gpu_index: int, free_gb: float = 90.0) -> None:
    """Write hardware_state.json with VRAM numbers for budget tests."""
    total_mb = int((free_gb / 0.85) * 1024)
    used_mb = total_mb - int(free_gb * 1024)
    payload = {
        "profiled_at": "test",
        "driver": "test",
        "gpus": [
            {
                "index": gpu_index,
                "uuid": "gpu-test",
                "name": "TestGPU",
                "compute_capability": "12.0",
                "vram_total_mb": total_mb,
                "vram_used_mb": used_mb,
                "vram_free_mb": int(free_gb * 1024),
                "pcie_gen": 4,
                "pcie_width": 16,
                "running_processes": [],
            }
        ],
        "links": [],
    }
    (home / "hardware_state.json").write_text(json.dumps(payload), encoding="utf-8")


def _gpu_used_mb(gpu_index: int) -> float:
    """Return used VRAM MiB for a GPU index."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0.0
    return float(proc.stdout.strip())


def _live_free_gb(gpu_index: int) -> float:
    """Query NVML free VRAM for mem_fraction_static checks."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi free query failed: {proc.stderr}")
    return float(proc.stdout.strip()) / 1024.0


def _read_stub_argv(model_dir: Path) -> dict:
    """Load stub launch argv json from a model directory."""
    return json.loads((model_dir / "stub_launch_argv.json").read_text(encoding="utf-8"))


def _clean_meshd_stale(home: Path) -> None:
    """Remove mesh.sock and meshd.lock after a controlled meshd stop."""
    for name in ("mesh.sock", "meshd.lock"):
        p = home / name
        if p.exists():
            p.unlink()


def _start_meshd(home: Path) -> Proc:
    """Start meshd against a temp MESH_HOME."""
    meshd = require_cli("meshd")
    proc = spawn([str(meshd)], env=home_mesh_env(home), label="meshd")
    if not wait_until(lambda: (home / "mesh.sock").is_socket(), timeout=20.0):
        raise RuntimeError("meshd did not create mesh.sock")
    return proc


def _nginx_pid(home: Path) -> int | None:
    """Return nginx master pid from MESH_HOME when present."""
    path = home / "nginx" / "nginx.pid"
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    """Return true when pid responds to signal 0."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_dead(pids: list[int | None], timeout: float) -> bool:
    """Return true when every pid in the list has exited."""
    tracked = [p for p in pids if p is not None and p > 0]

    def gone() -> bool:
        return all(not _pid_alive(p) for p in tracked)

    return wait_until(gone, timeout=timeout)


def _no_defunct_children() -> bool:
    """Return true when no defunct children belong to this test tree."""
    proc = subprocess.run(["ps", "-eo", "stat,pid,ppid="], capture_output=True, text=True)
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
    """Exercise mesh load, co-residency, foreign PID, and unload with stub engines."""
    mesh = require_cli("mesh")
    require_cli("meshd")
    gpus = require_gpus(1)
    gpu = gpus[0]
    static_w = 10.0
    kv_bytes = 1000
    ctx = 4096
    max_conc = 4
    dev_py = Path(os.environ.get("MESH_DEV_HOME", "/scratch/ternary-flash/mesh-home")) / "py-env" / "bin" / "python"
    with temp_home(stub_python=True, real_python=dev_py if dev_py.is_file() else None) as home:
        _hw_fixture(home, gpu, free_gb=90.0)
        model_a = home / "models" / "org__model_a"
        model_b = home / "models" / "org__model_b"
        model_a.mkdir(parents=True)
        model_b.mkdir(parents=True)
        (model_a / "mesh_card.toml").write_text(
            f'model_id="a"\ncontext_length={ctx}\nmax_running_requests={max_conc}\n',
            encoding="utf-8",
        )
        (model_b / "mesh_card.toml").write_text(
            'model_id="b"\ncontext_length=4096\nmax_running_requests=4\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "a", str(model_a.relative_to(home)), static_w, kv_bytes, ctx, max_conc, 1)
        seed_installed_model(home, "b", str(model_b.relative_to(home)), static_w, kv_bytes, ctx, max_conc, 1)
        ternary_root = home / "goethe" / "ternary-kernels"
        ternary_root.mkdir(parents=True, exist_ok=True)
        model_t = home / "models" / "org__ternary_t"
        model_t.mkdir(parents=True)
        (model_t / "mesh_card.toml").write_text(
            f'model_id="t"\nruntime="ternary"\npack_id="p"\nbase_model_id="b"\n'
            f'context_length=512\nmax_running_requests=2\n'
            f'pythonpath=["{home}/goethe/ternary-kernels"]\n'
            f'TERNARY_PACK_DIR="{home}/models/pack"\n'
            f'TERNARY_MODEL_DIR="{home}/models/base"\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "t", str(model_t.relative_to(home)), 5.0, 512, 512, 2, 1)
        meshd = _start_meshd(home)
        env = {**home_mesh_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}

        def _ps_json() -> list[dict]:
            rr = run_mesh([str(mesh), "ps", "--json"], env=env)
            raw = rr.stdout.strip() or "[]"
            return json.loads(raw)

        def _wait_model_status(mid: str, want: set[str], timeout: float) -> bool:
            def ok() -> bool:
                for row in _ps_json():
                    if row.get("model_id") == mid:
                        return row.get("status") in want
                return False

            return wait_until(ok, timeout=timeout)

        def _kill_engine(mid: str) -> None:
            row = sqlite_query(
                home / "state.db",
                "SELECT pid, pgid FROM loaded_instances WHERE model_id=?1 ORDER BY rowid DESC LIMIT 1",
                (mid,),
            )
            if not row:
                return
            pid, pgid = int(row[0][0]), int(row[0][1])
            target = pgid if pgid > 0 else pid
            if target > 0:
                os.killpg(target, signal.SIGKILL)

        def _restarts_at_least(mid: str, n: int) -> bool:
            for row in _ps_json():
                if row.get("model_id") == mid and int(row.get("restarts", 0)) >= n:
                    return True
            db_n = sqlite_query(
                home / "state.db",
                "SELECT COALESCE(restart_count,0) FROM loaded_instances WHERE model_id=?1 ORDER BY rowid DESC LIMIT 1",
                (mid,),
            )
            return bool(db_n) and int(db_n[0][0]) >= n

        release_gpu: Callable[[], None] = lambda: None
        release_gpu = _acquire_gpu_flocks([gpu])
        try:
            rr = run_mesh(
                [str(mesh), "load", "a", "--gpus", str(gpu), "--ctx", str(ctx), "--max-conc", str(max_conc)],
                env=env,
                timeout=300,
            )
            check(rr.returncode == 0, "load a rc")
            rows = sqlite_query(
                home / "state.db",
                "SELECT pid, pgid, internal_port, gpu_indices, status FROM loaded_instances WHERE model_id='a'",
            )
            check(len(rows) == 1, "one instance a")
            pid, pgid, port, gpu_indices, status = rows[0]
            check(int(pid) > 0 and int(pgid) == int(pid), "pgid==pid")
            check(30001 <= int(port) <= 30099, "engine port in range")
            port_a = int(port)
            check(str(gpu) in str(gpu_indices), "gpu_indices")
            check(status == "RUNNING", "status RUNNING")
            desired = sqlite_query(home / "state.db", "SELECT COUNT(*) FROM desired_loads")[0][0]
            check(int(desired) >= 1, "desired load row")
            stub = _read_stub_argv(model_a)
            parsed = stub["parsed"]
            b_gb = vram_per_gpu_gb(static_w, ctx, kv_bytes, max_conc, 1)
            free_gb = _live_free_gb(gpu)
            check(abs(float(parsed["mem_fraction_static"]) - b_gb / free_gb) < 0.02, "mem_fraction_static")
            check(parsed["host"] == "127.0.0.1", "host")
            check(int(parsed["port"]) == int(port), "stub port")
            check(int(parsed.get("tp_size", 1)) == 1, "tp-size")
            check(int(parsed.get("context_length", 0)) == ctx, "context-length")
            check(int(parsed.get("max_running_requests", 0)) == max_conc, "max-running-requests")
            check(parsed.get("kv_cache_dtype"), "kv-cache-dtype")
            check(stub["env"].get("CUDA_VISIBLE_DEVICES") == str(gpu), "cuda visible")
            argv_s = " ".join(str(x) for x in stub.get("argv", []))
            check("MESH_KERNEL_DIR" not in stub.get("env", {}), "no MESH_KERNEL_DIR")
            check("--enable-mesh-fused-gemv" not in argv_s, "no mesh-fused-gemv flag")
            rr2 = run_mesh([str(mesh), "load", "b", "--gpus", str(gpu)], env=env, timeout=300)
            combined = rr2.stdout + rr2.stderr
            check("currently hosts [" in combined, "co-residency notice")
            check("Free VRAM:" in combined, "free vram notice")

            def _two_ports() -> bool:
                rows2 = sqlite_query(
                    home / "state.db", "SELECT internal_port FROM loaded_instances ORDER BY internal_port"
                )
                return len(rows2) >= 2

            check(wait_until(_two_ports, timeout=60.0), "two engine rows")
            rows2 = sqlite_query(home / "state.db", "SELECT internal_port FROM loaded_instances ORDER BY internal_port")
            ports = [int(r[0]) for r in rows2]
            check(len(set(ports)) == len(ports), "unique ports")
            check(all(30001 <= p <= 30099 for p in ports), "engine ports in range")
            check(ports[1] != port_a, "second engine port distinct")
            seed_installed_model(home, "a", str(model_a.relative_to(home)), 200.0, kv_bytes, 131072, 64, 1)
            rr_budget = run(
                [str(mesh), "load", "a", "--gpus", str(gpu), "--ctx", "131072", "--max-conc", "64"],
                env=env,
            )
            check(rr_budget.returncode != 0, "budget abort rc")
            budget_text = (rr_budget.stderr + rr_budget.stdout).lower()
            check("missing" in budget_text, "missing text")
            check(bool(re.search(r"\d+\.?\d*\s*gb", budget_text)), "gb figure")
            seed_installed_model(home, "a", str(model_a.relative_to(home)), static_w, kv_bytes, ctx, max_conc, 1)
            run_mesh([str(mesh), "unload", "--all"], env=env)
            model_auto = home / "models" / "org__auto"
            model_auto.mkdir(parents=True)
            (model_auto / "mesh_card.toml").write_text(
                f'model_id="auto"\ntp_size=0\npreferred_gpus=[]\ncontext_length={ctx}\n'
                f"max_running_requests={max_conc}\n",
                encoding="utf-8",
            )
            seed_installed_model(
                home,
                "auto",
                str(model_auto.relative_to(home)),
                5.0,
                kv_bytes,
                ctx,
                max_conc,
                0,
                kv_elems=kv_bytes,
                num_attention_heads=8,
                num_kv_heads=2,
            )
            rr_auto = run_mesh([str(mesh), "load", "auto"], env=env, timeout=300)
            check(rr_auto.returncode == 0, "auto load rc")
            auto_text = rr_auto.stdout + rr_auto.stderr
            check("auto:" in auto_text or "tp=" in auto_text, "auto decision message")
            run_mesh([str(mesh), "unload", "auto"], env=env)
            avail = test_gpus()
            if len(avail) >= 2:
                g0, g1 = avail[0], avail[1]
                rr_tp2 = run_mesh(
                    [str(mesh), "load", "auto", "--gpus", f"{g0},{g1}"],
                    env={**env, "CUDA_VISIBLE_DEVICES": f"{g0},{g1}"},
                    timeout=300,
                )
                check(rr_tp2.returncode == 0, "two gpu load rc")
                combined2 = rr_tp2.stdout + rr_tp2.stderr
                check("tp=2" in combined2, "gpus infer tp=2")
                stub_auto = _read_stub_argv(model_auto)
                check(int(stub_auto["parsed"].get("tp_size", 0)) == 2, "stub tp_size 2")
                run_mesh([str(mesh), "unload", "auto"], env=env)
            rr_conflict = run(
                [str(mesh), "load", "auto", "--tp", "1", "--gpus", f"{gpu},{gpu}"],
                env=env,
            )
            check(rr_conflict.returncode != 0, "tp gpu conflict rc")
            check("disagrees" in (rr_conflict.stderr + rr_conflict.stdout).lower(), "tp conflict msg")
            foreign = foreign_gpu_process(gpu, mb=10240)
            try:
                check(
                    wait_until(lambda: _gpu_used_mb(gpu) > 8000, timeout=20.0),
                    "foreign vram allocated",
                )
                rr_f = run([str(mesh), "load", "a", "--gpus", str(gpu)], env=env)
                check(rr_f.returncode == 3, "foreign exit 3")
                foreign_text = rr_f.stderr + rr_f.stdout
                check(
                    f"unmanaged process PID {foreign.popen.pid}" in foreign_text
                    or "VRAM not attributed to mesh" in foreign_text
                    or "VRAM in use with no mesh engines" in foreign_text,
                    "foreign msg",
                )
                run_mesh([str(mesh), "load", "a", "--gpus", str(gpu), "--force"], env=env)
            finally:
                foreign.stop()
            t0 = time.time()
            run_mesh([str(mesh), "unload", "--all"], env=env)
            check(wait_until(lambda: _no_defunct_children(), timeout=3.0), "no zombies after unload")
            check(time.time() - t0 < 3.5, "unload within 3s")
            left = sqlite_query(home / "state.db", "SELECT COUNT(*) FROM loaded_instances")[0][0]
            check(int(left) == 0, "unload all rows")
            rr_ps = run_mesh([str(mesh), "ps", "--json"], env=env)
            check(rr_ps.stdout.strip() in ("[]", ""), "mesh ps empty")
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env={**env, "STUB_IGNORE_SIGTERM": "1"}, timeout=300)
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu)], env={**env, "STUB_IGNORE_SIGTERM": "1"}, timeout=300)
            t1 = time.time()
            run_mesh([str(mesh), "unload", "--all"], env={**env, "STUB_IGNORE_SIGTERM": "1"}, timeout=120)
            check(wait_until(lambda: _no_defunct_children(), timeout=4.0), "sigkill unload no zombies")
            check(time.time() - t1 < 4.5, "unload within 4s with STUB_IGNORE_SIGTERM")
            run_mesh([str(mesh), "load", "t", "--gpus", str(gpu)], env=env, timeout=300)
            stub_t = _read_stub_argv(model_t)
            tenv = stub_t.get("env", {})
            check("TERNARY_PACK_DIR" in tenv or "TERNARY_MODEL_DIR" in tenv, "ternary card env")
            pyp = tenv.get("PYTHONPATH", "")
            check("ternary-kernels" in pyp, "ternary PYTHONPATH")
            check("MESH_KERNEL_DIR" not in tenv, "ternary no kernel dir")
            run_mesh([str(mesh), "unload", "t"], env=env)
            run_mesh(
                [str(mesh), "load", "a", "--gpus", str(gpu), "--context-length", "4096"],
                env=env,
                timeout=300,
            )
            stub_s = _read_stub_argv(model_a)
            check(stub_s.get("parsed", {}).get("context_length") == 4096, "load trailing sglang arg")
            run_mesh([str(mesh), "unload", "a"], env=env)
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            check(_wait_model_status("a", {"RUNNING"}, timeout=60.0), "watchdog prekill RUNNING")
            time.sleep(2.0)
            _kill_engine("a")
            check(
                wait_until(lambda: _restarts_at_least("a", 1), timeout=120.0),
                "watchdog restart count after SIGKILL",
            )
            run_mesh([str(mesh), "unload", "a"], env=env)
            stub_s = _read_stub_argv(model_a)
            check("TERNARY_PACK_DIR" not in stub_s.get("env", {}), "sglang no ternary env")
            sglang_pyp = stub_s.get("env", {}).get("PYTHONPATH", "")
            check("ternary-kernels" not in sglang_pyp, "sglang no ternary pythonpath")
            health_timeout = os.environ.get("MESH_HEALTH_POLL_TIMEOUT_SEC")
            if health_timeout and health_timeout.isdigit():
                delay = float(health_timeout) + 2.0
                run(
                    [str(mesh), "load", "b", "--gpus", str(gpu)],
                    env={**env, "STUB_START_DELAY": str(delay)},
                    timeout=600,
                )
                st_row = sqlite_query(
                    home / "state.db",
                    "SELECT status FROM loaded_instances WHERE model_id='b' ORDER BY rowid DESC LIMIT 1",
                )
                check(bool(st_row) and st_row[0][0] == "FAILED", "health timeout FAILED")
                run_mesh([str(mesh), "unload", "b"], env=env)
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            md_pid = meshd.popen.pid
            os.kill(md_pid, signal.SIGTERM)
            meshd.popen.wait(timeout=15)
            meshd._stopped = True
            meshd = _start_meshd(home)
            check(
                _wait_model_status("a", {"RUNNING", "STARTING"}, timeout=120.0),
                "meshd SIGTERM restart restores desired load",
            )
            run_mesh([str(mesh), "unload", "a"], env=env)
            desired_a = sqlite_query(
                home / "state.db", "SELECT COUNT(*) FROM desired_loads WHERE model_id='a'"
            )[0][0]
            check(int(desired_a) == 0, "unload clears desired")
            meshd.stop()
            meshd = _start_meshd(home)
            time.sleep(2.0)
            left = sqlite_query(
                home / "state.db", "SELECT COUNT(*) FROM loaded_instances WHERE model_id='a'"
            )[0][0]
            check(int(left) == 0, "unloaded model not restored after meshd restart")
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            rr_stop = run_mesh([str(mesh), "stop"], env=home_mesh_env(home))
            check("meshd stopped" in rr_stop.stdout, "mesh stop message")
            meshd._stopped = True
            _clean_meshd_stale(home)
            meshd = _start_meshd(home)
            check(
                _wait_model_status("a", {"RUNNING", "STARTING"}, timeout=120.0),
                "mesh stop then start restores desired",
            )
            run_mesh([str(mesh), "unload", "--all"], env=env)
            meshd.stop()
            _clean_meshd_stale(home)
            meshd = _start_meshd(home)
            dup = subprocess.run(
                [str(require_cli("meshd"))],
                env=home_mesh_env(home),
                capture_output=True,
                text=True,
                timeout=10,
            )
            check(dup.returncode == 1, "second meshd exit 1")
            dup_msg = (dup.stderr + dup.stdout).lower()
            check("already running" in dup_msg or "mesh.sock" in dup_msg, "second meshd refused")
            run_mesh([str(mesh), "unload", "--all"], env=env)
            slow_env = {**env, "STUB_START_DELAY": "25"}
            run_mesh(
                [str(mesh), "load", "a", "--gpus", str(gpu), "--no-wait"],
                env=slow_env,
                timeout=60,
            )
            t_ps = time.time()
            rr_ps_load = run_mesh([str(mesh), "ps", "--json"], env=slow_env, timeout=30)
            check(rr_ps_load.returncode == 0 and time.time() - t_ps < 10.0, "ps during load")
            run_mesh([str(mesh), "unload", "a"], env=slow_env, timeout=120)
            for p in (model_b / "stub_health_mode.txt", model_b / "stub_health_delay.txt"):
                if p.is_file():
                    p.unlink()
            (model_b / "stub_health_mode.txt").write_text("late503", encoding="utf-8")
            (model_b / "stub_health_delay.txt").write_text("20", encoding="utf-8")
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu), "--no-wait"], env=env, timeout=60)
            time.sleep(8.0)
            row_b = next((r for r in _ps_json() if r.get("model_id") == "b"), None)
            check(row_b is not None and row_b.get("status") == "STARTING", "late health stays STARTING")
            check(meshd.popen.poll() is None, "meshd alive during late startup")
            check(_wait_model_status("b", {"RUNNING"}, timeout=90.0), "late health reaches RUNNING")
            run_mesh([str(mesh), "unload", "b"], env=env)
            (model_b / "stub_health_mode.txt").write_text("warmup", encoding="utf-8")
            (model_b / "stub_health_delay.txt").write_text("12", encoding="utf-8")
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu), "--no-wait"], env=env, timeout=60)
            time.sleep(2.0)
            row_b = next((r for r in _ps_json() if r.get("model_id") == "b"), None)
            check(row_b is not None and row_b.get("status") == "STARTING", "warmup not RUNNING early")
            check(_wait_model_status("b", {"RUNNING"}, timeout=60.0), "warmup RUNNING after health")
            run_mesh([str(mesh), "unload", "b"], env=env)
            run_mesh([str(mesh), "unload", "--all"], env=env)
            for model_dir in (model_a, model_b):
                for p in model_dir.glob("stub_*"):
                    if p.is_file():
                        p.unlink()
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            spawns_before = 0
            idx = home / "logs" / "index.jsonl"
            if idx.is_file():
                spawns_before = sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())
            eng_pgid = int(
                sqlite_query(
                    home / "state.db",
                    "SELECT pgid FROM loaded_instances WHERE model_id='a' LIMIT 1",
                )[0][0]
            )
            check(eng_pgid != meshd.popen.pid, "engine pgid not meshd")
            os.killpg(eng_pgid, signal.SIGKILL)
            check(
                wait_until(lambda: _restarts_at_least("a", 1), timeout=90.0),
                "single restart after kill -9",
            )
            check(_wait_model_status("a", {"RUNNING"}, timeout=120.0), "back running after kill")
            spawns_after = sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())
            check(spawns_after == spawns_before + 1, "exactly one extra spawn")
            (model_b / "stub_health_mode.txt").write_text("burst503", encoding="utf-8")
            (model_b / "stub_health_burst_delay.txt").write_text("2", encoding="utf-8")
            (model_b / "stub_health_burst_sec.txt").write_text("8", encoding="utf-8")
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu)], env=env, timeout=300)
            check(_wait_model_status("b", {"RUNNING"}, timeout=60.0), "burst503 load RUNNING")
            time.sleep(3.0)
            row_b = next(r for r in _ps_json() if r.get("model_id") == "b")
            check(row_b.get("status") == "RUNNING" and int(row_b.get("restarts", 0)) == 0, "burst active")
            time.sleep(9.0)
            row_b = next(r for r in _ps_json() if r.get("model_id") == "b")
            check(row_b.get("status") == "RUNNING" and int(row_b.get("restarts", 0)) == 0, "burst503 survives")
            run_mesh([str(mesh), "unload", "b"], env=env)
            (model_b / "stub_cheap_mode.txt").write_text("503_after", encoding="utf-8")
            run_mesh([str(mesh), "load", "b", "--gpus", str(gpu)], env=env, timeout=300)
            t_h = time.time()
            check(
                wait_until(lambda: _restarts_at_least("b", 1), timeout=120.0),
                "sustained cheap probe restart",
            )
            check(time.time() - t_h >= 25.0, "restart after sustained cheap fail")
            run_mesh([str(mesh), "unload", "b"], env=env, timeout=120)
            check(meshd.popen.poll() is None, "meshd alive after hung probe")
            run_mesh([str(mesh), "unload", "--all"], env=env, timeout=120)
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=300)
            check(
                sum(
                    1
                    for r in _ps_json()
                    if r.get("model_id") == "a" and r.get("status") == "RUNNING"
                )
                >= 2,
                "two RUNNING instances",
            )
            check(
                int(
                    sqlite_query(
                        home / "state.db",
                        "SELECT COUNT(*) FROM desired_loads WHERE model_id='a'",
                    )[0][0]
                )
                == 2,
                "two desired slots",
            )
            meshd.stop()
            _clean_meshd_stale(home)
            meshd = _start_meshd(home)
            check(
                wait_until(
                    lambda: sum(
                        1 for r in _ps_json() if r.get("model_id") == "a" and r.get("status") == "RUNNING"
                    )
                    >= 2,
                    timeout=300.0,
                ),
                "restore both slots",
            )
            run_mesh([str(mesh), "unload", "a"], env=env, timeout=120)
            left_inst = sqlite_query(
                home / "state.db", "SELECT COUNT(*) FROM loaded_instances WHERE model_id='a'"
            )[0][0]
            left_des = sqlite_query(
                home / "state.db", "SELECT COUNT(*) FROM desired_loads WHERE model_id='a'"
            )[0][0]
            check(int(left_inst) == 0 and int(left_des) == 0, "unload all slots of model")
        finally:
            release_gpu()
            meshd.stop()
            assert_no_stub_engines(home)
    print(f"stage4_supervisor_stub ok assertions={ASSERTS}")


if __name__ == "__main__":
    main_exit(main)
