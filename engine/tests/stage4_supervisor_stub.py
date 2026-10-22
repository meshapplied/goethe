#!/usr/bin/env python3
"""Stage 4 goethed supervisor gate with stub engines."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from lib.goethe_test import (
    Proc,
    _acquire_gpu_flocks,
    assert_alive,
    assert_no_stub_engines,
    assert_no_zombies,
    clean_test_gpu,
    foreign_gpu_process,
    goethe_rpc,
    gpu_compute_pids,
    gpu_pids,
    home_goethe_env,
    home_process_tree,
    kill_engine_group,
    main_exit,
    pid_alive,
    require_cli,
    require_gpus,
    run,
    run_goethe,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    stub_launch,
    temp_home,
    test_gpus,
    vram_per_gpu_gb,
    wait_gpu_pid,
    wait_until,
    write_card,
)

ASSERTS = 0
SOFT: list[str] = []
FOREIGN_MB = 3072


def check(cond: bool, msg: str) -> None:
    """Record one assertion."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def soft(cond: bool, msg: str) -> None:
    """Record an assertion whose failure is reported at the end without stopping later scenarios."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        SOFT.append(msg)
        print(f"SOFT FAIL: {msg}", flush=True)


def _hw_fixture(home: Path, gpu_index: int, free_gb: float = 90.0) -> None:
    """Write hardware_state.json with VRAM numbers for budget tests."""
    total_mb = int((free_gb / 0.85) * 1024)
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
                "vram_used_mb": total_mb - int(free_gb * 1024),
                "vram_free_mb": int(free_gb * 1024),
                "pcie_gen": 4,
                "pcie_width": 16,
                "running_processes": [],
            }
        ],
        "links": [],
    }
    (home / "hardware_state.json").write_text(json.dumps(payload), encoding="utf-8")


def _live_free_gb(gpu_index: int) -> float:
    """Query NVML free VRAM for mem_fraction_static checks."""
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip()) / 1024.0


def _clean_goethed_stale(home: Path) -> None:
    """Remove goethe.sock and goethed.lock after a controlled goethed stop."""
    for name in ("goethe.sock", "goethed.lock"):
        (home / name).unlink(missing_ok=True)


def _clear_stub_files(*dirs: Path) -> None:
    for model_dir in dirs:
        for p in model_dir.glob("stub_*"):
            p.unlink()


class Sup:
    """One goethed under a temp home plus helpers for the supervisor scenarios."""

    def __init__(self, home: Path, goethe: Path) -> None:
        self.home = home
        self.goethe = goethe
        self.env = home_goethe_env(home)
        self.db = home / "state.db"
        self.md: Proc = start_goethed(home)

    def goethe_ok(self, *args: str, timeout: float = 300) -> str:
        """Run goethe, require success, return stdout plus stderr."""
        rr = run_goethe([str(self.goethe), *args], env=self.env, timeout=timeout)
        return rr.stdout + rr.stderr

    def load(self, mid: str, gpus: str, *extra: str, timeout: float = 300) -> str:
        """Load with --yes (the alien processes on GPU 2 would otherwise exit 3)."""
        return self.goethe_ok("load", mid, "--gpus", gpus, "--yes", *extra, timeout=timeout)

    def ps(self) -> list[dict[str, Any]]:
        return json.loads(self.goethe_ok("ps", "--json").strip() or "[]")

    def status_is(self, mid: str, want: set[str], timeout: float) -> bool:
        """Wait until the newest ps row of mid has a status in want."""

        def ok() -> bool:
            rows = [r for r in self.ps() if r.get("model_id") == mid]
            return bool(rows) and rows[-1].get("status") in want

        return wait_until(ok, timeout=timeout)

    def restarts_at_least(self, mid: str, n: int) -> bool:
        """True once mid shows n restarts in ps or the DB."""
        if any(r.get("model_id") == mid and int(r.get("restarts", 0)) >= n for r in self.ps()):
            return True
        q = "SELECT COALESCE(restart_count,0) FROM loaded_instances WHERE model_id=?1 ORDER BY rowid DESC LIMIT 1"
        row = sqlite_query(self.db, q, (mid,))
        return bool(row) and int(row[0][0]) >= n

    def kill_engine(self, mid: str) -> None:
        """SIGKILL the newest engine group of mid after checking it belongs to this home."""
        q = "SELECT pgid FROM loaded_instances WHERE model_id=?1 ORDER BY rowid DESC LIMIT 1"
        kill_engine_group(self.home, int(sqlite_query(self.db, q, (mid,))[0][0]))

    def restart(self) -> None:
        """Stop goethed (SIGTERM via Proc.stop, a no-op once it exited) and start a new one."""
        self.md.stop()
        _clean_goethed_stale(self.home)
        self.md = start_goethed(self.home)

    def count(self, sql: str) -> int:
        return int(sqlite_query(self.db, sql)[0][0])


def _foreign_scenario(sup: Sup, model_a: Path) -> None:
    """Unmanaged GPU process: exit 3 with the contractual notice, structured RPC result, --yes loads alongside."""
    clean = clean_test_gpu()
    host = clean if clean is not None else test_gpus()[-1]
    foreign = foreign_gpu_process(host, mb=FOREIGN_MB)
    try:
        check(wait_gpu_pid(host, foreign.popen.pid, int(FOREIGN_MB * 0.9), timeout=60), "foreign VRAM visible")
        strict = {p for p, _mb in gpu_pids(host)} == {foreign.popen.pid}
        rr = run([str(sup.goethe), "load", "a", "--gpus", str(host)], env=sup.env, timeout=120)
        text = rr.stderr + rr.stdout
        check(rr.returncode == 3, f"foreign load exit {rr.returncode}, want 3: {text[-300:]}")
        listed = {int(p) for p in re.findall(r"unmanaged process PID (\d+)", text)}
        on_gpu = {p for p, _mb in gpu_pids(host)}
        if strict:
            check(foreign.popen.pid in listed, f"notice names our foreign PID {foreign.popen.pid}: {text[-300:]}")
        else:
            check(bool(listed) and listed <= on_gpu, f"notice names a real foreign PID on GPU {host}: {listed}")
        params = {
            "model_id": "a", "gpus": str(host), "ctx": None, "max_conc": None, "tp": None, "force": False,
            "yes": False, "is_tty": True, "extra_sglang_args": [], "async_finish": True,
        }
        t0 = time.time()
        reply = goethe_rpc(sup.home, "load", params, timeout=30)
        check(time.time() - t0 < 25, "daemon answered the foreign load without waiting on a prompt")
        soft(reply.get("code") == 3, f"RPC foreign code {reply.get('code')} want 3 (daemon must never prompt)")
        soft(f"unmanaged process PID {foreign.popen.pid}" in str(reply.get("message", "")), "RPC message names PID")
        rows = (reply.get("result") or {}).get("foreign") or []
        soft(
            any(int(r.get("pid", -1)) == foreign.popen.pid and int(r.get("gpu", -1)) == host for r in rows),
            f"RPC result.foreign lists gpu {host} pid {foreign.popen.pid}: {reply.get('result')}",
        )
        soft(
            "Re-run with --yes" in text and f"kill {foreign.popen.pid}" in text,
            "notice suggests --yes and the user's own kill command",
        )
        check(pid_alive(foreign.popen.pid), "goethe never stops the foreign process")
        sup.load("a", str(host))
        check(pid_alive(foreign.popen.pid), "foreign process still alive after loading alongside it")
    finally:
        foreign.stop()
    sup.goethe_ok("unload", "--all")
    _clear_stub_files(model_a)


def main() -> None:
    """Exercise goethe load, co-residency, foreign PID, unload and watchdog paths with stub engines."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    gpu = require_gpus(1)[0]
    aliens = gpu_compute_pids(test_gpus())
    static_w, kv_bytes, ctx, max_conc = 10.0, 1000, 4096, 4
    release_gpu = _acquire_gpu_flocks([gpu])
    try:
        with temp_home(stub_python=True) as home:
            _hw_fixture(home, gpu, free_gb=90.0)
            model_a = home / "models" / "org__model_a"
            model_b = home / "models" / "org__model_b"
            model_t = home / "models" / "org__ternary_t"
            model_auto = home / "models" / "org__auto"
            write_card(model_a, "a", ctx, max_conc)
            write_card(model_b, "b", 4096, 4, extra='kv_cache_dtype="fp8_e4m3"')
            seed_installed_model(home, "a", "models/org__model_a", static_w, kv_bytes, ctx, max_conc, 1)
            seed_installed_model(home, "b", "models/org__model_b", static_w, kv_bytes, ctx, max_conc, 1)
            (home / "goethe" / "ternary-kernels").mkdir(parents=True)
            write_card(
                model_t, "t", 512, 2,
                extra=f'runtime="ternary"\npack_id="p"\nbase_model_id="b"\n'
                f'pythonpath=["{home}/goethe/ternary-kernels"]\n'
                f'TERNARY_PACK_DIR="{home}/models/pack"\nTERNARY_MODEL_DIR="{home}/models/base"',
            )
            seed_installed_model(home, "t", "models/org__ternary_t", 5.0, 512, 512, 2, 1)
            sup = Sup(home, goethe)
            try:
                _scenarios(sup, gpu, model_a, model_b, model_t, model_auto, static_w, kv_bytes, ctx, max_conc)
            finally:
                sup.md.stop()
            assert_no_stub_engines(home)
            check(wait_until(lambda: not home_process_tree(home), timeout=10), "no process left under the home")
    finally:
        release_gpu()
    assert_alive(aliens, "alien GPU processes")
    if SOFT:
        raise RuntimeError(f"{len(SOFT)} contract checks failed: " + " | ".join(SOFT))
    print(f"stage4_supervisor_stub ok assertions={ASSERTS}")


def _scenarios(
    sup: Sup, gpu: int, model_a: Path, model_b: Path, model_t: Path, model_auto: Path,
    static_w: float, kv_bytes: int, ctx: int, max_conc: int,
) -> None:
    """Run the supervisor scenarios in order against one goethed."""
    home = sup.home
    sup.load("a", str(gpu), "--ctx", str(ctx), "--max-conc", str(max_conc))
    rows = sqlite_query(sup.db, "SELECT pid, pgid, internal_port, gpu_indices, status FROM loaded_instances")
    check(len(rows) == 1, "one instance a")
    pid, pgid, port, gpu_indices, status = rows[0]
    check(int(pid) > 0 and int(pgid) == int(pid), "pgid==pid")
    check(30001 <= int(port) <= 30099, "engine port in range")
    check(str(gpu) in str(gpu_indices) and status == "RUNNING", "gpu_indices and RUNNING")
    check(sup.count("SELECT COUNT(*) FROM desired_loads") >= 1, "desired load row")
    stub = stub_launch(model_a)
    parsed = stub["parsed"]
    b_gb = vram_per_gpu_gb(static_w, ctx, kv_bytes, max_conc, 1)
    check(abs(float(parsed["mem_fraction_static"]) - b_gb / _live_free_gb(gpu)) < 0.02, "mem_fraction_static")
    check(parsed["host"] == "127.0.0.1" and int(parsed["port"]) == int(port), "host and port")
    check(int(parsed["tp_size"]) == 1 and int(parsed["context_length"]) == ctx, "tp-size and context-length")
    check(int(parsed["max_running_requests"]) == max_conc, "max-running-requests")
    soft("--kv-cache-dtype" not in stub["argv"], "card without kv_cache_dtype must not pass --kv-cache-dtype")
    check(stub["env"].get("CUDA_VISIBLE_DEVICES") == str(gpu), "cuda visible")
    check("GOETHE_KERNEL_DIR" not in stub["env"], "no GOETHE_KERNEL_DIR")
    check("--enable-goethe-fused-gemv" not in stub["argv"], "no goethe-fused-gemv flag")
    soft(bool(stub["env"].get("GOETHE_INSTANCE_ID")), "engine env carries GOETHE_INSTANCE_ID")
    combined = sup.load("b", str(gpu))
    check("currently hosts [" in combined and "Free VRAM:" in combined, "co-residency notice")
    check(stub_launch(model_b)["parsed"].get("kv_cache_dtype") == "fp8_e4m3", "card kv_cache_dtype passed")
    ports = [int(r[0]) for r in sqlite_query(sup.db, "SELECT internal_port FROM loaded_instances")]
    check(len(ports) == 2 and len(set(ports)) == 2, "two engines on distinct ports")
    check(all(30001 <= p <= 30099 for p in ports), "engine ports in range")
    seed_installed_model(home, "a", "models/org__model_a", 200.0, kv_bytes, 131072, 64, 1)
    on_gpu = {p for p, _mb in gpu_pids(gpu)}
    rr = run(
        [str(sup.goethe), "load", "a", "--gpus", str(gpu), "--ctx", "131072", "--max-conc", "64", "--yes"],
        env=sup.env,
    )
    budget_text = (rr.stderr + rr.stdout).lower()
    check(rr.returncode != 0, "budget abort fails")
    soft(rr.returncode == 1, f"budget abort exit {rr.returncode}, want 1")
    check("missing" in budget_text and bool(re.search(r"\d+\.?\d*\s*gb", budget_text)), "missing GB figure")
    if on_gpu:
        soft("unmanaged process pid" in budget_text, "budget abort lists the foreign processes holding VRAM")
    seed_installed_model(home, "a", "models/org__model_a", static_w, kv_bytes, ctx, max_conc, 1)
    sup.goethe_ok("unload", "--all")
    write_card(model_auto, "auto", ctx, max_conc, extra="tp_size=0\npreferred_gpus=[]")
    seed_installed_model(
        home, "auto", "models/org__auto", 5.0, kv_bytes, ctx, max_conc, 0,
        kv_elems=kv_bytes, num_attention_heads=8, num_kv_heads=2,
    )
    auto_text = sup.goethe_ok("load", "auto", "--yes")
    check("auto:" in auto_text or "tp=" in auto_text, "auto decision message")
    sup.goethe_ok("unload", "auto")
    avail = test_gpus()
    if len(avail) >= 2:
        two = f"{avail[0]},{avail[1]}"
        out2 = sup.load("auto", two)
        check("tp=2" in out2, "gpus infer tp=2")
        launch = stub_launch(model_auto)
        check(int(launch["parsed"]["tp_size"]) == 2 and len(launch["tp_pids"]) == 1, "stub tp_size 2 with one rank")
        check(sup.count("SELECT COUNT(*) FROM loaded_instances WHERE model_id='auto'") == 1, "one TP instance")
        sup.goethe_ok("unload", "auto")
        check(not any(pid_alive(p) for p in [launch["pid"], *launch["tp_pids"]]), "TP leader and rank gone")
    rr_conflict = run([str(sup.goethe), "load", "auto", "--tp", "1", "--gpus", f"{gpu},{gpu}", "--yes"], env=sup.env)
    check(rr_conflict.returncode != 0, "tp gpu conflict rc")
    check("disagrees" in (rr_conflict.stderr + rr_conflict.stdout).lower(), "tp conflict msg")
    _foreign_scenario(sup, model_a)
    check(sup.count("SELECT COUNT(*) FROM loaded_instances") == 0, "unload all rows")
    check(sup.goethe_ok("ps", "--json").strip() in ("[]", ""), "goethe ps empty")
    sup.load("a", str(gpu))
    t0 = time.time()
    sup.goethe_ok("unload", "--all")
    soft(time.time() - t0 < 3.5, f"unload within 3.5s ({time.time() - t0:.1f}s)")
    assert_no_zombies(home)
    write_card(model_a, "a", ctx, max_conc, env={"STUB_IGNORE_SIGTERM": "1"})
    write_card(model_b, "b", 4096, 4, env={"STUB_IGNORE_SIGTERM": "1"})
    sup.load("a", str(gpu))
    sup.load("b", str(gpu))
    term_pids = [stub_launch(m)["pid"] for m in (model_a, model_b)]
    check(all(stub_launch(m)["knobs"].get("STUB_IGNORE_SIGTERM") == "1" for m in (model_a, model_b)), "knob active")
    t1 = time.time()
    sup.goethe_ok("unload", "--all", timeout=120)
    dt = time.time() - t1
    soft(dt < 4.5, f"TERM-ignoring unload --all within 4s grace ({dt:.1f}s)")
    check(not any(pid_alive(p) for p in term_pids), "TERM-ignoring engines killed")
    assert_no_zombies(home, timeout=4.0)
    write_card(model_a, "a", ctx, max_conc)
    write_card(model_b, "b", 4096, 4, extra='kv_cache_dtype="fp8_e4m3"')
    _clear_stub_files(model_a, model_b)
    sup.load("t", str(gpu))
    tenv = stub_launch(model_t)["env"]
    check("TERNARY_PACK_DIR" in tenv or "TERNARY_MODEL_DIR" in tenv, "ternary card env")
    check("ternary-kernels" in tenv.get("PYTHONPATH", ""), "ternary PYTHONPATH")
    check("GOETHE_KERNEL_DIR" not in tenv, "ternary no kernel dir")
    sup.goethe_ok("unload", "t")
    sup.load("a", str(gpu), "--context-length", "4096")
    check(stub_launch(model_a)["parsed"].get("context_length") == 4096, "load trailing sglang arg")
    sup.goethe_ok("unload", "a")
    sup.load("a", str(gpu))
    check(sup.status_is("a", {"RUNNING"}, timeout=60.0), "watchdog prekill RUNNING")
    time.sleep(2.0)
    sup.kill_engine("a")
    check(wait_until(lambda: sup.restarts_at_least("a", 1), timeout=120.0), "watchdog restart count after SIGKILL")
    sup.goethe_ok("unload", "a")
    stub_s = stub_launch(model_a)
    check("TERNARY_PACK_DIR" not in stub_s["env"], "sglang no ternary env")
    check("ternary-kernels" not in stub_s["env"].get("PYTHONPATH", ""), "sglang no ternary pythonpath")
    health_timeout = os.environ.get("GOETHE_HEALTH_POLL_TIMEOUT_SEC")
    if health_timeout and health_timeout.isdigit():
        write_card(model_b, "b", 4096, 4, env={"STUB_START_DELAY": str(float(health_timeout) + 2.0)})
        run([str(sup.goethe), "load", "b", "--gpus", str(gpu), "--yes"], env=sup.env, timeout=600)
        q = "SELECT status FROM loaded_instances WHERE model_id='b' ORDER BY rowid DESC LIMIT 1"
        st_row = sqlite_query(sup.db, q)
        check(bool(st_row) and st_row[0][0] == "FAILED", "health timeout FAILED")
        sup.goethe_ok("unload", "b")
        write_card(model_b, "b", 4096, 4, extra='kv_cache_dtype="fp8_e4m3"')
    sup.load("a", str(gpu))
    os.kill(sup.md.popen.pid, signal.SIGTERM)
    sup.md.popen.wait(timeout=30)
    sup.restart()
    check(sup.status_is("a", {"RUNNING", "STARTING"}, timeout=120.0), "goethed SIGTERM restart restores desired load")
    sup.goethe_ok("unload", "a")
    check(sup.count("SELECT COUNT(*) FROM desired_loads WHERE model_id='a'") == 0, "unload clears desired")
    sup.restart()
    time.sleep(2.0)
    check(sup.count("SELECT COUNT(*) FROM loaded_instances WHERE model_id='a'") == 0, "unloaded model not restored")
    sup.load("a", str(gpu))
    check("goethed stopped" in sup.goethe_ok("stop"), "goethe stop message")
    sup.md.popen.wait(timeout=30)
    sup.restart()
    check(sup.status_is("a", {"RUNNING", "STARTING"}, timeout=120.0), "goethe stop then start restores desired")
    sup.goethe_ok("unload", "--all")
    sup.restart()
    dup = run([str(require_cli("goethed"))], env=sup.env, timeout=10)
    check(dup.returncode == 1, "second goethed exit 1")
    dup_msg = (dup.stderr + dup.stdout).lower()
    check("already running" in dup_msg or "goethe.sock" in dup_msg, "second goethed refused")
    write_card(model_a, "a", ctx, max_conc, env={"STUB_START_DELAY": "25"})
    _clear_stub_files(model_a)
    sup.load("a", str(gpu), "--no-wait", timeout=60)
    t_ps = time.time()
    run_goethe([str(sup.goethe), "ps", "--json"], env=sup.env, timeout=30)
    check(time.time() - t_ps < 10.0, "ps during load")
    check(wait_until(lambda: (model_a / "stub_launch_argv.json").is_file(), timeout=15), "delayed stub started")
    check(stub_launch(model_a)["knobs"].get("STUB_START_DELAY") == "25", "start delay knob active")
    sup.goethe_ok("unload", "a", timeout=120)
    write_card(model_a, "a", ctx, max_conc)
    _clear_stub_files(model_a, model_b)
    (model_b / "stub_health_mode.txt").write_text("late503", encoding="utf-8")
    (model_b / "stub_health_delay.txt").write_text("20", encoding="utf-8")
    sup.load("b", str(gpu), "--no-wait", timeout=60)
    time.sleep(8.0)
    row_b = next((r for r in sup.ps() if r.get("model_id") == "b"), None)
    check(row_b is not None and row_b.get("status") == "STARTING", "late health stays STARTING")
    check(sup.md.popen.poll() is None, "goethed alive during late startup")
    check(sup.status_is("b", {"RUNNING"}, timeout=90.0), "late health reaches RUNNING")
    sup.goethe_ok("unload", "b")
    (model_b / "stub_health_mode.txt").write_text("warmup", encoding="utf-8")
    (model_b / "stub_health_delay.txt").write_text("12", encoding="utf-8")
    sup.load("b", str(gpu), "--no-wait", timeout=60)
    time.sleep(2.0)
    row_b = next((r for r in sup.ps() if r.get("model_id") == "b"), None)
    check(row_b is not None and row_b.get("status") == "STARTING", "warmup not RUNNING early")
    check(sup.status_is("b", {"RUNNING"}, timeout=60.0), "warmup RUNNING after health")
    sup.goethe_ok("unload", "--all")
    _clear_stub_files(model_a, model_b)
    sup.load("a", str(gpu))
    idx = home / "logs" / "index.jsonl"
    spawns_before = sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())
    eng_pgid = int(sqlite_query(sup.db, "SELECT pgid FROM loaded_instances WHERE model_id='a' LIMIT 1")[0][0])
    check(eng_pgid != sup.md.popen.pid, "engine pgid not goethed")
    kill_engine_group(home, eng_pgid)
    check(wait_until(lambda: sup.restarts_at_least("a", 1), timeout=90.0), "single restart after kill -9")
    check(sup.status_is("a", {"RUNNING"}, timeout=120.0), "back running after kill")
    spawns_after = sum(1 for ln in idx.read_text(encoding="utf-8").splitlines() if ln.strip())
    check(spawns_after == spawns_before + 1, "exactly one extra spawn")
    (model_b / "stub_health_mode.txt").write_text("burst503", encoding="utf-8")
    (model_b / "stub_health_burst_delay.txt").write_text("2", encoding="utf-8")
    (model_b / "stub_health_burst_sec.txt").write_text("8", encoding="utf-8")
    sup.load("b", str(gpu))
    check(sup.status_is("b", {"RUNNING"}, timeout=60.0), "burst503 load RUNNING")
    time.sleep(3.0)
    row_b = next(r for r in sup.ps() if r.get("model_id") == "b")
    check(row_b.get("status") == "RUNNING" and int(row_b.get("restarts", 0)) == 0, "burst active")
    time.sleep(9.0)
    row_b = next(r for r in sup.ps() if r.get("model_id") == "b")
    check(row_b.get("status") == "RUNNING" and int(row_b.get("restarts", 0)) == 0, "burst503 survives")
    sup.goethe_ok("unload", "b")
    _clear_stub_files(model_b)
    (model_b / "stub_cheap_mode.txt").write_text("503_after", encoding="utf-8")
    sup.load("b", str(gpu))
    t_h = time.time()
    check(wait_until(lambda: sup.restarts_at_least("b", 1), timeout=120.0), "sustained cheap probe restart")
    check(time.time() - t_h >= 25.0, "restart after sustained cheap fail")
    sup.goethe_ok("unload", "b", timeout=120)
    check(sup.md.popen.poll() is None, "goethed alive after hung probe")
    sup.goethe_ok("unload", "--all", timeout=120)
    _clear_stub_files(model_b)
    sup.load("a", str(gpu))
    sup.load("a", str(gpu))
    running_a = sum(1 for r in sup.ps() if r.get("model_id") == "a" and r.get("status") == "RUNNING")
    check(running_a >= 2, "two RUNNING instances")
    check(sup.count("SELECT COUNT(*) FROM desired_loads WHERE model_id='a'") == 2, "two desired slots")
    sup.restart()
    check(
        wait_until(
            lambda: sum(1 for r in sup.ps() if r.get("model_id") == "a" and r.get("status") == "RUNNING") >= 2,
            timeout=300.0,
        ),
        "restore both slots",
    )
    sup.goethe_ok("unload", "a", timeout=120)
    left_inst = sup.count("SELECT COUNT(*) FROM loaded_instances WHERE model_id='a'")
    left_des = sup.count("SELECT COUNT(*) FROM desired_loads WHERE model_id='a'")
    check(left_inst == 0 and left_des == 0, "unload all slots of model")
    assert_no_zombies(home)


if __name__ == "__main__":
    main_exit(main)
