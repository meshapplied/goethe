#!/usr/bin/env python3
"""Stage 4: goethe never signals processes it did not start and reports them with commands the user can run."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lib.goethe_test import (
    Proc,
    assert_no_stub_engines,
    foreign_gpu_process,
    free_port,
    goethe_rpc,
    gpu_pids,
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
    start_ticks,
    temp_home,
    wait_until,
)


def check(cond: bool, msg: str) -> None:
    """Fail the gate with msg when cond is false."""
    if not cond:
        raise RuntimeError(msg)


def _boot_id() -> str:
    """Current kernel boot id."""
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


class _AlwaysOk(BaseHTTPRequestHandler):
    """Answer every GET with 200 so only the identity check can refuse adoption."""

    def do_GET(self) -> None:
        """Reply 200 ok."""
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _seed_stale_row(home: Path, gpu: int, instance_id: str, pid: int, port: int, status: str) -> None:
    """Insert an instance row that points at pid with this boot id but a wrong start time."""
    con = sqlite3.connect(str(home / "state.db"))
    try:
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, "
            "allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, slot_key, boot_id, "
            "start_ticks, engine_key) VALUES (?, 'ghost', ?, ?, ?, ?, 1.0, 0.5, 4096, 4, ?, ?, ?, ?, 'k')",
            (
                instance_id,
                pid,
                pid,
                port,
                str(gpu),
                status,
                f"ghost|{gpu}|0",
                _boot_id(),
                start_ticks(pid) + 7919,
            ),
        )
        con.execute(
            "INSERT OR REPLACE INTO desired_loads (slot_key, model_id) VALUES (?, 'ghost')", (f"ghost|{gpu}|0",)
        )
        con.commit()
    finally:
        con.close()


def _ghost_rows(home: Path) -> list[tuple[object, ...]]:
    """Instance rows (id, status, pid) of the seeded ghost model."""
    return sqlite_query(
        home / "state.db", "SELECT instance_id, status, pid FROM loaded_instances WHERE model_id='ghost'"
    )


def main() -> None:
    """Alien VRAM holder on the roomiest test GPU: exit 3 notice, --yes load, and nothing goethe does touches it."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu(8.0)
    with temp_home(stub_python=True, tag="alien") as home:
        model = home / "models" / "org__alien_stub"
        model.mkdir(parents=True)
        (model / "goethe_card.toml").write_text(
            'model_id="alien-stub"\ncontext_length=4096\nmax_running_requests=4\n', encoding="utf-8"
        )
        alien = foreign_gpu_process(gpu, mb=512)
        apid = alien.popen.pid
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysOk)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        md: Proc | None = None
        try:
            check(wait_until(lambda: any(p == apid for p, _ in gpu_pids(gpu)), timeout=90.0), f"alien on GPU {gpu}")
            md = start_goethed(home)
            seed_installed_model(home, "alien-stub", str(model.relative_to(home)), 1.0, 1000, 4096, 4, 1)
            env = home_goethe_env(home)
            rr = run([str(goethe), "load", "--model", "alien-stub", "--gpus", str(gpu)], env=env, input_text="")
            text = rr.stdout + rr.stderr
            check(rr.returncode == 3, f"load without --yes must exit 3, got {rr.returncode}: {text[-400:]}")
            check(f"unmanaged process PID {apid}" in text, f"notice names the alien pid: {text[-400:]}")
            check(f"kill {apid}" in text and "Re-run with --yes" in text, "notice suggests kill and --yes")
            reply = goethe_rpc(home, "load", {"model_id": "alien-stub", "gpus": str(gpu)})
            foreign = reply.get("result", {}).get("foreign", []) if isinstance(reply.get("result"), dict) else []
            check(reply.get("code") == 3 and any(f.get("pid") == apid for f in foreign), f"rpc foreign: {reply}")
            check(pid_alive(apid), "alien alive after the notice")
            load = [str(goethe), "load", "--model", "alien-stub", "--gpus", str(gpu), "--yes"]
            run_goethe(load, env=env, timeout=120)
            run_goethe([str(goethe), "unload", "--model", "alien-stub"], env=env, timeout=60)
            check(pid_alive(apid), "alien survives unload")
            _seed_stale_row(home, gpu, "inst-ghost-watchdog", apid, free_port(), "RUNNING")
            tomb = [("inst-ghost-watchdog", "FAILED", 0)]
            check(wait_until(lambda: _ghost_rows(home) == tomb, timeout=30.0), "watchdog tombstones an unverified row")
            check(pid_alive(apid), "alien survives the watchdog")
            _seed_stale_row(home, gpu, "inst-ghost-unload", apid, free_port(), "STARTING")
            run_goethe([str(goethe), "unload", "--model", "ghost"], env=env, timeout=60)
            check(not _ghost_rows(home) and pid_alive(apid), "alien survives unload of a row holding its pid")
            _seed_stale_row(home, gpu, "inst-ghost-stop", apid, free_port(), "STARTING")
            rr_stop = run_goethe([str(goethe), "down"], env=env, timeout=60)
            check("goethed stopped" in rr_stop.stdout, "goethe down")
            md.popen.wait(timeout=15)
            md.stop()
            check(pid_alive(apid), "alien survives goethe down")
            _seed_stale_row(home, gpu, "inst-ghost-adopt", apid, httpd.server_port, "RUNNING")
            orphan = home / "run" / "inst-ghost-orphan.json"
            orphan.parent.mkdir(exist_ok=True)
            entry = {
                "instance_id": "inst-ghost-orphan",
                "pid": apid,
                "boot_id": _boot_id(),
                "start_ticks": start_ticks(apid) + 7919,
            }
            orphan.write_text(json.dumps(entry), encoding="utf-8")
            md = start_goethed(home)
            check(not _ghost_rows(home), f"stale row must not be adopted: {_ghost_rows(home)}")
            check(not orphan.exists(), "stale run entry dropped")
            check(pid_alive(apid), "alien survives reconcile and orphan reap")
            md.stop()
            md = None
            assert_no_stub_engines(home)
        finally:
            httpd.shutdown()
            if md is not None:
                md.stop()
            alien.stop()
        check(not pid_alive(apid), "own alien stopped")
    print("stage4_alien_safety_stub ok")


if __name__ == "__main__":
    main_exit(main)
