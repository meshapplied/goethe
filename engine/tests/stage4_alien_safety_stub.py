#!/usr/bin/env python3
"""Stage 4: goethe never signals processes it did not start and reports them with commands the user can run."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lib.goethe_test import (
    Proc,
    assert_no_stub_engines,
    foreign_gpu_process,
    free_port,
    gpu_pids,
    home_goethe_env,
    main_exit,
    require_cli,
    run,
    run_goethe,
    seed_installed_model,
    spawn,
    sqlite_query,
    temp_home,
    wait_goethed_socket,
    wait_until,
)

ALIEN_GPU = 3
DEV_PY = Path(os.environ.get("GOETHE_DEV_HOME", "/scratch/ternary-flash/goethe-home")) / "py-env" / "bin" / "python"


def check(cond: bool, msg: str) -> None:
    """Fail the gate with msg when cond is false."""
    if not cond:
        raise RuntimeError(msg)


def _alive(pid: int) -> bool:
    """Return true while pid exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_ticks(pid: int) -> int:
    """Start time of pid in clock ticks (field 22 of /proc/<pid>/stat)."""
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(stat[stat.rindex(")") + 2 :].split()[19])


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


def _start_goethed(home: Path) -> Proc:
    """Start goethed for the temp home and wait for its socket."""
    proc = spawn([str(require_cli("goethed"))], env=home_goethe_env(home), label="goethed")
    wait_goethed_socket(home)
    return proc


def _seed_stale_row(home: Path, instance_id: str, pid: int, port: int, status: str) -> None:
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
                str(ALIEN_GPU),
                status,
                f"ghost|{ALIEN_GPU}|0",
                _boot_id(),
                _start_ticks(pid) + 7919,
            ),
        )
        con.execute(
            "INSERT OR REPLACE INTO desired_loads (slot_key, model_id) VALUES (?, 'ghost')", (f"ghost|{ALIEN_GPU}|0",)
        )
        con.commit()
    finally:
        con.close()


def _ghost_rows(home: Path) -> list[tuple[object, ...]]:
    """Instance rows (id, status, pid) of the seeded ghost model."""
    return sqlite_query(
        home / "state.db", "SELECT instance_id, status, pid FROM loaded_instances WHERE model_id='ghost'"
    )


def _rpc(home: Path, method: str, params: dict[str, object]) -> dict[str, object]:
    """Send one JSON-lines request to goethe.sock and return the parsed reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(60)
        sock.connect(str(home / "goethe.sock"))
        sock.sendall((json.dumps({"method": method, "params": params}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def main() -> None:
    """Alien VRAM holder on GPU 3: exit 3 notice, --yes load, and unload/stop/watchdog/restart never touch it."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    with temp_home(stub_python=True, real_python=DEV_PY if DEV_PY.is_file() else None, tag="alien") as home:
        model = home / "models" / "org__alien_stub"
        model.mkdir(parents=True)
        (model / "goethe_card.toml").write_text(
            'model_id="alien-stub"\ncontext_length=4096\nmax_running_requests=4\n', encoding="utf-8"
        )
        alien = foreign_gpu_process(ALIEN_GPU, mb=512)
        apid = alien.popen.pid
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysOk)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        md: Proc | None = None
        try:
            check(wait_until(lambda: any(p == apid for p, _ in gpu_pids(ALIEN_GPU)), timeout=90.0), "alien on GPU 3")
            md = _start_goethed(home)
            seed_installed_model(home, "alien-stub", str(model.relative_to(home)), 1.0, 1000, 4096, 4, 1)
            env = home_goethe_env(home)
            rr = run([str(goethe), "load", "alien-stub", "--gpus", str(ALIEN_GPU)], env=env, input_text="")
            text = rr.stdout + rr.stderr
            check(rr.returncode == 3, f"load without --yes must exit 3, got {rr.returncode}: {text[-400:]}")
            check(f"unmanaged process PID {apid}" in text, f"notice names the alien pid: {text[-400:]}")
            check(f"kill {apid}" in text and "Re-run with --yes" in text, "notice suggests kill and --yes")
            reply = _rpc(home, "load", {"model_id": "alien-stub", "gpus": str(ALIEN_GPU)})
            foreign = reply.get("result", {}).get("foreign", []) if isinstance(reply.get("result"), dict) else []
            check(reply.get("code") == 3 and any(f.get("pid") == apid for f in foreign), f"rpc foreign: {reply}")
            check(_alive(apid), "alien alive after the notice")
            run_goethe([str(goethe), "load", "alien-stub", "--gpus", str(ALIEN_GPU), "--yes"], env=env, timeout=120)
            run_goethe([str(goethe), "unload", "alien-stub"], env=env, timeout=60)
            check(_alive(apid), "alien survives unload")
            _seed_stale_row(home, "inst-ghost-watchdog", apid, free_port(), "RUNNING")
            tomb = [("inst-ghost-watchdog", "FAILED", 0)]
            check(wait_until(lambda: _ghost_rows(home) == tomb, timeout=30.0), "watchdog tombstones an unverified row")
            check(_alive(apid), "alien survives the watchdog")
            _seed_stale_row(home, "inst-ghost-unload", apid, free_port(), "STARTING")
            run_goethe([str(goethe), "unload", "ghost"], env=env, timeout=60)
            check(not _ghost_rows(home) and _alive(apid), "alien survives unload of a row holding its pid")
            _seed_stale_row(home, "inst-ghost-stop", apid, free_port(), "STARTING")
            rr_stop = run_goethe([str(goethe), "stop"], env=env, timeout=60)
            check("goethed stopped" in rr_stop.stdout, "goethe stop")
            md.popen.wait(timeout=15)
            md._stopped = True
            check(_alive(apid), "alien survives goethe stop")
            _seed_stale_row(home, "inst-ghost-adopt", apid, httpd.server_port, "RUNNING")
            orphan = home / "run" / "inst-ghost-orphan.json"
            orphan.parent.mkdir(exist_ok=True)
            entry = {
                "instance_id": "inst-ghost-orphan",
                "pid": apid,
                "boot_id": _boot_id(),
                "start_ticks": _start_ticks(apid) + 7919,
            }
            orphan.write_text(json.dumps(entry), encoding="utf-8")
            md = _start_goethed(home)
            check(not _ghost_rows(home), f"stale row must not be adopted: {_ghost_rows(home)}")
            check(not orphan.exists(), "stale run entry dropped")
            check(_alive(apid), "alien survives reconcile and orphan reap")
            md.stop()
            md = None
            assert_no_stub_engines(home)
        finally:
            httpd.shutdown()
            if md is not None:
                md.stop()
            alien.stop()
        check(not _alive(apid), "own alien stopped")
    print("stage4_alien_safety_stub ok")


if __name__ == "__main__":
    main_exit(main)
