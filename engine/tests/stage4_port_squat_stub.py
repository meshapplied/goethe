#!/usr/bin/env python3
"""Stage 4: an HTTP 200 squatter on a new instance's port must never make that instance RUNNING (F1)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lib.goethe_test import (
    free_port,
    gpu_flock,
    home_goethe_env,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    stub_launch,
    temp_home,
    wait_until,
    write_card,
)

MODEL = "sq-model"


class _Squatter(BaseHTTPRequestHandler):
    """Answers 200 to everything, even listing our served name, without being our engine."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _ok(self) -> None:
        """Reply 200 with a model list that names the instance's model."""
        body = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._ok()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        self._ok()


def _engine_range() -> tuple[int, int]:
    """Three consecutive free localhost ports for GOETHE_ENGINE_PORTS."""
    for _ in range(50):
        lo = free_port()
        if lo < 65000 and all(_free(p) for p in (lo, lo + 1, lo + 2)):
            return lo, lo + 2
    raise RuntimeError("no three consecutive free ports")


def _free(port: int) -> bool:
    """True when 127.0.0.1:port can be bound right now."""
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> None:
    """Squat the port of a delayed-start engine between port choice and bind; RUNNING must only follow our engine."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu()
    lo, hi = _engine_range()
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="squat") as home:
        model_dir = home / "models" / "org__sq"
        write_card(model_dir, MODEL, 512, 2, env={"STUB_START_DELAY": "4"})
        seed_installed_model(home, MODEL, "models/org__sq", 1.0, 512, 512, 2, 1)
        md = start_goethed(home, env={"GOETHE_ENGINE_PORTS": f"{lo}-{hi}"})
        env = {**os.environ, **home_goethe_env(home)}
        load = subprocess.Popen(
            [str(goethe), "load", MODEL, "--gpus", str(gpu), "--yes"],
            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        squatter: ThreadingHTTPServer | None = None
        seen: list[tuple[str, int]] = []
        try:
            q = "SELECT internal_port FROM loaded_instances WHERE model_id=?1 ORDER BY rowid DESC LIMIT 1"
            if not wait_until(lambda: bool(sqlite_query(home / "state.db", q, (MODEL,))), timeout=20, interval=0.05):
                raise RuntimeError("no instance row appeared")
            port = int(sqlite_query(home / "state.db", q, (MODEL,))[0][0])
            squatter = ThreadingHTTPServer(("127.0.0.1", port), _Squatter)
            threading.Thread(target=squatter.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
            deadline = time.time() + 90
            while load.poll() is None and time.time() < deadline:
                rows = sqlite_query(home / "state.db", "SELECT status, internal_port FROM loaded_instances")
                seen += [(str(s), int(p)) for s, p in rows]
                time.sleep(0.2)
            if load.poll() is None:
                load.kill()
                raise RuntimeError("load neither failed nor became RUNNING within 90 s")
            out, err = load.communicate(timeout=10)
            if any(s == "RUNNING" and p == port for s, p in seen):
                raise RuntimeError(f"instance went RUNNING on {port} while only the squatter answered: {seen[-5:]}")
            if load.returncode == 0:
                final = sqlite_query(home / "state.db", "SELECT status, internal_port FROM loaded_instances")
                launch = stub_launch(model_dir)
                if final != [("RUNNING", int(launch["parsed"]["port"]))] or int(launch["parsed"]["port"]) == port:
                    raise RuntimeError(f"RUNNING must mean our engine on another port: {final}, stub port {launch}")
            elif str(port) not in err + out and "port" not in (err + out).lower():
                raise RuntimeError(f"a failed load must name the port or its holder: {(err + out)[-300:]}")
        finally:
            if load.poll() is None:
                load.kill()
                load.wait(timeout=10)
            if squatter is not None:
                squatter.shutdown()
                squatter.server_close()
            subprocess.run([str(goethe), "unload", "--all"], env=env, stdin=subprocess.DEVNULL, capture_output=True)
            md.stop()
    print(f"stage4_port_squat_stub ok rc={load.returncode} squatted={port}")


if __name__ == "__main__":
    main_exit(main)
