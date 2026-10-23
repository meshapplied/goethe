#!/usr/bin/env python3
"""Stage 5: `goethe gateway bind` moves the gateway listener: wildcard changes restart nginx, others reload (F5)."""

from __future__ import annotations

import re
import socket
import time
from pathlib import Path

from lib.goethe_test import (
    ensure_nginx,
    home_goethe_env,
    home_process_tree,
    http_get,
    main_exit,
    make_tls_certs,
    proc_cmdline,
    require_cli,
    run,
    start_goethed,
    temp_home,
    test_public_port,
    wait_until,
)


def _host_ip() -> str | None:
    """A non-loopback IPv4 address of this host; the UDP connect only picks a route and sends nothing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("10.254.254.254", 1))
            ip = sock.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


def _listen(conf: Path) -> list[str]:
    """The listen addresses of the rendered gateway include."""
    if not conf.is_file():
        return []
    return re.findall(r"^\s*listen\s+(\S+)", conf.read_text(encoding="utf-8"), flags=re.M)


def _master(home: Path) -> int:
    """PID of this home's nginx master from its pid file (0 when absent or not ours)."""
    try:
        pid = int((home / "nginx" / "nginx.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0
    return pid if str(home) in proc_cmdline(pid) else 0


def _answers(addr: str, port: int) -> bool:
    return http_get(f"https://{addr}:{port}/health", timeout=3)[0] != 0


class Bind:
    """Runs goethe gateway bind/status for one home and checks the listener, the master and who answers."""

    def __init__(self, goethe: str, home: Path) -> None:
        self.goethe, self.home, self.env = goethe, home, home_goethe_env(home)
        self.port = test_public_port(home)
        self.conf = home / "nginx" / "goethe_gateway.conf"

    def move(self, addr: str, restart: bool) -> float:
        """Bind addr; a wildcard change must restart the master, others must hot-reload it; returns answer delay."""
        before = _master(self.home)
        t0 = time.time()
        rr = run([self.goethe, "gateway", "bind", addr], env=self.env, timeout=60)
        if rr.returncode != 0:
            raise RuntimeError(f"gateway bind {addr} failed (exit {rr.returncode}): {rr.stderr[-300:]}")
        if not wait_until(lambda: self.listen() == [f"{addr}:{self.port}"], timeout=15):
            raise RuntimeError(f"bind {addr} did not re-render the listen address: {self.listen()}")
        probe = "127.0.0.1" if addr == "0.0.0.0" else addr
        if not wait_until(lambda: _answers(probe, self.port), timeout=10, interval=0.2):
            raise RuntimeError(f"after bind {addr} the gateway did not answer on {probe} within 10 s")
        delay = time.time() - t0
        after = _master(self.home)
        if restart != ("(restarted)" in rr.stdout) or restart != (after != before) or not after:
            want = "restart" if restart else "hot reload"
            raise RuntimeError(f"bind {addr} must {want} nginx: master {before} -> {after}, said {rr.stdout!r}")
        return delay

    def listen(self) -> list[str]:
        return _listen(self.conf)

    def status_shows(self, addr: str) -> None:
        status = run([self.goethe, "gateway", "status"], env=self.env)
        if status.returncode != 0 or addr not in status.stdout:
            raise RuntimeError(f"gateway status must show the {addr} bind: {status.stdout[-300:]!r}")


def main() -> None:
    """Loopback by default; 0.0.0.0 restarts and answers on the host address; loopback moves hot-reload."""
    goethe = str(require_cli("goethe"))
    with temp_home(stub_python=False, tag="bind") as home:
        ensure_nginx(home)
        make_tls_certs(home)
        md = start_goethed(home)
        gw = Bind(goethe, home)
        ip = _host_ip()
        delays: list[float] = []
        try:
            if not wait_until(lambda: _answers("127.0.0.1", gw.port), timeout=20):
                raise RuntimeError(f"gateway did not answer on loopback: {gw.listen()}")
            if gw.listen() != [f"127.0.0.1:{gw.port}"] or (ip and _answers(ip, gw.port)):
                raise RuntimeError(f"the default gateway must listen on loopback only: {gw.listen()}")
            delays.append(gw.move("0.0.0.0", restart=True))
            if ip and not wait_until(lambda: _answers(ip, gw.port), timeout=10):
                raise RuntimeError(f"after bind 0.0.0.0 nothing answers on {ip}:{gw.port}")
            gw.status_shows("0.0.0.0")
            delays.append(gw.move("127.0.0.1", restart=True))
            if ip and not wait_until(lambda: not _answers(ip, gw.port), timeout=10):
                raise RuntimeError(f"after bind 127.0.0.1 the gateway still answers on {ip}:{gw.port}")
            delays.append(gw.move("127.0.0.2", restart=False))
            if not wait_until(lambda: not _answers("127.0.0.1", gw.port), timeout=10):
                raise RuntimeError("after bind 127.0.0.2 the gateway still answers on 127.0.0.1")
            delays.append(gw.move("127.0.0.1", restart=False))
            refused = run([goethe, "gateway", "bind", "192.0.2.1"], env=gw.env)
            if refused.returncode == 0 or "not an address of this host" not in refused.stderr + refused.stdout:
                raise RuntimeError(f"a non-local bind must be refused: exit {refused.returncode} {refused.stderr!r}")
            if gw.listen() != [f"127.0.0.1:{gw.port}"] or not _answers("127.0.0.1", gw.port):
                raise RuntimeError(f"a refused bind must leave the gateway on loopback: {gw.listen()}")
            gw.status_shows("127.0.0.1")
        finally:
            if gw.listen() != [f"127.0.0.1:{gw.port}"]:
                run([goethe, "gateway", "bind", "127.0.0.1"], env=gw.env, timeout=60)
            md.stop()
        if not wait_until(lambda: not home_process_tree(home), timeout=10):
            raise RuntimeError(f"processes left after goethed stopped: {sorted(home_process_tree(home))}")
    print(f"stage5_gateway_bind ok host_ip={ip or 'none'} answer_s={','.join(f'{d:.1f}' for d in delays)}")


if __name__ == "__main__":
    main_exit(main)
