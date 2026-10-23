#!/usr/bin/env python3
"""Stage 5: `goethe gateway bind` re-renders nginx to listen on 0.0.0.0 and back on loopback (F5)."""

from __future__ import annotations

import re
import socket
from pathlib import Path

from lib.goethe_test import (
    RunResult,
    ensure_nginx,
    home_goethe_env,
    home_process_tree,
    http_get,
    main_exit,
    make_tls_certs,
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


def _bind(goethe: str, env: dict[str, str], addr: str) -> RunResult:
    """goethe gateway bind addr, confirming with --yes only when the CLI asks for it."""
    rr = run([goethe, "gateway", "bind", addr], env=env)
    if rr.returncode != 0 and "--yes" in rr.stdout + rr.stderr:
        rr = run([goethe, "gateway", "bind", addr, "--yes"], env=env)
    return rr


def _answers(url: str) -> bool:
    return http_get(url, timeout=3)[0] != 0


def main() -> None:
    """Loopback by default; bind 0.0.0.0 answers on the host address; bind 127.0.0.1 closes it again."""
    goethe = str(require_cli("goethe"))
    with temp_home(stub_python=False, tag="bind") as home:
        ensure_nginx(home)
        make_tls_certs(home)
        md = start_goethed(home)
        env = home_goethe_env(home)
        port = test_public_port(home)
        conf = home / "nginx" / "goethe_gateway.conf"
        loop = f"https://127.0.0.1:{port}/health"
        ip = _host_ip()
        outside = f"https://{ip}:{port}/health" if ip else ""
        try:
            if not wait_until(lambda: _answers(loop), timeout=20):
                raise RuntimeError(f"gateway did not answer on loopback: {_listen(conf)}")
            if _listen(conf) != [f"127.0.0.1:{port}"] or (ip and _answers(outside)):
                raise RuntimeError(f"the default gateway must listen on loopback only: {_listen(conf)}")
            rr = _bind(goethe, env, "0.0.0.0")
            if rr.returncode != 0:
                raise RuntimeError(f"gateway bind 0.0.0.0 failed (exit {rr.returncode}): {rr.stderr[-300:]}")
            if not wait_until(lambda: _listen(conf) == [f"0.0.0.0:{port}"], timeout=15):
                raise RuntimeError(f"bind 0.0.0.0 did not re-render the listen address: {_listen(conf)}")
            if ip and not wait_until(lambda: _answers(outside), timeout=15):
                raise RuntimeError(f"after bind 0.0.0.0 nothing answers on {outside}")
            status = run([goethe, "gateway", "status"], env=env)
            if status.returncode != 0 or "0.0.0.0" not in status.stdout:
                raise RuntimeError(f"gateway status must show the 0.0.0.0 bind: {status.stdout[-300:]!r}")
            rr = _bind(goethe, env, "127.0.0.1")
            if rr.returncode != 0 or not wait_until(lambda: _listen(conf) == [f"127.0.0.1:{port}"], timeout=15):
                raise RuntimeError(f"bind 127.0.0.1 did not revert (exit {rr.returncode}): {_listen(conf)}")
            if ip and not wait_until(lambda: not _answers(outside), timeout=15):
                raise RuntimeError(f"after bind 127.0.0.1 the gateway still answers on {outside}")
            if not _answers(loop):
                raise RuntimeError("after bind 127.0.0.1 the gateway stopped answering on loopback")
            status = run([goethe, "gateway", "status"], env=env)
            if status.returncode != 0 or "127.0.0.1" not in status.stdout:
                raise RuntimeError(f"gateway status must show the loopback bind: {status.stdout[-300:]!r}")
        finally:
            if _listen(conf) != [f"127.0.0.1:{port}"]:
                _bind(goethe, env, "127.0.0.1")
            md.stop()
        if not wait_until(lambda: not home_process_tree(home), timeout=10):
            raise RuntimeError(f"processes left after goethed stopped: {sorted(home_process_tree(home))}")
    print(f"stage5_gateway_bind ok host_ip={ip or 'none'}")


if __name__ == "__main__":
    main_exit(main)
