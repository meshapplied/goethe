"""goethe api configure on a VPN interface in a throwaway network namespace: choice, allowlist, following."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.setdefault("GOETHE_HOME", "/tmp/goethe-net-home"))
PORT = 8443
FAILURES: list[str] = []


def check(cond: bool, what: str) -> None:
    """Print one result and collect the failures."""
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILURES.append(what)


def sh(cmd: str) -> None:
    """Run a setup command; any failure stops the test."""
    subprocess.run(cmd, shell=True, check=True)


def goethe(*args: str, ok: bool = True) -> tuple[int, str]:
    """Run the CLI and return its exit code and combined output."""
    r = subprocess.run(["goethe", *args], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    if ok and r.returncode != 0:
        print(out)
        raise SystemExit(f"goethe {' '.join(args)} failed")
    return r.returncode, out


def listening(ip: str) -> bool:
    """A TCP connection to the gateway port on this address succeeds."""
    try:
        with socket.create_connection((ip, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for(pred, timeout: float = 10.0) -> float | None:
    """Seconds until pred() holds, or None at the timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if pred():
            return time.monotonic() - start
        time.sleep(0.2)
    return None


def api(ip: str, key: str, source: str | None = None) -> str:
    """HTTP status of GET /v1/models on the gateway at ip, optionally from a given source address."""
    cmd = [
        "curl",
        "-sk",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "--max-time",
        "5",
        "-H",
        f"Authorization: Bearer {key}",
    ]
    if source:
        cmd += ["--interface", source]
    cmd.append(f"https://{ip}:{PORT}/v1/models")
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def gateway() -> dict:
    """goethe gateway --json."""
    return json.loads(goethe("gateway", "--json")[1])


def doctor_network() -> str:
    """The doctor's network check as 'STATUS hint'."""
    rows = json.loads(goethe("doctor", "--json", ok=False)[1])
    rows = rows.get("checks", rows) if isinstance(rows, dict) else rows
    row = next((r for r in rows if r.get("name") == "network"), {})
    return f"{row.get('status')} {row.get('hint')}"


def gateway_log() -> str:
    path = HOME / "logs" / "gateway.log"
    return path.read_text() if path.is_file() else ""


def master_pid() -> str:
    path = HOME / "nginx" / "nginx.pid"
    return path.read_text().strip() if path.is_file() else ""


def tun(name: str, cidr: str) -> None:
    sh(f"ip tuntap add dev {name} mode tun && ip addr add {cidr} dev {name} && ip link set {name} up")


def start_goethed() -> subprocess.Popen:
    """goethed in its own process group, up when the gateway answers on loopback."""
    log = open(HOME / "goethed.out", "a")
    proc = subprocess.Popen(["goethed"], stdout=log, stderr=log, start_new_session=True)
    if wait_for(lambda: listening("127.0.0.1"), 30) is None:
        raise SystemExit("goethed did not bring the gateway up")
    return proc


def stop_goethed(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)


def main() -> int:
    """Run every step; the namespace and home are thrown away with the container."""
    HOME.mkdir(parents=True, exist_ok=True)
    tun("vpn0", "10.77.0.2/24")
    tun("ts0", "100.88.1.2/32")
    sh("ip route add 100.90.0.5 dev ts0 table 52")
    sh("ip link add lan0 type veth peer name lan0p && ip addr add 192.168.60.2/24 dev lan0")
    sh("ip link set lan0 up && ip link set lan0p up")
    proc = start_goethed()

    code, out = goethe("api", "configure", "--mode", "vpn", "--yes", ok=False)
    check(
        code != 0 and "several networks" in out and "--interface" in out,
        "without a terminal, several networks need --interface",
    )

    code, out = goethe("api", "configure", "--mode", "vpn", "--interface", "ts0", "--yes")
    key = re.search(r"API key \(default\): (\S+)", out)
    check(key is not None, "a first key is created")
    key = key.group(1) if key else ""
    check("allow 100.64.0.0/10" in out, "per-peer routes widen to their private block (100.64.0.0/10)")
    check(wait_for(lambda: listening("100.88.1.2"), 5) is not None, "the gateway listens on ts0's address")
    goethe("allowlist", "--remove", "100.64.0.0/10")

    code, out = goethe("api", "configure", "--mode", "vpn", "--interface", "vpn0", "--yes")
    check(
        "10.77.0.2 (vpn0; follows its address)" in out and "allow 10.77.0.0/24" in out, "vpn0 is chosen with its subnet"
    )
    g = gateway()
    check(
        g.get("interface") == "vpn0" and g.get("bind") == "10.77.0.2" and not g.get("waiting"),
        f"gateway --json reports the interface: {g}",
    )
    check(listening("10.77.0.2") and listening("127.0.0.1"), "listening on vpn0 and loopback")
    check(not listening("100.88.1.2") and not listening("192.168.60.2"), "not on other networks")
    check(api("10.77.0.2", key) == "200", "API answers on the VPN address")
    check(api("10.77.0.2", key, source="192.168.60.2") == "403", "a client outside the allowed range is refused")

    pid = master_pid()
    sh("ip addr del 10.77.0.2/24 dev vpn0 && ip addr add 10.77.0.9/24 dev vpn0")
    took = wait_for(lambda: listening("10.77.0.9"), 10)
    check(took is not None, f"an address change is followed (in {took and round(took, 1)} s)")
    check(api("10.77.0.9", key) == "200", "API answers on the new address")
    check("vpn0 moved from 10.77.0.2 to 10.77.0.9" in gateway_log(), "the move is logged")
    check(master_pid() == pid, "the move was a hot reload (same nginx master)")

    sh("ip addr flush dev vpn0")
    took = wait_for(lambda: "vpn0 has no IPv4 address" in gateway_log(), 10)
    check(
        took is not None and gateway().get("waiting") is True,
        f"losing the address moves the gateway to loopback (in {took and round(took, 1)} s)",
    )
    loopback_only = wait_for(lambda: (HOME / "nginx" / "listen_ip").read_text().strip() == "127.0.0.1", 5)
    check(loopback_only is not None, "nginx now listens on loopback only")
    check(api("127.0.0.1", key) == "200", "local clients still work while waiting")
    check("waiting for vpn0" in goethe("status")[1], "goethe status says it is waiting for vpn0")
    check(doctor_network().startswith("WARN vpn0 has no IPv4 address"), f"doctor warns: {doctor_network()}")

    sh("ip link del vpn0")
    tun("vpn0", "10.77.0.3/24")
    took = wait_for(lambda: listening("10.77.0.3"), 10)
    check(took is not None, f"a recreated interface is followed (in {took and round(took, 1)} s)")
    check("vpn0 is up at 10.77.0.3" in gateway_log(), "the return is logged")

    sh("ip link del vpn0")
    stop_goethed(proc)
    proc = start_goethed()
    check(
        gateway().get("waiting") is True and listening("127.0.0.1"), "goethed starts while vpn0 is missing, local only"
    )
    tun("vpn0", "10.77.0.4/24")
    check(wait_for(lambda: listening("10.77.0.4"), 10) is not None, "and follows vpn0 once it appears")
    check(api("10.77.0.4", key) == "200", "API answers there")
    check(doctor_network().startswith("OK gateway follows vpn0 at 10.77.0.4"), f"doctor: {doctor_network()}")

    goethe("allowlist", "--remove", "10.77.0.0/24")
    check(
        "no allowlist entry covers" in doctor_network(), "doctor spots an allowlist that refuses the network's clients"
    )

    code, out = goethe("api", "configure", "--mode", "vpn", "--interface", "nope0", "--yes", ok=False)
    check(code != 0 and "has no IPv4 address" in out, "an unknown interface is refused")
    code, out = goethe(
        "api", "configure", "--mode", "vpn", "--interface", "vpn0", "--allow", "0.0.0.0/0", "--yes", ok=False
    )
    check(code != 0 and "lets every address in" in out, "0.0.0.0/0 is refused in VPN mode")
    check(gateway().get("interface") == "vpn0", "a refused change leaves the setting as it was")

    goethe("gateway", "--bind", "127.0.0.1")
    g = gateway()
    check(g.get("interface") is None and g.get("bind") == "127.0.0.1", "a fixed address replaces the interface")
    goethe("gateway", "--bind", "vpn0")
    check(
        gateway().get("interface") == "vpn0" and listening("10.77.0.4"), "goethe gateway --bind takes an interface too"
    )

    stop_goethed(proc)
    print(f"\n{len(FAILURES)} failures" if FAILURES else "\nall checks passed")
    for f in FAILURES:
        print("  -", f)
    if FAILURES:
        print("\ngateway.log:\n" + gateway_log())
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
