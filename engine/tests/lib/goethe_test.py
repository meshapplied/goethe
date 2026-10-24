"""Stdlib helpers for goethe stage-gate black-box tests."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
import shutil
import signal
import socket
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
LIB_DIR = Path(__file__).resolve().parent
TESTS_DIR = LIB_DIR.parent
GOETHE_PKG = Path(os.environ.get("GOETHE_PKG", str(TESTS_DIR.parent)))
STUB_ENGINE = LIB_DIR / "stub_engine.py"
DEFAULT_DEV_HOME = Path("/scratch/ternary-flash/goethe-home")

_gpu_pid_query: Callable[[int], list[tuple[int, int]]] | None = None
_spawned_pids: set[int] = set()
_gate_roots: set[int] = set()
_group_starts: dict[int, int | None] = {}
_gpu_locks_held: set[int] = set()


def gate_run_id() -> str:
    """Return the current gate run id for temp and hygiene scoping."""
    return os.environ.get("GOETHE_GATE_RUN_ID", str(os.getpid()))


def run_temp_prefix(tag: str) -> str:
    """Build a /tmp name prefix goethe-test-<run_id>-<tag>-."""
    return f"goethe-test-{gate_run_id()}-{tag}-"


def run_temp_marker() -> str:
    """Substring that identifies temp paths and cmdlines for this gate run."""
    return f"goethe-test-{gate_run_id()}-"


class Skip(Exception):
    """Skip this test with a reason."""


class Pending(Exception):
    """Product binary or subcommand missing; gate.sh reports it as a failure."""


@dataclass
class RunResult:
    """Captured subprocess result."""

    returncode: int
    stdout: str
    stderr: str


def _stat(pid: int) -> tuple[str, int, int, int] | None:
    """Return (state, ppid, pgid, start_ticks) from /proc/<pid>/stat, or None when gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rest = raw[raw.rfind(")") + 2 :].split()
    return rest[0], int(rest[1]), int(rest[2]), int(rest[19])


def start_ticks(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat: process start time in clock ticks since boot."""
    st = _stat(pid)
    return st[3] if st else None


def process_table() -> dict[int, tuple[str, int, int, int]]:
    """Snapshot (state, ppid, pgid, start_ticks) for every visible process."""
    table: dict[int, tuple[str, int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            st = _stat(int(entry.name))
            if st is not None:
                table[int(entry.name)] = st
    return table


def _proc_bytes(pid: int, name: str) -> bytes:
    """Read /proc/<pid>/<name>, empty when unreadable."""
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except OSError:
        return b""


def _group_members(pgid: int) -> list[int]:
    """Live (non-zombie) processes in a process group."""
    return sorted(pid for pid, st in process_table().items() if st[2] == pgid and st[0] != "Z")


def pid_alive(pid: int) -> bool:
    """Return true when pid exists and is not a zombie."""
    st = _stat(pid)
    return st is not None and st[0] != "Z"


_pid_alive = pid_alive
_process_alive = pid_alive


@dataclass
class Proc:
    """A process group started with setsid for clean teardown."""

    popen: subprocess.Popen[Any]
    pgid: int
    label: str
    _stopped: bool = False
    _gpu_flock_release: Callable[[], None] | None = None
    term_timeout: float = 3.0

    def _group_alive(self) -> bool:
        """True while our group has live members; never true for a recycled group id."""
        if self.popen.poll() is not None and Path(f"/proc/{self.pgid}").exists():
            return False
        return bool(_group_members(self.pgid))

    def _signal(self, sig: int) -> None:
        """Signal our group unless the leader was reaped and its id now names another process."""
        if self.popen.poll() is not None and Path(f"/proc/{self.pgid}").exists():
            return
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def stop(self, sigterm_timeout: float | None = None) -> None:
        """Send SIGTERM to the group, then SIGKILL to whatever is left after the grace period."""
        if self._stopped:
            return
        self._stopped = True
        try:
            if self._group_alive():
                self._signal(signal.SIGTERM)
                grace = self.term_timeout if sigterm_timeout is None else sigterm_timeout
                if not wait_until(lambda: not self._group_alive(), timeout=grace, interval=0.05):
                    self._signal(signal.SIGKILL)
                    wait_until(lambda: not self._group_alive(), timeout=5.0, interval=0.05)
            try:
                self.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.popen.kill()
        finally:
            _gate_roots.discard(self.pgid)
            _group_starts.pop(self.pgid, None)
            unregister_pid(self.popen.pid)
            if self in _ACTIVE:
                _ACTIVE.remove(self)
            if self._gpu_flock_release is not None:
                self._gpu_flock_release()
                self._gpu_flock_release = None

    def __enter__(self) -> Proc:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


_ACTIVE: list[Proc] = []
_temp_homes_active: set[Path] = set()
_signal_handlers_installed = False
_cleaning_up = False


def register_pid(pid: int) -> None:
    """Track a PID spawned by the current test."""
    _spawned_pids.add(pid)
    _write_artifact()


def unregister_pid(pid: int) -> None:
    """Stop tracking a PID."""
    _spawned_pids.discard(pid)
    _write_artifact()


def _artifact_path() -> Path:
    return Path(f"/tmp/{run_temp_prefix('artifacts')}{os.getpid()}.json")


def _write_artifact() -> None:
    """Record test-owned pids and groups (with leader start ticks) for gate.sh reaping."""
    path = _artifact_path()
    if not _spawned_pids and not _gate_roots:
        path.unlink(missing_ok=True)
        return
    groups = [{"pgid": g, "start": _group_starts.get(g)} for g in sorted(_gate_roots)]
    path.write_text(
        json.dumps({"pids": sorted(_spawned_pids), "pgids": sorted(_gate_roots), "groups": groups}),
        encoding="utf-8",
    )


def _atexit_cleanup() -> None:
    """Stop registered process groups and remove active temp homes."""
    global _cleaning_up
    if _cleaning_up:
        return
    _cleaning_up = True
    for proc in list(_ACTIVE):
        proc.stop()
    for td in list(_temp_homes_active):
        _shutdown_home_daemon(td)
        shutil.rmtree(td, ignore_errors=True)
    _temp_homes_active.clear()
    _artifact_path().unlink(missing_ok=True)
    _cleaning_up = False


def _signal_teardown(signum: int, _frame: object) -> None:
    """Run full test teardown on SIGTERM or SIGINT then exit."""
    _atexit_cleanup()
    code = 128 + signum if signum in (signal.SIGTERM, signal.SIGINT) else 1
    raise SystemExit(code)


def install_test_signal_handlers() -> None:
    """Install SIGTERM and SIGINT handlers for test process teardown."""
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return
    signal.signal(signal.SIGTERM, _signal_teardown)
    signal.signal(signal.SIGINT, _signal_teardown)
    _signal_handlers_installed = True


def cleanup_test_procs() -> None:
    """Stop any Proc instances registered by the current test process."""
    _atexit_cleanup()


atexit.register(_atexit_cleanup)


def set_gpu_pid_query(fn: Callable[[int], list[tuple[int, int]]] | None) -> None:
    """Override nvidia-smi PID query for selftests."""
    global _gpu_pid_query
    _gpu_pid_query = fn


def resolve_goethe_home() -> Path:
    """Resolve GOETHE_HOME using the product contract."""
    if os.environ.get("GOETHE_HOME"):
        return Path(os.environ["GOETHE_HOME"])
    workspace = Path("/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / ".goethe"
    varlib = Path("/var/lib/goethe")
    if varlib.is_dir() and os.access(varlib, os.W_OK):
        return varlib
    return Path.home() / ".goethe"


def dev_home() -> Path:
    """Read-only shared dev home (py-env, models) named by GOETHE_DEV_HOME."""
    return Path(os.environ.get("GOETHE_DEV_HOME", str(DEFAULT_DEV_HOME)))


def dev_python() -> Path:
    """Dev py-env interpreter with torch, SGLang and hf, passed by gate.sh as GOETHE_DEV_PY."""
    env = os.environ.get("GOETHE_DEV_PY")
    return Path(env) if env else dev_home() / "py-env" / "bin" / "python"


def resolve_python() -> Path:
    """Interpreter for test helpers: GOETHE_PYTHON, then the dev py-env, then this interpreter."""
    env = os.environ.get("GOETHE_PYTHON")
    if env:
        return Path(env)
    dev = dev_python()
    return dev if dev.is_file() else Path(sys.executable)


def goethe_bin(name: str) -> Path:
    """Path to goethe or goethed: GOETHE_BIN_DIR, then GOETHE_HOME/bin, then target/release."""
    bindir = os.environ.get("GOETHE_BIN_DIR")
    if bindir:
        return Path(bindir) / name
    cand = resolve_goethe_home() / "bin" / name
    if cand.is_file():
        return cand
    return GOETHE_PKG / "target" / "release" / name


def require_cli(name: str) -> Path:
    """Require goethe or goethed on disk."""
    path = goethe_bin(name)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise Pending(f"{name} binary not built at {path}")
    return path


def test_gpus() -> list[int]:
    """Physical GPU indices allowed for tests."""
    raw = os.environ.get("GOETHE_TEST_GPUS", "2,3")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _acquire_gpu_flocks(indices: Sequence[int]) -> Callable[[], None]:
    """Acquire /tmp/goethe-gpu locks in ascending index order."""
    gpus = sorted({int(i) for i in indices if int(i) not in _gpu_locks_held})
    acquired: list[int] = []
    fds: list[int] = []
    try:
        for idx in gpus:
            path = f"/tmp/goethe-gpu{idx}.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            deadline = time.time() + 1800.0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        os.close(fd)
                        raise TimeoutError(f"timed out waiting for GPU {idx} lock") from None
                    time.sleep(0.2)
            fds.append(fd)
            acquired.append(idx)
            _gpu_locks_held.add(idx)
    except Exception:
        for fd in reversed(fds):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        for idx in acquired:
            _gpu_locks_held.discard(idx)
        raise

    def release() -> None:
        for fd, idx in zip(reversed(fds), reversed(acquired), strict=True):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            _gpu_locks_held.discard(idx)

    return release


@contextmanager
def gpu_flock(indices: Sequence[int] | None = None) -> Iterator[None]:
    """Hold goethe GPU flock locks until the context exits."""
    release = _acquire_gpu_flocks(indices if indices is not None else test_gpus())
    try:
        yield
    finally:
        release()


def require_gpus(n: int) -> list[int]:
    """Require at least n test GPUs or skip."""
    gpus = test_gpus()
    if len(gpus) < n:
        raise Skip(f"need {n} GPUs in GOETHE_TEST_GPUS, have {len(gpus)}")
    return gpus


def run(
    cmd: Sequence[str],
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> RunResult:
    """Run a command with stdin closed (or fed input_text) and capture stdout/stderr."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        stdin=None if input_text is not None else subprocess.DEVNULL,
    )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def _goethe_error_snippet(rr: RunResult) -> str:
    """Last 800 chars of goethe stderr with warning noise removed."""
    raw = rr.stderr or rr.stdout or ""
    lines = [ln for ln in raw.splitlines() if "FutureWarning" not in ln and "warnings.warn" not in ln]
    text = "\n".join(lines).strip() or raw.strip()
    return text[-800:] if len(text) > 800 else text


def classify_goethe_result(rr: RunResult, binary_missing_ok: bool = False) -> None:
    """Map goethe CLI exit codes to Pending (unknown subcommand) or failure."""
    err = (rr.stderr or "").lower()
    if rr.returncode == 2 and ("unrecognized" in err or "unknown" in err):
        raise Pending(rr.stderr.strip() or "unknown subcommand")
    if rr.returncode != 0:
        raise RuntimeError(f"goethe exit {rr.returncode}: {_goethe_error_snippet(rr)}")


def run_goethe(
    cmd: Sequence[str],
    timeout: float = 180.0,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> RunResult:
    """Run goethe CLI and classify failures."""
    rr = run(cmd, timeout=timeout, env=env, input_text=input_text)
    classify_goethe_result(rr)
    return rr


def spawn(
    cmd: Sequence[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    label: str = "proc",
) -> Proc:
    """Start a command with stdin closed in its own session and record its group for reaping."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    pop = subprocess.Popen(
        list(cmd),
        env=merged,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pgid = pop.pid
    _group_starts[pgid] = start_ticks(pop.pid)
    _gate_roots.add(pgid)
    register_pid(pop.pid)
    proc = Proc(popen=pop, pgid=pgid, label=label)
    _ACTIVE.append(proc)
    return proc


def wait_until(pred: Callable[[], bool], timeout: float = 30.0, interval: float = 0.2) -> bool:
    """Poll until pred returns true or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def free_port() -> int:
    """Bind an ephemeral localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _nvidia_smi_query(args: list[str]) -> str:
    proc = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or "nvidia-smi failed")
    return proc.stdout


def gpu_mem_used_mb(index: int) -> int:
    """Used VRAM in MiB for a physical GPU index."""
    out = _nvidia_smi_query(["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={index}"])
    return int(out.strip().split("\n")[0].strip())


def gpu_mem_snapshot(indices: Sequence[int]) -> dict[int, int]:
    """Snapshot used VRAM for each GPU index."""
    return {i: gpu_mem_used_mb(i) for i in indices}


def gpu_pids(index: int) -> list[tuple[int, int]]:
    """Return (pid, used_mb) compute processes on a GPU."""
    if _gpu_pid_query is not None:
        return _gpu_pid_query(index)
    out = _nvidia_smi_query(
        ["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits", f"--id={index}"]
    )
    rows: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    return rows


def gpu_free_gb(index: int) -> float:
    """Free VRAM of a physical GPU in GiB, as the product's budget check sees it."""
    out = _nvidia_smi_query(["--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={index}"])
    return int(out.strip().split("\n")[0]) / 1024.0


def roomy_test_gpus(min_free_gb: float) -> list[int]:
    """GOETHE_TEST_GPUS indices with at least min_free_gb GiB free, most free first."""
    ranked = sorted(((gpu_free_gb(i), i) for i in test_gpus()), reverse=True)
    return [i for free, i in ranked if free >= min_free_gb]


def roomiest_test_gpu(min_free_gb: float = 15.0) -> int:
    """Test GPU with the most free VRAM for stub loads (budgets use live free memory); Skip when none has room."""
    roomy = roomy_test_gpus(min_free_gb)
    if not roomy:
        raise Skip(f"no GOETHE_TEST_GPUS index has {min_free_gb:.0f} GiB free for stub load budgets")
    return roomy[0]


def clean_test_gpu() -> int | None:
    """First GOETHE_TEST_GPUS index with no compute process on it, or None."""
    for idx in test_gpus():
        if not gpu_pids(idx):
            return idx
    return None


def wait_gpu_pid(index: int, pid: int, min_mb: int, timeout: float = 60.0) -> bool:
    """Wait until nvidia-smi lists pid on GPU index with at least min_mb MiB."""
    return wait_until(
        lambda: any(p == pid and mb >= min_mb for p, mb in gpu_pids(index)), timeout=timeout, interval=0.5
    )


def leaked_temp_paths() -> list[str]:
    """Return leftover temp paths for the current gate run only."""
    prefix = f"goethe-test-{gate_run_id()}-"
    paths: list[str] = []
    for p in Path("/tmp").glob(f"{prefix}*"):
        if not p.exists():
            continue
        if "-artifacts-" in p.name and p.suffix == ".json":
            continue
        if "-gate-" in p.name:
            continue
        paths.append(str(p))
    return sorted(paths)


def _allocate_mux_public_ports() -> tuple[int, int]:
    """Pick two distinct free localhost ports for mux and public nginx."""
    mux = free_port()
    public = free_port()
    while public == mux:
        public = free_port()
    return mux, public


def wait_goethed_socket(home: Path, timeout: float = 30.0) -> None:
    """Wait until goethed creates goethe.sock under GOETHE_HOME."""
    if not wait_until(lambda: (home / "goethe.sock").is_socket(), timeout=timeout):
        raise RuntimeError("goethed did not create goethe.sock")


def start_goethed(home: Path, env: dict[str, str] | None = None, label: str = "goethed") -> Proc:
    """Start goethed for a temp home, wait for its socket, and allow it a 20 s graceful stop."""
    proc = spawn([str(require_cli("goethed"))], env={**home_goethe_env(home), **(env or {})}, label=label)
    proc.term_timeout = 20.0
    wait_goethed_socket(home)
    return proc


def goethe_rpc(home: Path, method: str, params: Any, timeout: float = 60.0) -> dict[str, Any]:
    """Send one JSON-lines RPC to goethed under home and return the decoded reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(home / "goethe.sock"))
        sock.sendall((json.dumps({"method": method, "params": params}) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


def proc_cmdline(pid: int) -> str:
    return _proc_bytes(pid, "cmdline").replace(b"\0", b" ").decode("utf-8", errors="replace")


_cmdline = proc_cmdline


def kill_engine_group(home: Path, pgid: int, sig: int = signal.SIGKILL) -> None:
    """Signal an engine group goethed started for this temp home, after checking the leader belongs to it."""
    if str(home) not in proc_cmdline(pgid):
        raise RuntimeError(f"refusing to signal pgid {pgid}: leader is not an engine of {home}")
    os.killpg(pgid, sig)


def gpu_compute_pids(indices: Sequence[int]) -> set[int]:
    """Every compute PID on the given GPUs (aliens included) for a never-killed-anything check."""
    return {pid for idx in indices for pid, _mb in gpu_pids(idx)}


def assert_alive(pids: set[int], what: str) -> None:
    """Fail if any of pids disappeared; goethe must never stop processes it did not start."""
    gone = sorted(p for p in pids if not pid_alive(p))
    if gone:
        raise RuntimeError(f"{what} {gone} disappeared during the test; goethe must never stop them")


def stub_engine_pids(home: Path) -> list[int]:
    """Return live stub_engine.py PIDs whose cmdline references this GOETHE_HOME."""
    needles = {str(home), str(home.resolve())}
    out: list[int] = []
    for pid, st in process_table().items():
        if st[0] == "Z":
            continue
        cmd = _cmdline(pid)
        if "stub_engine.py" in cmd and any(n in cmd for n in needles):
            out.append(pid)
    return sorted(out)


def assert_no_stub_engines(home: Path) -> None:
    """Fail when any stub engine for this temp home is still alive."""
    alive = [p for p in stub_engine_pids(home) if pid_alive(p)]
    if alive:
        raise RuntimeError(f"stub engines still running for {home}: {alive}")


def home_process_tree(home: Path) -> set[int]:
    """PIDs whose cmdline or environ references home, plus all of their descendants."""
    needles = {str(home).encode(), str(home.resolve()).encode()}
    table = process_table()
    tree = {
        pid
        for pid in table
        if any(n in _proc_bytes(pid, "cmdline") or n in _proc_bytes(pid, "environ") for n in needles)
    }
    grew = True
    while grew:
        added = {pid for pid, st in table.items() if st[1] in tree and pid not in tree}
        tree |= added
        grew = bool(added)
    return tree


def home_zombies(home: Path) -> list[int]:
    """Defunct processes anywhere in the process tree rooted under home."""
    table = process_table()
    return sorted(pid for pid in home_process_tree(home) if pid in table and table[pid][0] == "Z")


def assert_no_zombies(home: Path, timeout: float = 3.0) -> None:
    """Fail when the temp home's process tree still holds defunct processes after timeout, naming them."""
    if not wait_until(lambda: not home_zombies(home), timeout=timeout, interval=0.1):
        table = process_table()
        named = [
            f"{z} ({_proc_bytes(z, 'comm').decode(errors='replace').strip()}, parent {table[z][1]} "
            f"{proc_cmdline(table[z][1])[:60]!r})"
            for z in home_zombies(home)
            if z in table
        ]
        raise RuntimeError(f"defunct processes under {home}: {named}")


def home_goethe_env(home: Path) -> dict[str, str]:
    """Environment for goethe/goethed using ports stored under a temp GOETHE_HOME."""
    ports_path = home / "goethe_test_ports.json"
    if ports_path.is_file():
        data = json.loads(ports_path.read_text(encoding="utf-8"))
        mux = int(data["mux"])
        public = int(data["public"])
    else:
        mux = int(os.environ.get("GOETHE_MUX_PORT", "30000"))
        public = int(os.environ.get("GOETHE_PUBLIC_PORT", "8443"))
    return {
        "GOETHE_HOME": str(home),
        "GOETHE_MUX_PORT": str(mux),
        "GOETHE_PUBLIC_PORT": str(public),
        "GOETHE_SOCK": str(home / "goethe.sock"),
    }


def test_mux_port(home: Path) -> int:
    """Multiplexer port configured for a temp GOETHE_HOME."""
    return int(home_goethe_env(home)["GOETHE_MUX_PORT"])


def test_public_port(home: Path) -> int:
    """Public nginx port configured for a temp GOETHE_HOME."""
    return int(home_goethe_env(home)["GOETHE_PUBLIC_PORT"])


def write_stub_python(pyenv_bin: Path, real: Path) -> None:
    """Replace py-env/bin/python (unlinked first: never write through a venv symlink) with a stub launcher."""
    wrapper = pyenv_bin / "python"
    wrapper.unlink(missing_ok=True)
    if real.resolve() == wrapper.resolve():
        raise RuntimeError(f"{real} resolves to the stub launcher itself; pass a venv-local alias of the interpreter")
    wrapper.write_text(
        "#!/bin/bash\n"
        f'REAL="{real}"\n'
        f'STUB="{STUB_ENGINE}"\n'
        'if [[ "$1" == "-m" && "$2" == "sglang.launch_server" ]]; then\n'
        '  exec "$REAL" "$STUB" "${@:3}"\n'
        "fi\n"
        'if [[ "$1" == "-c" && "$2" == *sglang* ]]; then echo ok; exit 0; fi\n'
        'exec "$REAL" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def _shutdown_home_daemon(home: Path) -> None:
    """Stop this temp home's goethed (engines and nginx with it), else the verified nginx a killed goethed left."""
    sock = home / "goethe.sock"
    if sock.exists():
        try:
            goethe_rpc(home, "shutdown", None, timeout=10.0)
            wait_until(lambda: not sock.exists(), timeout=15.0, interval=0.1)
        except (OSError, ValueError):
            pass
    if (home / "nginx" / "nginx.pid").is_file():
        try:
            run([str(require_cli("goethe")), "down"], env=home_goethe_env(home), timeout=30)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass


def _teardown_home(td: Path, strict: bool) -> None:
    """Remove a temp home; when strict, fail if any process (engine, CLI, download) or file outlived the test."""
    leftover = sorted(p for p in home_process_tree(td) if pid_alive(p))
    names = [proc_cmdline(p)[:100] for p in leftover]
    _shutdown_home_daemon(td)
    shutil.rmtree(td, ignore_errors=True)
    if leftover:
        wait_until(lambda: not any(pid_alive(p) for p in leftover), timeout=10.0)
        shutil.rmtree(td, ignore_errors=True)
    if strict and leftover:
        raise RuntimeError(f"processes outlived temp home {td}: {list(zip(leftover, names, strict=True))}")
    if strict and td.exists():
        raise RuntimeError(f"temp home {td} could not be removed: {[str(x) for x in td.rglob('*')][:5]}")


@contextmanager
def temp_home(
    stub_python: bool = True,
    real_python: Path | None = None,
    tag: str = "home",
    pyenv: bool = True,
) -> Iterator[Path]:
    """Create an isolated GOETHE_HOME under /tmp (py-env stub unless pyenv=False) and remove it after."""
    td = Path(tempfile.mkdtemp(prefix=run_temp_prefix(tag), dir="/tmp"))
    _temp_homes_active.add(td)
    mux, public = _allocate_mux_public_ports()
    (td / "goethe_test_ports.json").write_text(json.dumps({"mux": mux, "public": public}), encoding="utf-8")
    saved = {k: os.environ.get(k) for k in ("GOETHE_HOME", "GOETHE_MUX_PORT", "GOETHE_PUBLIC_PORT", "GOETHE_SOCK")}
    os.environ.update(
        {
            "GOETHE_HOME": str(td),
            "GOETHE_MUX_PORT": str(mux),
            "GOETHE_PUBLIC_PORT": str(public),
            "GOETHE_SOCK": str(td / "goethe.sock"),
        }
    )
    ok = False
    try:
        for sub in ("models", "logs", "nginx", "bin"):
            (td / sub).mkdir(parents=True, exist_ok=True)
        if pyenv:
            pyenv_bin = td / "py-env" / "bin"
            pyenv_bin.mkdir(parents=True)
            real = real_python or resolve_python()
            if stub_python:
                write_stub_python(pyenv_bin, real)
            else:
                shutil.copy2(real, pyenv_bin / "python")
        yield td
        ok = True
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        _temp_homes_active.discard(td)
        _teardown_home(td, strict=ok)


@contextmanager
def install_temp_home(stub_python: bool = True, real_python: Path | None = None) -> Iterator[Path]:
    """Temp GOETHE_HOME for install smoke tests, scoped to the gate run id."""
    with temp_home(stub_python=stub_python, real_python=real_python, tag="install") as home:
        yield home


def write_card(
    model_dir: Path,
    model_id: str,
    ctx: int = 4096,
    max_conc: int = 4,
    env: dict[str, str] | None = None,
    extra: str = "",
) -> Path:
    """Write goethe_card.toml; env becomes the [env] table goethed passes to the engine (stub knobs)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'model_id="{model_id}"', f"context_length={ctx}", f"max_running_requests={max_conc}"]
    if extra:
        lines.append(extra.rstrip("\n"))
    if env:
        lines += ["", "[env]", *(f'{k}="{v}"' for k, v in env.items())]
    path = model_dir / "goethe_card.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stub_launch(model_dir: Path) -> dict[str, Any]:
    """Launch record the stub engine wrote: argv, parsed flags, env and active STUB_* knobs."""
    return json.loads((model_dir / "stub_launch_argv.json").read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON-lines file, empty when missing."""
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def stub_requests(model_dir: Path) -> list[dict[str, Any]]:
    """Requests the stub engine saw: method, path, status and lower-cased headers."""
    return _jsonl(model_dir / "stub_requests.jsonl")


def stub_signals(model_dir: Path) -> list[dict[str, Any]]:
    """Signals the stub engine received and whether it ignored them."""
    return _jsonl(model_dir / "stub_signals.jsonl")


def make_tls_certs(home: Path) -> None:
    """Generate a self-signed TLS pair under GOETHE_HOME/nginx/certs."""
    cert = home / "nginx" / "certs"
    cert.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(cert / "server.key"), "-out", str(cert / "server.crt"),
            "-days", "1", "-nodes", "-subj", "/CN=goethe-test",
        ],
        check=True,
        capture_output=True,
    )


def _nginx_runs(binary: Path) -> bool:
    return binary.is_file() and subprocess.run([str(binary), "-v"], capture_output=True, timeout=30).returncode == 0


def ensure_nginx(home: Path) -> Path:
    """A working nginx: NGINX_BIN, else nginx, nginx-core or nginx-light extracted rootless under GOETHE_HOME."""
    env_bin = os.environ.get("NGINX_BIN")
    if env_bin and _nginx_runs(Path(env_bin)):
        return Path(env_bin)
    root = home / "nginx-root"
    binary = root / "usr" / "sbin" / "nginx"
    if _nginx_runs(binary):
        return binary
    debs = root / "debs"
    debs.mkdir(parents=True, exist_ok=True)
    for pkg in ("nginx", "nginx-core", "nginx-light"):
        if subprocess.run(["apt-get", "download", pkg, "nginx-common"], cwd=str(debs), capture_output=True).returncode:
            continue
        for deb in sorted(debs.glob("*.deb")):
            subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], check=True, capture_output=True, timeout=60)
        if _nginx_runs(binary):
            return binary
    raise RuntimeError("no working nginx: set NGINX_BIN or install nginx-core")


def _ssl_ctx() -> ssl.SSLContext:
    """TLS context that accepts goethe self-signed certs in tests."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[int, bytes]:
    """HTTP GET returning status and body (status 0 when no HTTP response arrived)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    kw: dict[str, Any] = {"timeout": timeout}
    if url.startswith("https:"):
        kw["context"] = _ssl_ctx()
    try:
        with urllib.request.urlopen(req, **kw) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return 0, b""


def http_post(
    url: str,
    body: bytes,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    """HTTP POST returning status and body."""
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    kw: dict[str, Any] = {"timeout": timeout}
    if url.startswith("https:"):
        kw["context"] = _ssl_ctx()
    try:
        with urllib.request.urlopen(req, **kw) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return 0, b""


def sse_events(
    url: str, body: bytes, headers: dict[str, str] | None = None, max_chunks: int = 64
) -> list[tuple[float, bytes]]:
    """POST and return (wall-clock arrival time, payload) for each SSE data line."""
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    kw: dict[str, Any] = {"timeout": 120}
    if url.startswith("https:"):
        kw["context"] = _ssl_ctx()
    events: list[tuple[float, bytes]] = []
    try:
        with urllib.request.urlopen(req, **kw) as resp:
            while len(events) < max_chunks:
                line = resp.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    events.append((time.time(), line[5:].strip()))
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        pass
    return events


def read_sse(url: str, body: bytes, headers: dict[str, str] | None = None, max_chunks: int = 64) -> list[float]:
    """POST and record arrival times (seconds since the request) for each SSE chunk."""
    start = time.time()
    return [t - start for t, _ in sse_events(url, body, headers=headers, max_chunks=max_chunks)]


def ensure_schema(home: Path) -> None:
    """Have the goethe CLI create state.db so seeded rows land in the product schema."""
    db = home / "state.db"
    if db.is_file() and sqlite_query(db, "PRAGMA user_version")[0][0] > 0:
        return
    rr = run([str(require_cli("goethe")), "alias"], env={"GOETHE_HOME": str(home)}, timeout=60)
    if rr.returncode != 0 or not db.is_file():
        raise RuntimeError(f"goethe could not create state.db: {_goethe_error_snippet(rr)}")


def seed_installed_model(
    home: Path,
    model_id: str,
    rel_dir: str,
    static_vram_gb: float = 10.0,
    kv_bytes: int = 1000,
    default_ctx: int = 4096,
    default_max_conc: int = 4,
    default_tp: int = 1,
    kv_elems: int | None = None,
    max_ctx: int | None = None,
    num_attention_heads: int | None = None,
    num_kv_heads: int | None = None,
    attention: str | None = None,
    managed: int | None = None,
) -> None:
    """Insert an installed_models row into the product-created schema."""
    ensure_schema(home)
    cols = [
        "model_id", "local_path", "architecture", "static_vram_gb", "kv_bytes_per_token", "default_ctx",
        "default_max_conc", "default_tp", "kv_elems_per_token", "max_ctx", "num_attention_heads", "num_kv_heads",
        "attention",
    ]
    vals: list[Any] = [
        model_id, rel_dir, "dense", static_vram_gb, kv_bytes, default_ctx, default_max_conc, default_tp,
        kv_elems if kv_elems is not None else kv_bytes, max_ctx if max_ctx is not None else default_ctx,
        num_attention_heads or 8, num_kv_heads or 2, attention or "gqa",
    ]
    if managed is not None:
        cols.append("managed")
        vals.append(managed)
    con = sqlite3.connect(str(home / "state.db"), timeout=30)
    try:
        con.execute(
            f"INSERT OR REPLACE INTO installed_models ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            vals,
        )
        con.commit()
    finally:
        con.close()


def sqlite_query(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Run a read-only sqlite query."""
    con = sqlite3.connect(str(db_path), timeout=30)
    try:
        return list(con.execute(sql, params).fetchall())
    finally:
        con.close()


def spec_ddl_tables() -> dict[str, list[str]]:
    """Expected installed_models schema columns from the contract."""
    return {
        "installed_models": [
            "model_id", "local_path", "architecture", "static_vram_gb", "kv_bytes_per_token", "default_ctx",
            "default_max_conc", "default_tp", "installed_at",
        ],
        "loaded_instances": [
            "instance_id", "model_id", "pid", "pgid", "internal_port", "gpu_indices", "allocated_vram_gb_per_gpu",
            "mem_fraction_static", "ctx_len", "max_conc", "status",
        ],
        "api_keys": ["key_id", "name", "token_hash", "raw_prefix", "rpm_limit", "created_at"],
        "ip_allowlist": ["cidr", "description", "created_at"],
    }


def vram_per_gpu_gb(w_gb: float, l_ctx: int, k_bytes: int, c_max: int, tp: int) -> float:
    """Contract VRAM per GPU formula."""
    kv = l_ctx * k_bytes * min(c_max, 4) * 1e-9
    return (w_gb + kv) / tp + 1.8


def foreign_gpu_process(gpu_index: int, mb: int = 3072) -> Proc:
    """Hold mb MiB on a test GPU (under its flock) with the dev py-env torch, as an unmanaged process."""
    release = _acquire_gpu_flocks([gpu_index])
    elems = max(1, (mb * 1024 * 1024) // 4)
    script = (
        "import time, torch\n"
        f"x = torch.empty(({elems},), dtype=torch.float32, device='cuda')\n"
        "time.sleep(3600)\n"
    )
    try:
        proc = spawn(
            [str(dev_python()), "-c", script],
            env={"CUDA_VISIBLE_DEVICES": str(gpu_index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            label=f"foreign-gpu-{gpu_index}",
        )
    except Exception:
        release()
        raise
    proc._gpu_flock_release = release
    return proc


def _readonly_mirror(src: Path, dst: Path) -> None:
    """Mirror a directory tree: real directories, copies of small files, symlinks to large ones."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if entry.is_symlink():
            target.symlink_to(os.readlink(entry))
        elif entry.is_dir():
            _readonly_mirror(entry, target)
        elif entry.stat().st_size <= 1 << 20:
            shutil.copy2(entry, target)
        else:
            target.symlink_to(entry)


def tree_fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    """Map relative file path to (size, mtime_ns) so a test can prove it left a tree untouched."""
    out: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            st = path.stat()
            out[str(path.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def link_dev_pyenv(home: Path) -> None:
    """Point home/py-env at the dev py-env (read-only use) for checks that need the real SGLang fork."""
    (home / "py-env").symlink_to(dev_home() / "py-env", target_is_directory=True)


def isolated_real_home(work: Path, scrub: Sequence[str]) -> dict[str, str]:
    """Build a private home from the dev home for real-engine gates and return its goethe env."""
    src = dev_home()
    if work.resolve() == src.resolve():
        raise RuntimeError("refusing to use the dev GOETHE_HOME as a test home")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    link_dev_pyenv(work)
    if (src / "models").is_dir():
        _readonly_mirror(src / "models", work / "models")
    if (src / "state.db").is_file():
        ro = sqlite3.connect(f"file:{src / 'state.db'}?mode=ro", uri=True, timeout=30)
        dst = sqlite3.connect(str(work / "state.db"))
        try:
            ro.backup(dst)
            present = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in scrub:
                if table in present:
                    dst.execute(f"DELETE FROM {table}")
            dst.commit()
        finally:
            dst.close()
            ro.close()
    mux, public = _allocate_mux_public_ports()
    env = {
        "GOETHE_HOME": str(work),
        "GOETHE_MUX_PORT": str(mux),
        "GOETHE_PUBLIC_PORT": str(public),
        "GOETHE_SOCK": str(work / "goethe.sock"),
    }
    _profile_hardware(work, src, env)
    return env


def _profile_hardware(work: Path, src: Path, env: dict[str, str]) -> None:
    """hardware_state.json for a real-engine home (budgets need it): goethe hw profile, else the dev home's copy."""
    rr = run([str(require_cli("goethe")), "hw", "profile"], env={**os.environ, **env}, timeout=120)
    if rr.returncode == 0 and (work / "hardware_state.json").is_file():
        return
    if not (src / "hardware_state.json").is_file():
        raise RuntimeError(f"no hardware profile for {work}: goethe hw profile exit {rr.returncode} {rr.stderr[-200:]}")
    shutil.copy2(src / "hardware_state.json", work / "hardware_state.json")


def make_release_key(dirpath: Path) -> tuple[Path, Path]:
    """Generate a P-256 signing key pair under dirpath and return (private, public)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    key, pub = dirpath / "release.key", dirpath / "release_pub.pem"
    subprocess.run(["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(key)], check=True)
    subprocess.run(["openssl", "ec", "-in", str(key), "-pubout", "-out", str(pub)], check=True, capture_output=True)
    return key, pub


def sign_cdn_tree(cdn: Path, key: Path, version: str | None = None) -> None:
    """Write VERSION (when given), SHA256SUMS over every file and SHA256SUMS.sig for a CDN tree."""
    if version is not None:
        (cdn / "VERSION").write_text(version + "\n", encoding="utf-8")
    lines = [
        f"{sha256_file(p)}  {p.relative_to(cdn).as_posix()}"
        for p in sorted(cdn.rglob("*"))
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sig")
    ]
    (cdn / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(cdn / "SHA256SUMS.sig"), str(cdn / "SHA256SUMS")],
        check=True,
    )


def _allowed_zombie_ppid(zombie_ppid: int, gate_pid: int, test_pids: set[int]) -> bool:
    return zombie_ppid in test_pids or zombie_ppid == gate_pid or zombie_ppid in _spawned_pids


def _kill_owned_group(pgid: int) -> None:
    """TERM then KILL a process group the caller has proven it owns."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if not wait_until(lambda: not _group_members(pgid), timeout=5.0, interval=0.1):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wait_until(lambda: not _group_members(pgid), timeout=5.0, interval=0.1)


def reap_run_artifacts(session_pgid: int | None = None) -> tuple[list[str], list[int]]:
    """Kill groups tests of this run recorded (identity-checked) and return (leftovers, recorded pids)."""
    groups: list[tuple[int, int | None]] = []
    pids: list[int] = []
    for path in Path("/tmp").glob(f"{run_temp_prefix('artifacts')}*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        pids += [int(p) for p in data.get("pids", [])]
        groups += [(int(g["pgid"]), g.get("start")) for g in data.get("groups", [])]
        path.unlink(missing_ok=True)
    if session_pgid:
        groups.append((session_pgid, None))
    leftovers: list[str] = []
    for pgid, start in groups:
        members = _group_members(pgid)
        if not members:
            continue
        leader = _stat(pgid)
        if leader is not None and (start is None or leader[3] != start):
            continue
        leftovers.append(f"pgid {pgid} still had {members}: {_cmdline(members[0])[:160]}")
        _kill_owned_group(pgid)
    return leftovers, pids


def _end_leaked(leaked: list[tuple[int, str]]) -> None:
    """Stop processes this gate run leaked (their command line names its temp paths) so later tests start clean."""
    for pid, cmd in leaked:
        if pid_alive(pid) and _cmdline(pid) == cmd:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
    if not wait_until(lambda: not any(pid_alive(p) for p, _ in leaked), timeout=5.0, interval=0.1):
        for pid, cmd in leaked:
            if pid_alive(pid) and _cmdline(pid) == cmd:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


def post_test_hygiene(
    gpu_indices: Sequence[int],
    pre_mem: dict[int, int],
    gate_pid: int,
    test_pids: Sequence[int] | None = None,
    recorded_pids: Sequence[int] = (),
) -> None:
    """Verify no leaked test processes, zombies, temp dirs, or GPU PIDs for this gate run."""
    marker = run_temp_marker()
    tset = set(test_pids or []) | set(_spawned_pids)
    table = process_table()
    leaked = [(p, _cmdline(p)) for p, st in table.items() if st[0] != "Z" and p not in tset and marker in _cmdline(p)]
    if leaked:
        _end_leaked(leaked)
        raise RuntimeError(f"leftover process {leaked[0][0]}: {leaked[0][1][:200]} ({len(leaked)} in all, now stopped)")
    for zpid, st in table.items():
        if st[0] != "Z":
            continue
        ppid = st[1]
        if marker not in _cmdline(zpid) and marker not in _cmdline(ppid):
            continue
        if _allowed_zombie_ppid(ppid, gate_pid, tset):
            raise RuntimeError(f"defunct process pid={zpid} ppid={ppid}")
    for path in leaked_temp_paths():
        raise RuntimeError(f"leaked temp path: {path}")
    for pid in set(_spawned_pids) | set(recorded_pids):
        if pid_alive(pid):
            raise RuntimeError(f"spawned pid {pid} still alive")
        for idx in gpu_indices:
            if any(gpid == pid for gpid, _used in gpu_pids(idx)):
                raise RuntimeError(f"spawned pid {pid} still on GPU {idx}")
    if os.environ.get("GOETHE_TEST_EXCLUSIVE_GPUS", "0") == "1":
        for idx in gpu_indices:
            growth = gpu_mem_used_mb(idx) - pre_mem.get(idx, 0)
            if growth > 300:
                raise RuntimeError(f"GPU {idx} mem grew {growth} MiB under exclusive mode")


def gate_after_test(pre_json: str, session_pgid: int) -> None:
    """gate.sh hook: reap groups the finished test left behind, then run hygiene checks."""
    leftovers, pids = reap_run_artifacts(session_pgid)
    if leftovers:
        raise SystemExit("test left processes running (reaped): " + "; ".join(leftovers))
    post_test_hygiene(
        test_gpus(), json.loads(pre_json), gate_pid=int(os.environ.get("GOETHE_GATE_PARENT", "0")),
        test_pids=[os.getpid()], recorded_pids=pids,
    )


def huggingface_reachable(timeout: float = 3.0) -> bool:
    """Return True if huggingface.co responds within timeout seconds."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request("https://huggingface.co", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main_exit(fn: Callable[[], None]) -> None:
    """Run a test main and map Skip/Pending to exit codes 75/77."""
    install_test_signal_handlers()
    try:
        fn()
    except Pending as exc:
        print(str(exc))
        sys.exit(77)
    except Skip as exc:
        print(str(exc))
        sys.exit(75)
