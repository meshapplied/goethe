"""Stdlib helpers for mesh stage-gate black-box tests."""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import ssl
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
MESH_PKG = Path(os.environ.get("MESH_PKG", "/scratch/ternary-flash/goethe-release/mesh"))
TESTS_DIR = MESH_PKG / "tests"
LIB_DIR = TESTS_DIR / "lib"
STUB_ENGINE = LIB_DIR / "stub_engine.py"

_gpu_pid_query: Callable[[int], list[tuple[int, int]]] | None = None
_spawned_pids: set[int] = set()
_gate_roots: set[int] = set()
_gpu_locks_held: set[int] = set()


def gate_run_id() -> str:
    """Return the current gate run id for temp and hygiene scoping."""
    return os.environ.get("MESH_GATE_RUN_ID", str(os.getpid()))


def run_temp_prefix(tag: str) -> str:
    """Build a /tmp name prefix mesh-test-<run_id>-<tag>-."""
    return f"mesh-test-{gate_run_id()}-{tag}-"


def run_temp_marker() -> str:
    """Substring that identifies temp paths and cmdlines for this gate run."""
    return f"mesh-test-{gate_run_id()}-"


class Skip(Exception):
    """Skip this test with a reason."""


class Pending(Exception):
    """Product binary or stage not implemented yet."""


@dataclass
class RunResult:
    """Captured subprocess result."""

    returncode: int
    stdout: str
    stderr: str


@dataclass
class Proc:
    """A process group started with setsid for clean teardown."""

    popen: subprocess.Popen[Any]
    pgid: int
    label: str
    _stopped: bool = False
    _gpu_flock_release: Callable[[], None] | None = None

    def stop(self, sigterm_timeout: float = 3.0) -> None:
        """Send SIGTERM to the group, then SIGKILL if needed."""
        if self._stopped:
            return
        self._stopped = True
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            unregister_pid(self.popen.pid)
            return
        deadline = time.time() + sigterm_timeout
        while time.time() < deadline:
            if self.popen.poll() is not None:
                break
            time.sleep(0.05)
        if self.popen.poll() is None:
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.popen.kill()
        unregister_pid(self.popen.pid)
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
    path = _artifact_path()
    if not _spawned_pids and not _gate_roots:
        if path.is_file():
            path.unlink(missing_ok=True)
        return
    path.write_text(
        json.dumps({"pids": sorted(_spawned_pids), "pgids": sorted(_gate_roots)}),
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
        shutil.rmtree(td, ignore_errors=True)
    _temp_homes_active.clear()
    path = _artifact_path()
    if path.is_file():
        path.unlink(missing_ok=True)
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


def resolve_mesh_home() -> Path:
    """Resolve MESH_HOME using the product contract."""
    if os.environ.get("MESH_HOME"):
        return Path(os.environ["MESH_HOME"])
    workspace = Path("/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / ".mesh"
    varlib = Path("/var/lib/mesh")
    if varlib.is_dir() and os.access(varlib, os.W_OK):
        return varlib
    return Path.home() / ".mesh"


def resolve_python() -> Path:
    """Return the hermetic Python interpreter for tests."""
    home = resolve_mesh_home()
    candidate = home / "py-env" / "bin" / "python"
    if candidate.is_file():
        return candidate
    env = os.environ.get("MESH_PYTHON")
    if env:
        return Path(env)
    return Path(sys.executable)


def mesh_bin(name: str) -> Path:
    """Path to mesh or meshd binary."""
    bindir = os.environ.get("MESH_BIN_DIR")
    if bindir:
        return Path(bindir) / name
    release = MESH_PKG / "target" / "release" / name
    if release.is_file():
        return release
    home = resolve_mesh_home()
    cand = home / "bin" / name
    if cand.is_file():
        return cand
    return release


def require_cli(name: str) -> Path:
    """Require mesh or meshd on disk."""
    path = mesh_bin(name)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise Pending(f"{name} binary not built at {path}")
    return path


def test_gpus() -> list[int]:
    """Physical GPU indices allowed for tests."""
    raw = os.environ.get("MESH_TEST_GPUS", "2,3")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _acquire_gpu_flocks(indices: Sequence[int]) -> Callable[[], None]:
    """Acquire /tmp/mesh-gpu locks in ascending index order."""
    gpus = sorted({int(i) for i in indices if int(i) not in _gpu_locks_held})
    acquired: list[int] = []
    fds: list[int] = []
    try:
        for idx in gpus:
            path = f"/tmp/mesh-gpu{idx}.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            deadline = time.time() + 1800.0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        os.close(fd)
                        raise TimeoutError(f"timed out waiting for GPU {idx} lock")
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
        for fd, idx in zip(reversed(fds), reversed(acquired)):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            _gpu_locks_held.discard(idx)

    return release


@contextmanager
def gpu_flock(indices: Sequence[int] | None = None) -> Iterator[None]:
    """Hold mesh GPU flock locks until the context exits."""
    release = _acquire_gpu_flocks(indices if indices is not None else test_gpus())
    try:
        yield
    finally:
        release()


def require_gpus(n: int) -> list[int]:
    """Require at least n test GPUs or skip."""
    gpus = test_gpus()
    if len(gpus) < n:
        raise Skip(f"need {n} GPUs in MESH_TEST_GPUS, have {len(gpus)}")
    return gpus


def run(
    cmd: Sequence[str],
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> RunResult:
    """Run a command and capture stdout/stderr."""
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
    )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def _mesh_error_snippet(rr: RunResult) -> str:
    """Last 800 chars of mesh stderr with warning noise removed."""
    raw = rr.stderr or rr.stdout or ""
    lines = [ln for ln in raw.splitlines() if "FutureWarning" not in ln and "warnings.warn" not in ln]
    text = "\n".join(lines).strip() or raw.strip()
    return text[-800:] if len(text) > 800 else text


def classify_mesh_result(rr: RunResult, binary_missing_ok: bool = False) -> None:
    """Map mesh CLI exit codes to Pending vs FAIL."""
    err = (rr.stderr or "").lower()
    if rr.returncode == 2 and ("unrecognized" in err or "unknown" in err):
        raise Pending(rr.stderr.strip() or "unknown subcommand")
    if rr.returncode != 0:
        raise RuntimeError(f"mesh exit {rr.returncode}: {_mesh_error_snippet(rr)}")


def run_mesh(
    cmd: Sequence[str],
    timeout: float = 180.0,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> RunResult:
    """Run mesh CLI and classify failures."""
    rr = run(cmd, timeout=timeout, env=env, input_text=input_text)
    classify_mesh_result(rr)
    return rr


def spawn(
    cmd: Sequence[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    label: str = "proc",
) -> Proc:
    """Start a command in its own process group."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    pop = subprocess.Popen(
        list(cmd),
        env=merged,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    pgid = os.getpgid(pop.pid)
    register_pid(pop.pid)
    _gate_roots.add(pgid)
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
    return False


def free_port() -> int:
    """Bind an ephemeral localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _nvidia_smi_query(args: list[str]) -> str:
    proc = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or "nvidia-smi failed")
    return proc.stdout


def gpu_mem_used_mb(index: int) -> int:
    """Used VRAM in MiB for a physical GPU index."""
    out = _nvidia_smi_query(
        [
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--id={index}",
        ]
    )
    return int(out.strip().split("\n")[0].strip())


def gpu_mem_snapshot(indices: Sequence[int]) -> dict[int, int]:
    """Snapshot used VRAM for each GPU index."""
    return {i: gpu_mem_used_mb(i) for i in indices}


def gpu_pids(index: int) -> list[tuple[int, int]]:
    """Return (pid, used_mb) compute processes on a GPU."""
    if _gpu_pid_query is not None:
        return _gpu_pid_query(index)
    out = _nvidia_smi_query(
        [
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
            f"--id={index}",
        ]
    )
    rows: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    return rows


def leaked_temp_paths() -> list[str]:
    """Return leftover temp paths for the current gate run only."""
    tmp = Path("/tmp")
    prefix = f"mesh-test-{gate_run_id()}-"
    paths: list[str] = []
    for p in tmp.glob(f"{prefix}*"):
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


def wait_meshd_socket(home: Path, timeout: float = 30.0) -> None:
    """Wait until meshd creates mesh.sock under MESH_HOME."""
    if not wait_until(lambda: (home / "mesh.sock").is_socket(), timeout=timeout):
        raise RuntimeError("meshd did not create mesh.sock")


def stub_engine_pids(home: Path) -> list[int]:
    """Return stub_engine.py PIDs whose cmdline references this MESH_HOME."""
    home_s = str(home.resolve())
    proc = subprocess.run(["pgrep", "-f", "stub_engine.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    out: list[int] = []
    for tok in proc.stdout.split():
        if not tok.strip().isdigit():
            continue
        pid = int(tok)
        ps = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        if ps.returncode == 0 and home_s in ps.stdout:
            out.append(pid)
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def assert_no_stub_engines(home: Path) -> None:
    """Fail when any stub engine for this temp home is still alive."""
    alive = [p for p in stub_engine_pids(home) if _pid_alive(p)]
    if alive:
        raise RuntimeError(f"stub engines still running for {home}: {alive}")


def home_mesh_env(home: Path) -> dict[str, str]:
    """Environment for mesh/meshd using ports stored under a temp MESH_HOME."""
    ports_path = home / "mesh_test_ports.json"
    if ports_path.is_file():
        data = json.loads(ports_path.read_text(encoding="utf-8"))
        mux = int(data["mux"])
        public = int(data["public"])
    else:
        mux = int(os.environ.get("MESH_MUX_PORT", "30000"))
        public = int(os.environ.get("MESH_PUBLIC_PORT", "8443"))
    return {
        "MESH_HOME": str(home),
        "MESH_MUX_PORT": str(mux),
        "MESH_PUBLIC_PORT": str(public),
        "MESH_SOCK": str(home / "mesh.sock"),
    }


def test_mux_port(home: Path) -> int:
    """Multiplexer port configured for a temp MESH_HOME."""
    return int(home_mesh_env(home)["MESH_MUX_PORT"])


def test_public_port(home: Path) -> int:
    """Public nginx port configured for a temp MESH_HOME."""
    return int(home_mesh_env(home)["MESH_PUBLIC_PORT"])


@contextmanager
def temp_home(
    stub_python: bool = True,
    real_python: Path | None = None,
    tag: str = "home",
) -> Iterator[Path]:
    """Create an isolated MESH_HOME under /tmp and remove it after."""
    td = Path(tempfile.mkdtemp(prefix=run_temp_prefix(tag), dir="/tmp"))
    _temp_homes_active.add(td)
    mux, public = _allocate_mux_public_ports()
    (td / "mesh_test_ports.json").write_text(json.dumps({"mux": mux, "public": public}), encoding="utf-8")
    prev = os.environ.get("MESH_HOME")
    prev_mux = os.environ.get("MESH_MUX_PORT")
    prev_public = os.environ.get("MESH_PUBLIC_PORT")
    os.environ["MESH_HOME"] = str(td)
    os.environ["MESH_MUX_PORT"] = str(mux)
    os.environ["MESH_PUBLIC_PORT"] = str(public)
    try:
        (td / "models").mkdir(parents=True, exist_ok=True)
        (td / "logs").mkdir(exist_ok=True)
        (td / "nginx").mkdir(exist_ok=True)
        pyenv = td / "py-env" / "bin"
        pyenv.mkdir(parents=True)
        real = real_python or resolve_python()
        if stub_python:
            wrapper = pyenv / "python"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
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
        else:
            shutil.copy2(real, pyenv / "python")
        (td / "bin").mkdir(exist_ok=True)
        yield td
    finally:
        if prev is None:
            os.environ.pop("MESH_HOME", None)
        else:
            os.environ["MESH_HOME"] = prev
        if prev_mux is None:
            os.environ.pop("MESH_MUX_PORT", None)
        else:
            os.environ["MESH_MUX_PORT"] = prev_mux
        if prev_public is None:
            os.environ.pop("MESH_PUBLIC_PORT", None)
        else:
            os.environ["MESH_PUBLIC_PORT"] = prev_public
        _temp_homes_active.discard(td)
        shutil.rmtree(td, ignore_errors=True)


@contextmanager
def install_temp_home(stub_python: bool = True, real_python: Path | None = None) -> Iterator[Path]:
    """Temp MESH_HOME for install smoke tests, scoped to the gate run id."""
    with temp_home(stub_python=stub_python, real_python=real_python, tag="install") as home:
        yield home


def ensure_nginx(home: Path) -> Path:
    """Return nginx binary, extracting rootless nginx into MESH_HOME if needed."""
    env_bin = os.environ.get("NGINX_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    rootless = home / "nginx-root" / "usr" / "sbin" / "nginx"
    if rootless.is_file():
        return rootless
    cache = home / "nginx-root"
    cache.mkdir(parents=True, exist_ok=True)
    deb_dir = cache / "debs"
    deb_dir.mkdir(exist_ok=True)
    for pkg in ("nginx", "nginx-common"):
        matches = sorted(deb_dir.glob(f"{pkg}_*.deb"))
        if not matches:
            subprocess.run(
                ["apt-get", "download", pkg],
                cwd=str(deb_dir),
                check=True,
                capture_output=True,
                timeout=120,
            )
            matches = sorted(deb_dir.glob(f"{pkg}_*.deb"))
        if not matches:
            raise RuntimeError(f"apt-get download missing deb for {pkg}")
        subprocess.run(
            ["dpkg-deb", "-x", str(matches[0]), str(cache)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    if not rootless.is_file():
        raise RuntimeError("rootless nginx extraction failed")
    return rootless


def _ssl_ctx() -> ssl.SSLContext:
    """TLS context that accepts mesh self-signed certs in tests."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[int, bytes]:
    """HTTP GET returning status and body."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        kw: dict[str, Any] = {"timeout": timeout}
        if url.startswith("https:"):
            kw["context"] = _ssl_ctx()
        with urllib.request.urlopen(req, **kw) as resp:
            if os.environ.get("MESH_DEBUG_HTTP"):
                print(f"HTTP GET {url} -> {resp.status}", flush=True)
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        if os.environ.get("MESH_DEBUG_HTTP"):
            print(f"HTTP GET {url} -> {err.code}", flush=True)
        return err.code, err.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        if os.environ.get("MESH_DEBUG_HTTP"):
            print(f"HTTP GET {url} -> 0", flush=True)
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
    try:
        kw: dict[str, Any] = {"timeout": timeout}
        if url.startswith("https:"):
            kw["context"] = _ssl_ctx()
        with urllib.request.urlopen(req, **kw) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return 0, b""


def read_sse(url: str, body: bytes, headers: dict[str, str] | None = None, max_chunks: int = 64) -> list[float]:
    """POST and record arrival times for each SSE chunk."""
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    times: list[float] = []
    start = time.time()
    kw: dict[str, Any] = {"timeout": 120}
    if url.startswith("https:"):
        kw["context"] = _ssl_ctx()
    try:
        with urllib.request.urlopen(req, **kw) as resp:
            while len(times) < max_chunks:
                line = resp.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    times.append(time.time() - start)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException):
        return []
    return times


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
) -> None:
    """Insert an installed_models row for supervisor and gateway gates."""
    db = home / "state.db"
    con = sqlite3.connect(str(db))
    try:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS installed_models (
              model_id TEXT PRIMARY KEY,
              local_path TEXT NOT NULL,
              architecture TEXT NOT NULL,
              static_vram_gb REAL NOT NULL,
              kv_bytes_per_token INTEGER NOT NULL,
              default_ctx INTEGER NOT NULL,
              default_max_conc INTEGER NOT NULL,
              default_tp INTEGER NOT NULL,
              kv_elems_per_token INTEGER,
              max_ctx INTEGER,
              num_attention_heads INTEGER,
              num_kv_heads INTEGER,
              attention TEXT,
              installed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS loaded_instances (
              instance_id TEXT PRIMARY KEY,
              model_id TEXT NOT NULL,
              pid INTEGER NOT NULL,
              pgid INTEGER NOT NULL,
              internal_port INTEGER NOT NULL,
              gpu_indices TEXT NOT NULL,
              allocated_vram_gb_per_gpu REAL NOT NULL,
              mem_fraction_static REAL NOT NULL,
              ctx_len INTEGER NOT NULL,
              max_conc INTEGER NOT NULL,
              status TEXT NOT NULL
            );
            """
        )
        elems = kv_elems if kv_elems is not None else kv_bytes
        mctx = max_ctx if max_ctx is not None else default_ctx
        con.execute(
            """
            INSERT OR REPLACE INTO installed_models
            (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
             default_ctx, default_max_conc, default_tp, kv_elems_per_token, max_ctx,
             num_attention_heads, num_kv_heads, attention, installed_at)
            VALUES (?, ?, 'dense', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                model_id,
                rel_dir,
                static_vram_gb,
                kv_bytes,
                default_ctx,
                default_max_conc,
                default_tp,
                elems,
                mctx,
                num_attention_heads or 8,
                num_kv_heads or 2,
                attention or "gqa",
            ),
        )
        con.commit()
    finally:
        con.close()


def sqlite_query(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Run a read-only sqlite query."""
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(sql, params)
        return list(cur.fetchall())
    finally:
        con.close()


def spec_ddl_tables() -> dict[str, list[str]]:
    """Expected installed_models schema columns from the contract."""
    return {
        "installed_models": [
            "model_id",
            "local_path",
            "architecture",
            "static_vram_gb",
            "kv_bytes_per_token",
            "default_ctx",
            "default_max_conc",
            "default_tp",
            "installed_at",
        ],
        "loaded_instances": [
            "instance_id",
            "model_id",
            "pid",
            "pgid",
            "internal_port",
            "gpu_indices",
            "allocated_vram_gb_per_gpu",
            "mem_fraction_static",
            "ctx_len",
            "max_conc",
            "status",
        ],
        "api_keys": ["key_id", "name", "token_hash", "raw_prefix", "rpm_limit", "created_at"],
        "ip_allowlist": ["cidr", "description", "created_at"],
    }


def vram_per_gpu_gb(w_gb: float, l_ctx: int, k_bytes: int, c_max: int, tp: int) -> float:
    """Contract VRAM per GPU formula."""
    kv = l_ctx * k_bytes * min(c_max, 4) * 1e-9
    return (w_gb + kv) / tp + 1.8


def foreign_gpu_process(gpu_index: int, mb: int = 10240) -> Proc:
    """Allocate VRAM on a test GPU using the real py-env torch."""
    release = _acquire_gpu_flocks([gpu_index])
    real_py = resolve_python()
    elems = max(1, (mb * 1024 * 1024) // 4)
    script = (
        "import os, time, torch\n"
        f"os.environ['CUDA_VISIBLE_DEVICES']='{gpu_index}'\n"
        f"torch.empty(({elems},), dtype=torch.float32, device='cuda')\n"
        "time.sleep(3600)\n"
    )
    proc = spawn([str(real_py), "-c", script], label=f"foreign-gpu-{gpu_index}")
    proc._gpu_flock_release = release
    return proc


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _allowed_zombie_ppid(zombie_ppid: int, gate_pid: int, test_pids: set[int]) -> bool:
    if zombie_ppid in test_pids or zombie_ppid == gate_pid:
        return True
    if zombie_ppid in _spawned_pids:
        return True
    return False


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def post_test_hygiene(
    gpu_indices: Sequence[int],
    pre_mem: dict[int, int],
    gate_pid: int,
    test_pids: Sequence[int] | None = None,
) -> None:
    """Verify no leaked test processes, temp dirs, or GPU PIDs for this gate run."""
    marker = run_temp_marker()
    tset = set(test_pids or []) | set(_spawned_pids)
    leaked = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if leaked.returncode == 0:
        for line in leaked.stdout.splitlines():
            if marker not in line:
                continue
            pid_s = line.strip().split(None, 1)[0]
            if int(pid_s) not in tset:
                raise RuntimeError(f"leftover process: {line.strip()}")
    ps_z = subprocess.run(["ps", "-eo", "stat,pid,ppid="], capture_output=True, text=True)
    if ps_z.returncode == 0:
        for line in ps_z.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3 or "Z" not in parts[0]:
                continue
            zpid = int(parts[1])
            ppid = int(parts[2])
            if marker not in _cmdline(zpid) and marker not in _cmdline(ppid):
                continue
            if not _allowed_zombie_ppid(ppid, gate_pid, tset):
                continue
            raise RuntimeError(f"defunct process pid={zpid} ppid={ppid}")
    for path in leaked_temp_paths():
        raise RuntimeError(f"leaked temp path: {path}")
    for pid in list(_spawned_pids):
        if _process_alive(pid):
            raise RuntimeError(f"spawned pid {pid} still alive")
        for idx in gpu_indices:
            for gpid, _used in gpu_pids(idx):
                if gpid == pid:
                    raise RuntimeError(f"spawned pid {pid} still on GPU {idx}")
    if os.environ.get("MESH_TEST_EXCLUSIVE_GPUS", "0") == "1":
        for idx in gpu_indices:
            growth = gpu_mem_used_mb(idx) - pre_mem.get(idx, 0)
            if growth > 300:
                raise RuntimeError(f"GPU {idx} mem grew {growth} MiB under exclusive mode")


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
    """Run a test main and map Skip/Pending to exit codes."""
    install_test_signal_handlers()
    try:
        fn()
    except Pending as exc:
        print(str(exc))
        sys.exit(77)
    except Skip as exc:
        print(str(exc))
        sys.exit(75)
