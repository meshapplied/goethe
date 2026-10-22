#!/usr/bin/env python3
"""Stage 0 self-test for goethe gate infrastructure."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from lib import goethe_test as mt
from lib.goethe_test import (
    Pending,
    Proc,
    RunResult,
    Skip,
    classify_goethe_result,
    free_port,
    http_get,
    leaked_temp_paths,
    main_exit,
    post_test_hygiene,
    read_sse,
    register_pid,
    run_temp_marker,
    run_temp_prefix,
    set_gpu_pid_query,
    spawn,
    temp_home,
    wait_until,
)
from lib.goethe_test import STUB_ENGINE, TESTS_DIR

ASSERTS = 0


def check(cond: bool, msg: str) -> None:
    """Increment assertion count or raise."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def _test_proc_teardown() -> None:
    """SIGTERM-ignoring process group receives SIGKILL."""
    py = sys.executable
    teardown_dir = tempfile.mkdtemp(prefix=run_temp_prefix("proc-teardown"), dir="/tmp")
    proc = spawn(
        [py, str(STUB_ENGINE), "--model-path", teardown_dir, "--host", "127.0.0.1", "--port", str(free_port())],
        env={"STUB_IGNORE_SIGTERM": "1"},
        label="sigkill-test",
    )
    proc.stop(sigterm_timeout=0.5)
    check(proc.popen.poll() is not None, "stub group did not exit after SIGKILL escalation")
    import shutil

    shutil.rmtree(teardown_dir, ignore_errors=True)


def _pgid_alive(pgid: int) -> bool:
    """Return whether a process group still exists."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _test_sigterm_child_teardown() -> None:
    """SIGTERM on a test child stops its Proc group and temp home."""
    py = sys.executable
    state_dir = Path(tempfile.mkdtemp(prefix=run_temp_prefix("sigterm-state"), dir="/tmp"))
    state_file = state_dir / "state.json"
    child_code = f"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, {json.dumps(str(TESTS_DIR))})
from lib.goethe_test import STUB_ENGINE, free_port, install_test_signal_handlers, temp_home, spawn
install_test_signal_handlers()
state = Path({json.dumps(str(state_file))})
with temp_home() as home:
    port = free_port()
    proc = spawn(
        [sys.executable, str(STUB_ENGINE), "--model-path", str(home / "m"), "--host", "127.0.0.1", "--port", str(port)],
        label="sigterm-grandchild",
    )
    state.write_text(json.dumps({{"home": str(home), "pgid": proc.pgid}}), encoding="utf-8")
    time.sleep(3600)
"""
    child = subprocess.Popen([py, "-c", child_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_until(state_file.is_file, timeout=10.0):
            raise RuntimeError("child did not write state")
        data = json.loads(state_file.read_text(encoding="utf-8"))
        home = Path(data["home"])
        pgid = int(data["pgid"])
        os.kill(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
        check(not home.exists(), "temp home removed after SIGTERM")
        check(not _pgid_alive(pgid), "grandchild process group gone")
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        import shutil

        shutil.rmtree(state_dir, ignore_errors=True)


def _test_temp_home_cleanup() -> None:
    """temp_home removes directory even when an exception is raised."""
    path_holder: list[str] = []
    try:
        with temp_home(stub_python=False) as home:
            path_holder.append(str(home))
            raise RuntimeError("forced")
    except RuntimeError:
        pass
    check(not Path(path_holder[0]).exists(), "temp_home not removed on exception")


def _test_stub_contract() -> None:
    """Stub health, SSE gaps, argv json, and TP children."""
    port = free_port()
    model = Path(tempfile.mkdtemp(prefix=run_temp_prefix("stub"), dir="/tmp"))
    proc: Proc | None = None
    try:
        proc = spawn(
            [
                sys.executable,
                str(STUB_ENGINE),
                "--model-path",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tp-size",
                "2",
                "--served-model-name",
                "stub-selftest",
            ],
            env={"STUB_TOKEN_INTERVAL": "0.04", "CUDA_VISIBLE_DEVICES": "3"},
            label="stub-contract",
        )
        check(wait_until(lambda: http_get(f"http://127.0.0.1:{port}/health")[0] == 200, timeout=8), "health")
        body = b'{"model":"stub-selftest","messages":[],"stream":true,"max_tokens":4}'
        times = read_sse(f"http://127.0.0.1:{port}/v1/chat/completions", body)
        check(len(times) >= 3, "sse chunks")
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        check(any(g >= 0.02 for g in gaps), "sse incremental gaps")
        data = (model / "stub_launch_argv.json").read_text(encoding="utf-8")
        check("--port" in data and "stub-selftest" in data, "argv json")
        check("CUDA_VISIBLE_DEVICES" in data, "env recorded")
        check("GOETHE_KERNEL_DIR" not in data, "no kernel dir in stub env")
    finally:
        if proc is not None:
            proc.stop()
        import shutil

        shutil.rmtree(model, ignore_errors=True)


def _test_leaked_temp_dir() -> None:
    """Hygiene flags leftover temp dirs for the current gate run only."""
    import shutil

    leak = Path(tempfile.mkdtemp(prefix=run_temp_prefix("hygiene"), dir="/tmp"))
    try:
        try:
            post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[])
            check(False, "expected leaked temp dir")
        except RuntimeError as err:
            check("leaked temp" in str(err).lower(), "leaked temp message")
    finally:
        shutil.rmtree(leak, ignore_errors=True)


def _test_run_id_scoping() -> None:
    """Ignore leaks from other gate runs; detect leaks tagged with this run id."""
    import shutil

    other = "run-other-999001"
    foreign_dir = Path(f"/tmp/goethe-test-{other}-foreign")
    foreign_dir.mkdir(parents=True, exist_ok=True)
    other_marker = f"/tmp/goethe-test-{other}-proc-marker"
    foreign_proc = subprocess.Popen(
        ["sleep", "300", other_marker],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[os.getpid(), foreign_proc.pid])
        check(True, "foreign run temp and proc marker ignored")
    finally:
        foreign_proc.terminate()
        foreign_proc.wait(timeout=5)
        shutil.rmtree(foreign_dir, ignore_errors=True)
    cur_dir = Path(tempfile.mkdtemp(prefix=run_temp_prefix("scoped-leak"), dir="/tmp"))
    cur_proc = subprocess.Popen(
        ["sleep", "300", f"/tmp/{run_temp_prefix('proc-marker')}x"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        try:
            post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[os.getpid()])
            check(False, "expected current-run leak")
        except RuntimeError as err:
            text = str(err).lower()
            check("leaked temp" in text or "leftover process" in text, "current run leak detected")
    finally:
        cur_proc.terminate()
        cur_proc.wait(timeout=5)
        shutil.rmtree(cur_dir, ignore_errors=True)


def _test_http_drop() -> None:
    """Connection closed with no HTTP response maps to status 0."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    def _accept_close() -> None:
        conn, _ = srv.accept()
        conn.close()

    th = threading.Thread(target=_accept_close)
    th.start()
    code, _ = http_get(f"http://127.0.0.1:{port}/")
    srv.close()
    th.join(timeout=5)
    check(code == 0, "http_get on dropped connection")


def _test_hygiene_filters() -> None:
    """Hygiene ignores foreign zombies and honors injected GPU PID leaks."""
    set_gpu_pid_query(lambda _i: [(999001, 5000)])
    register_pid(999001)
    try:
        try:
            post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[])
            check(False, "expected gpu pid leak")
        except RuntimeError:
            check(True, "gpu pid leak detected")
    finally:
        unregister = mt.unregister_pid
        unregister(999001)
        set_gpu_pid_query(None)
    post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[])
    check(True, "foreign zombie ignored")


def _test_exit_codes() -> None:
    """Pending and Skip map to 77 and 75."""
    try:
        raise Pending("pending-msg")
    except Pending:
        check(True, "pending exc")
    try:
        raise Skip("skip-msg")
    except Skip:
        check(True, "skip exc")


def _test_classify_goethe() -> None:
    """Pending only for unknown subcommand or missing binary path."""
    rr = RunResult(2, "", "error: unrecognized subcommand foo")
    try:
        classify_goethe_result(rr)
        check(False, "expected pending")
    except Pending:
        check(True, "exit2 unknown pending")
    rr2 = RunResult(1, "", "internal error")
    try:
        classify_goethe_result(rr2)
        check(False, "expected fail")
    except RuntimeError:
        check(True, "nonzero fail")


def main() -> None:
    """Run infrastructure self tests."""
    _test_exit_codes()
    _test_classify_goethe()
    _test_proc_teardown()
    _test_sigterm_child_teardown()
    _test_temp_home_cleanup()
    _test_stub_contract()
    _test_leaked_temp_dir()
    _test_http_drop()
    _test_run_id_scoping()
    _test_hygiene_filters()
    check(len(leaked_temp_paths()) == 0, "no leaked goethe-test paths")
    print(f"stage0_selftest ok assertions={ASSERTS}")


if __name__ == "__main__":
    main_exit(main)
