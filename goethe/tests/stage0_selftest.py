#!/usr/bin/env python3
"""Stage 0 self-test for goethe gate infrastructure."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from lib import goethe_test as mt
from lib.goethe_test import (
    STUB_ENGINE,
    TESTS_DIR,
    Pending,
    Proc,
    RunResult,
    classify_goethe_result,
    free_port,
    home_zombies,
    http_get,
    http_post,
    leaked_temp_paths,
    main_exit,
    post_test_hygiene,
    reap_run_artifacts,
    register_pid,
    run,
    run_temp_prefix,
    set_gpu_pid_query,
    spawn,
    sse_events,
    stub_launch,
    stub_requests,
    stub_signals,
    temp_home,
    wait_until,
)

ASSERTS = 0


def check(cond: bool, msg: str) -> None:
    """Increment assertion count or raise."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def _raised(fn: Callable[[], object], exc: type[BaseException] = RuntimeError) -> str | None:
    """Run fn and return the text of the expected exception, or None when it did not raise."""
    try:
        fn()
    except exc as err:
        return str(err) or type(err).__name__
    return None


def _stub_cmd(model: Path, port: int, *extra: str) -> list[str]:
    return [sys.executable, str(STUB_ENGINE), "--model-path", str(model), "--port", str(port), *extra]


def _test_run_stdin_closed() -> None:
    """run() hands children /dev/null on stdin so nothing can block on a terminal prompt."""
    probe = "import os, stat; s = os.fstat(0); print(stat.S_ISCHR(s.st_mode) and s.st_rdev == os.makedev(1, 3))"
    check(run([sys.executable, "-c", probe], timeout=10).stdout.strip() == "True", "run() stdin is /dev/null")


def _test_proc_teardown() -> None:
    """SIGTERM-ignoring process group receives SIGKILL."""
    teardown_dir = tempfile.mkdtemp(prefix=run_temp_prefix("proc-teardown"), dir="/tmp")
    try:
        proc = spawn(_stub_cmd(Path(teardown_dir), free_port()), env={"STUB_IGNORE_SIGTERM": "1"}, label="sigkill")
        check(wait_until(lambda: (Path(teardown_dir) / "stub_launch_argv.json").is_file(), timeout=10), "stub up")
        proc.stop(sigterm_timeout=0.5)
        check(proc.popen.poll() is not None, "stub group did not exit after SIGKILL escalation")
    finally:
        shutil.rmtree(teardown_dir, ignore_errors=True)


def _test_sigterm_child_teardown() -> None:
    """SIGTERM on a test child stops its Proc group and temp home."""
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
    (home / "m").mkdir()
    cmd = [sys.executable, str(STUB_ENGINE), "--model-path", str(home / "m"), "--port", str(free_port())]
    proc = spawn(cmd, label="sigterm-grandchild")
    state.write_text(json.dumps({{"home": str(home), "pgid": proc.pgid}}), encoding="utf-8")
    time.sleep(3600)
"""
    child = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_until(state_file.is_file, timeout=10.0):
            raise RuntimeError("child did not write state")
        data = json.loads(state_file.read_text(encoding="utf-8"))
        os.kill(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
        check(not Path(data["home"]).exists(), "temp home removed after SIGTERM")
        check(not mt._group_members(int(data["pgid"])), "grandchild process group gone")
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        shutil.rmtree(state_dir, ignore_errors=True)


def _test_temp_home_cleanup() -> None:
    """temp_home removes directory even when an exception is raised."""
    path_holder: list[str] = []

    def body() -> None:
        with temp_home(stub_python=False) as home:
            path_holder.append(str(home))
            raise RuntimeError("forced")

    check(_raised(body) == "forced", "exception propagates out of temp_home")
    check(not Path(path_holder[0]).exists(), "temp_home not removed on exception")


def _test_temp_home_strict_teardown() -> None:
    """Leaving a stub engine running fails the temp_home exit, and the stub dies with its home."""
    holder: list[Proc] = []

    def body() -> None:
        with temp_home() as home:
            model = home / "models" / "m"
            model.mkdir()
            holder.append(spawn(_stub_cmd(model, free_port()), label="leaky-stub"))
            check(wait_until(lambda: (model / "stub_launch_argv.json").is_file(), timeout=10), "leaky stub up")

    err = _raised(body)
    check(err is not None and "outlived temp home" in err, f"strict teardown flags leftover stub: {err}")
    check(wait_until(lambda: holder[0].popen.poll() is not None, timeout=10), "stub exits once its home is gone")
    holder[0].stop()


def _test_stub_contract() -> None:
    """Stub HTTP/1.1 OpenAI schema, api key, knobs, request and signal records, TP rank in the leader group."""
    port = free_port()
    model = Path(tempfile.mkdtemp(prefix=run_temp_prefix("stub"), dir="/tmp"))
    proc: Proc | None = None
    try:
        proc = spawn(
            _stub_cmd(model, port, "--tp-size", "2", "--served-model-name", "stub-selftest", "--api-key", "k-1"),
            env={"STUB_TOKEN_INTERVAL": "0.04", "STUB_IGNORE_SIGTERM": "1", "CUDA_VISIBLE_DEVICES": "3"},
            label="stub-contract",
        )
        base = f"http://127.0.0.1:{port}"
        check(wait_until(lambda: http_get(f"{base}/health")[0] == 200, timeout=8), "health open without key")
        check(http_get(f"{base}/v1/models")[0] == 401, "api key enforced on /v1/models")
        auth = {"Authorization": "Bearer k-1"}
        code, body = http_get(f"{base}/v1/models", headers=auth)
        check(code == 200 and json.loads(body)["data"][0]["id"] == "stub-selftest", "models with key")
        req = {"model": "stub-selftest", "messages": [], "stream": True, "max_tokens": 4}
        req["stream_options"] = {"include_usage": True}
        events = sse_events(f"{base}/v1/chat/completions", json.dumps(req).encode(), headers=auth)
        payloads = [e[1] for e in events]
        check(len(payloads) == 7 and payloads[-1] == b"[DONE]", f"4 tokens, finish, usage, DONE: {len(payloads)}")
        first = json.loads(payloads[0])
        delta = first["choices"][0]["delta"]
        check(first["object"] == "chat.completion.chunk" and delta["role"] == "assistant", "chunk schema")
        check(json.loads(payloads[5])["usage"]["completion_tokens"] == 4, "usage chunk")
        gaps = [events[i + 1][0] - events[i][0] for i in range(3)]
        check(all(g >= 0.03 for g in gaps), f"incremental sse gaps {gaps}")
        code, body = http_post(f"{base}/v1/embeddings", b'{"model":"m","input":["a","b"]}', headers=auth)
        check(code == 200 and len(json.loads(body)["data"]) == 2, "embeddings")
        code, body = http_post(f"{base}/v1/completions", b'{"model":"m","prompt":"x","max_tokens":2}', headers=auth)
        check(code == 200 and json.loads(body)["object"] == "text_completion", "completions")
        launch = stub_launch(model)
        check(launch["knobs"].get("STUB_IGNORE_SIGTERM") == "1", "knobs recorded")
        check(launch["env"].get("CUDA_VISIBLE_DEVICES") == "3", "env recorded")
        check(len(launch["tp_pids"]) == 1 and mt._group_members(proc.pgid) == sorted([proc.pgid, *launch["tp_pids"]]),
              "TP rank shares the leader's process group")
        seen = stub_requests(model)
        check(any(r["status"] == 401 and "authorization" not in r["headers"] for r in seen), "401 request recorded")
        check(any(r["headers"].get("authorization") == "Bearer k-1" for r in seen), "auth header recorded")
        os.killpg(proc.pgid, signal.SIGTERM)
        check(wait_until(lambda: bool(stub_signals(model)), timeout=5), "signal recorded")
        check(stub_signals(model)[0]["ignored"] and proc.popen.poll() is None, "TERM ignored by request")
    finally:
        if proc is not None:
            proc.stop(sigterm_timeout=0.5)
        shutil.rmtree(model, ignore_errors=True)


def _test_home_zombies() -> None:
    """A defunct child of a process that references the temp home is reported."""
    with temp_home(stub_python=False, tag="zombie") as home:
        code = "import os, time\nif os.fork() == 0:\n    os._exit(0)\ntime.sleep(60)\n"
        parent = spawn([sys.executable, "-c", code], env={"GOETHE_HOME": str(home)}, label="zombie-parent")
        check(wait_until(lambda: bool(home_zombies(home)), timeout=10), "zombie under temp home detected")
        parent.stop()
        check(wait_until(lambda: not home_zombies(home), timeout=5), "zombie gone with its parent")


def _test_reap_artifacts() -> None:
    """Groups recorded by a test that died by SIGKILL are reaped from its artifact."""
    child_code = f"""
import sys, time
sys.path.insert(0, {json.dumps(str(TESTS_DIR))})
from lib.goethe_test import spawn
p = spawn(["sleep", "300"], label="orphan")
print(p.pgid, flush=True)
time.sleep(300)
"""
    env = {**os.environ, "GOETHE_GATE_RUN_ID": mt.gate_run_id()}
    child = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True, env=env)
    try:
        orphan = int(child.stdout.readline().strip())
        child.kill()
        child.wait(timeout=5)
        check(bool(mt._group_members(orphan)), "orphan group alive after test SIGKILL")
        leftovers, _pids = reap_run_artifacts()
        check(any(f"pgid {orphan}" in ln for ln in leftovers), f"orphan reported: {leftovers}")
        check(not mt._group_members(orphan), "orphan group reaped")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def _test_leaked_temp_dir() -> None:
    """Hygiene flags leftover temp dirs for the current gate run only."""
    leak = Path(tempfile.mkdtemp(prefix=run_temp_prefix("hygiene"), dir="/tmp"))
    try:
        err = _raised(lambda: post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[]))
        check(err is not None and "leaked temp" in err.lower(), f"leaked temp message: {err}")
    finally:
        shutil.rmtree(leak, ignore_errors=True)


def _marker_proc(marker: str) -> subprocess.Popen[bytes]:
    """Long-lived process whose cmdline carries marker."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", marker])


def _test_run_id_scoping() -> None:
    """Ignore leaks from other gate runs; detect processes and temp dirs tagged with this run id."""
    other = "run-other-999001"
    foreign_dir = Path(f"/tmp/goethe-test-{other}-foreign")
    foreign_dir.mkdir(parents=True, exist_ok=True)
    foreign_proc = _marker_proc(f"/tmp/goethe-test-{other}-proc-marker")
    try:
        check(wait_until(lambda: "proc-marker" in mt._cmdline(foreign_proc.pid), timeout=5), "foreign marker up")
        post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[os.getpid()])
    finally:
        foreign_proc.terminate()
        foreign_proc.wait(timeout=5)
        shutil.rmtree(foreign_dir, ignore_errors=True)
    cur_proc = _marker_proc(f"/tmp/{run_temp_prefix('proc-marker')}x")
    try:
        check(wait_until(lambda: "proc-marker" in mt._cmdline(cur_proc.pid), timeout=5), "current marker up")
        err = _raised(lambda: post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[os.getpid()]))
        check(err is not None and f"leftover process {cur_proc.pid}" in err, f"current run process leak: {err}")
    finally:
        cur_proc.terminate()
        cur_proc.wait(timeout=5)


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
    """Hygiene honors injected GPU PID leaks and passes once the pid is untracked."""
    set_gpu_pid_query(lambda _i: [(999001, 5000)])
    register_pid(999001)
    try:
        err = _raised(lambda: post_test_hygiene([3], {}, gate_pid=os.getpid(), test_pids=[]))
        check(err is not None and "still on GPU 3" in err, f"gpu pid leak detected: {err}")
    finally:
        mt.unregister_pid(999001)
        set_gpu_pid_query(None)
    post_test_hygiene([], {}, gate_pid=os.getpid(), test_pids=[])


def _test_exit_codes() -> None:
    """main_exit maps Pending to 77 and Skip to 75."""
    for name, code in (("Pending", 77), ("Skip", 75)):
        src = (
            f"import sys; sys.path.insert(0, {json.dumps(str(TESTS_DIR))})\n"
            f"from lib.goethe_test import main_exit, {name}\n"
            f"def f():\n    raise {name}('x')\n"
            "main_exit(f)\n"
        )
        check(run([sys.executable, "-c", src], timeout=30).returncode == code, f"{name} exits {code}")


def _test_classify_goethe() -> None:
    """Pending only for unknown subcommand; any other nonzero exit fails."""
    check(_raised(lambda: classify_goethe_result(RunResult(2, "", "error: unrecognized subcommand foo")), Pending)
          is not None, "exit2 unknown pending")
    err = _raised(lambda: classify_goethe_result(RunResult(1, "", "internal error")))
    check(err is not None and "goethe exit 1" in err, f"nonzero fail: {err}")


def main() -> None:
    """Run infrastructure self tests."""
    _test_exit_codes()
    _test_classify_goethe()
    _test_run_stdin_closed()
    _test_proc_teardown()
    _test_sigterm_child_teardown()
    _test_temp_home_cleanup()
    _test_temp_home_strict_teardown()
    _test_stub_contract()
    _test_home_zombies()
    _test_reap_artifacts()
    _test_leaked_temp_dir()
    _test_http_drop()
    _test_run_id_scoping()
    _test_hygiene_filters()
    check(len(leaked_temp_paths()) == 0, "no leaked goethe-test paths")
    print(f"stage0_selftest ok assertions={ASSERTS}")


if __name__ == "__main__":
    main_exit(main)
