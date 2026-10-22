#!/usr/bin/env python3
"""Stage 4: mux drain and forced unload with slow stub streams."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from lib.mesh_test import (
    Proc,
    _acquire_gpu_flocks,
    assert_no_stub_engines,
    home_mesh_env,
    http_get,
    http_post,
    main_exit,
    read_sse,
    require_cli,
    require_gpus,
    run,
    run_mesh,
    seed_installed_model,
    spawn,
    temp_home,
    test_mux_port,
    wait_meshd_socket,
    wait_until,
)

SLOW_TOKENS = 6


def _no_zombies() -> bool:
    proc = __import__("subprocess").run(
        ["ps", "-eo", "stat,pid,ppid="],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return True
    my = os.getpid()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or "Z" not in parts[0]:
            continue
        if int(parts[2]) == my:
            return False
    return True


def _mux_url(home: Path, path: str) -> str:
    return f"http://127.0.0.1:{test_mux_port(home)}{path}"


def _ps_rows(home: Path, env: dict[str, str]) -> list[dict]:
    mesh = require_cli("mesh")
    rr = run([str(mesh), "ps", "--json"], env=env, timeout=30)
    if rr.returncode != 0:
        return []
    return json.loads(rr.stdout.strip())


def _in_flight(home: Path, env: dict[str, str], model: str) -> int:
    for row in _ps_rows(home, env):
        if row.get("model_id") == model:
            return int(row.get("in_flight") or 0)
    return 0


def _slow_stream(
    home: Path,
    token: str,
    out: dict[str, float | int | list[float]],
) -> None:
    body = json.dumps(
        {"model": "a", "messages": [{"role": "user", "content": "s"}], "stream": True, "max_tokens": SLOW_TOKENS}
    ).encode()
    t0 = time.time()
    chunks = read_sse(
        _mux_url(home, "/v1/chat/completions"),
        body,
        headers={"Authorization": f"Bearer {token}"},
        max_chunks=SLOW_TOKENS + 2,
    )
    out["chunks"] = chunks
    out["end"] = time.time()
    out["start"] = t0


def main() -> None:
    """Graceful drain, forced unload, and idle unload via mux."""
    mesh = require_cli("mesh")
    require_cli("meshd")
    gpu = require_gpus(1)[0]
    release = _acquire_gpu_flocks([gpu])
    md: Proc | None = None
    try:
        with temp_home(stub_python=True) as home:
            model_a = home / "models" / "org__a"
            model_a.mkdir(parents=True)
            (model_a / "mesh_card.toml").write_text(
                'model_id="a"\ncontext_length=4096\nmax_running_requests=4\n\n'
                '[env]\nSTUB_TOKEN_INTERVAL="1.0"\n',
                encoding="utf-8",
            )
            seed_installed_model(home, "a", str(model_a.relative_to(home)), 10.0, 1000, 4096, 4, 1)
            md = spawn([str(require_cli("meshd"))], env=home_mesh_env(home), label="meshd")
            wait_meshd_socket(home)
            env = {**home_mesh_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
            rr_key = run_mesh([str(mesh), "key", "create", "--name", "d", "--raw"], env=env)
            token = [ln for ln in rr_key.stdout.splitlines() if ln.startswith("mesh_sk_")][0]
            hdr = {"Authorization": f"Bearer {token}"}
            chat_body = json.dumps(
                {"model": "a", "messages": [{"role": "user", "content": "n"}], "max_tokens": 1}
            ).encode()

            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=120)
            stream_out: dict[str, float | int | list[float]] = {}
            st = threading.Thread(target=_slow_stream, args=(home, token, stream_out))
            st.start()
            inflight_seen = False

            def _streaming_inflight() -> bool:
                nonlocal inflight_seen
                if _in_flight(home, env, "a") >= 1:
                    inflight_seen = True
                return inflight_seen

            if not wait_until(_streaming_inflight, timeout=12.0):
                st.join(timeout=30)
                raise RuntimeError("in_flight never reached 1")
            unload_finish_at: dict[str, float] = {}
            unload_err: list[str] = []

            def _graceful_unload() -> None:
                try:
                    run_mesh([str(mesh), "unload", "a"], env=env, timeout=180)
                except Exception as exc:
                    unload_err.append(str(exc))
                finally:
                    unload_finish_at["t"] = time.time()

            ut = threading.Thread(target=_graceful_unload)
            ut.start()
            time.sleep(0.3)
            code, body = http_post(_mux_url(home, "/v1/chat/completions"), chat_body, headers=hdr)
            if code != 503 or b"Model is draining" not in body:
                raise RuntimeError(f"draining chat st={code} body={body[:200]!r}")
            code, mbody = http_get(_mux_url(home, "/v1/models"), headers=hdr)
            if code != 200:
                raise RuntimeError(f"models st={code}")
            ids = {m.get("id") for m in json.loads(mbody.decode("utf-8")).get("data", [])}
            if "a" in ids:
                raise RuntimeError("draining model still listed")
            if not inflight_seen:
                raise RuntimeError("in_flight not observed while streaming")
            st.join(timeout=30)
            ut.join(timeout=180)
            if unload_err:
                raise RuntimeError(unload_err[0])
            chunks = stream_out.get("chunks")
            if not isinstance(chunks, list) or len(chunks) < SLOW_TOKENS:
                raise RuntimeError(f"graceful stream truncated len={len(chunks) if isinstance(chunks, list) else 0}")
            stream_end = float(stream_out.get("end", 0))
            ufin = unload_finish_at.get("t", 0.0)
            if ufin < stream_end - 0.5:
                raise RuntimeError(f"unload early by {stream_end - 0.5 - ufin:.2f}s")
            assert_no_stub_engines(home)
            if not wait_until(_no_zombies, timeout=2.0):
                raise RuntimeError("zombies after graceful drain")

            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=120)
            force_out: dict[str, float | int | list[float]] = {}
            fst = threading.Thread(target=_slow_stream, args=(home, token, force_out))
            fst.start()
            if not wait_until(lambda: _in_flight(home, env, "a") >= 1, timeout=8.0):
                fst.join(timeout=30)
                raise RuntimeError("force stream never started")
            t_force = time.time()
            run_mesh([str(mesh), "unload", "a", "--force"], env=env, timeout=30)
            if time.time() - t_force >= 2.0:
                raise RuntimeError("force unload exceeded 2s")
            fst.join(timeout=15)
            fchunks = force_out.get("chunks")
            flen = len(fchunks) if isinstance(fchunks, list) else 0
            if flen >= SLOW_TOKENS:
                raise RuntimeError(f"force stream completed all chunks ({flen})")
            assert_no_stub_engines(home)

            run_mesh([str(mesh), "load", "a", "--gpus", str(gpu)], env=env, timeout=120)
            t_idle = time.time()
            run_mesh([str(mesh), "unload", "a"], env=env, timeout=30)
            if time.time() - t_idle >= 2.0:
                raise RuntimeError("idle unload exceeded 2s")
            assert_no_stub_engines(home)
            md.stop()
            md = None
            print("stage4_drain_stub ok")
    finally:
        if md is not None:
            md.stop()
        release()


if __name__ == "__main__":
    main_exit(main)
