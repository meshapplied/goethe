#!/usr/bin/env python3
"""Stage 4: mux drain, and forced unload cutting slow stub streams cleanly through the mux and the gateway."""

from __future__ import annotations

import http.client
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from lib.goethe_test import (
    assert_no_stub_engines,
    assert_no_zombies,
    ensure_nginx,
    gpu_flock,
    home_goethe_env,
    http_get,
    http_post,
    main_exit,
    make_tls_certs,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sse_events,
    start_goethed,
    stub_launch,
    temp_home,
    test_mux_port,
    test_public_port,
    wait_until,
    write_card,
)

SLOW_TOKENS = 6
SLOW_BODY = json.dumps(
    {"model": "a", "messages": [{"role": "user", "content": "s"}], "stream": True, "max_tokens": SLOW_TOKENS}
).encode()


def _mux_url(home: Path, path: str) -> str:
    return f"http://127.0.0.1:{test_mux_port(home)}{path}"


def _in_flight(goethe: Path, env: dict[str, str], model: str) -> int:
    """in_flight of model from goethe ps --json (0 when absent)."""
    rr = run([str(goethe), "ps", "--json"], env=env, timeout=30)
    rows = json.loads(rr.stdout.strip() or "[]") if rr.returncode == 0 else []
    return next((int(r.get("in_flight") or 0) for r in rows if r.get("model_id") == model), 0)


def _slow_stream(home: Path, token: str, out: dict[str, Any]) -> None:
    """Stream SLOW_TOKENS tokens of model a through the mux and record the events."""
    out["events"] = sse_events(
        _mux_url(home, "/v1/chat/completions"),
        SLOW_BODY,
        headers={"Authorization": f"Bearer {token}"},
        max_chunks=SLOW_TOKENS + 4,
    )
    out["end"] = time.time()


def _strict_stream(home: Path, token: str, out: dict[str, Any]) -> None:
    """Stream through the mux keeping every data payload and any transport error (a reset or a truncated body)."""
    conn = http.client.HTTPConnection("127.0.0.1", test_mux_port(home), timeout=60)
    payloads: list[bytes] = []
    out["error"] = None
    try:
        hdrs = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body=SLOW_BODY, headers=hdrs)
        resp = conn.getresponse()
        while line := resp.readline():
            if line.startswith(b"data:"):
                payloads.append(line[5:].strip())
    except (OSError, http.client.HTTPException) as exc:
        out["error"] = repr(exc)
    finally:
        conn.close()
        out["payloads"] = payloads


def _expect_cut(payloads: list[bytes], via: str) -> None:
    """A stream cut by unload --force ends with the engine_stopped error event and [DONE], before all its tokens."""
    try:
        err = json.loads(payloads[-2]).get("error", {}) if len(payloads) >= 2 else {}
    except ValueError:
        err = {}
    if payloads[-1:] != [b"[DONE]"] or err.get("code") != "engine_stopped" or err.get("type") != "unavailable_error":
        raise RuntimeError(f"{via}: a force-cut stream must end with engine_stopped and [DONE]: {payloads[-3:]!r}")
    if len(payloads) - 2 >= SLOW_TOKENS:
        raise RuntimeError(f"{via}: force unload let the stream complete ({len(payloads) - 2} events)")


def _curl_stream(home: Path, token: str) -> subprocess.Popen[str]:
    """Stream the slow request through the TLS gateway with curl over HTTP/2, the key kept in a 0600 header file."""
    url = f"https://127.0.0.1:{test_public_port(home)}/v1/chat/completions"
    hdr_file, body_file = home / "curl-headers.txt", home / "curl-body.json"
    hdr_file.write_text(f"Authorization: Bearer {token}\nContent-Type: application/json\n", encoding="utf-8")
    hdr_file.chmod(0o600)
    body_file.write_bytes(SLOW_BODY)
    cmd = ["curl", "-sS", "-k", "-N", "--http2", "--max-time", "60", "-w", "\nhttp_version=%{http_version}\n"]
    cmd += ["-H", f"@{hdr_file}"]
    return subprocess.Popen(
        [*cmd, "--data-binary", f"@{body_file}", url],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def main() -> None:
    """Graceful drain, forced unload via the mux and via the gateway, and idle unload."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        model_a = home / "models" / "org__a"
        write_card(model_a, "a", env={"STUB_TOKEN_INTERVAL": "1.0"})
        seed_installed_model(home, "a", "models/org__a", 10.0, 1000, 4096, 4, 1)
        ensure_nginx(home)
        make_tls_certs(home)
        md = start_goethed(home)
        env = home_goethe_env(home)
        try:
            _scenarios(goethe, home, env, model_a, gpu)
        finally:
            run([str(goethe), "unload", "--all", "--force"], env=env, timeout=60)
            md.stop()
        print("stage4_drain_stub ok")


def _scenarios(goethe: Path, home: Path, env: dict[str, str], model_a: Path, gpu: int) -> None:
    """The drain, force and idle unload scenarios in order against one goethed and its gateway."""
    load = [str(goethe), "load", "a", "--gpus", str(gpu), "--yes"]
    rr_key = run_goethe([str(goethe), "key", "create", "--name", "d", "--raw"], env=env)
    token = next(ln for ln in rr_key.stdout.splitlines() if ln.startswith("goethe_sk_"))
    hdr = {"Authorization": f"Bearer {token}"}
    chat_body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "n"}], "max_tokens": 1}).encode()

    run_goethe(load, env=env, timeout=120)
    if stub_launch(model_a)["knobs"].get("STUB_TOKEN_INTERVAL") != "1.0":
        raise RuntimeError("engine did not receive STUB_TOKEN_INTERVAL from the card [env] table")
    stream_out: dict[str, Any] = {}
    st = threading.Thread(target=_slow_stream, args=(home, token, stream_out))
    st.start()
    if not wait_until(lambda: _in_flight(goethe, env, "a") >= 1, timeout=12.0):
        st.join(timeout=30)
        raise RuntimeError("in_flight never reached 1")
    unload_finish_at: dict[str, float] = {}
    unload_err: list[str] = []

    def _graceful_unload() -> None:
        try:
            run_goethe([str(goethe), "unload", "a"], env=env, timeout=180)
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
    if "a" in {m.get("id") for m in json.loads(mbody.decode("utf-8")).get("data", [])}:
        raise RuntimeError("draining model still listed")
    st.join(timeout=30)
    ut.join(timeout=180)
    if unload_err:
        raise RuntimeError(unload_err[0])
    payloads = [p for _t, p in stream_out.get("events", [])]
    if len(payloads) < SLOW_TOKENS or payloads[-1] != b"[DONE]":
        raise RuntimeError(f"graceful stream truncated: {len(payloads)} events, last {payloads[-1:]!r}")
    if unload_finish_at.get("t", 0.0) < float(stream_out["end"]) - 0.5:
        raise RuntimeError("unload finished before the in-flight stream completed")
    assert_no_stub_engines(home)
    assert_no_zombies(home)

    run_goethe(load, env=env, timeout=120)
    force_out: dict[str, Any] = {}
    fst = threading.Thread(target=_strict_stream, args=(home, token, force_out))
    fst.start()
    if not wait_until(lambda: _in_flight(goethe, env, "a") >= 1, timeout=8.0):
        fst.join(timeout=30)
        raise RuntimeError("force stream never started")
    t_force = time.time()
    run_goethe([str(goethe), "unload", "a", "--force"], env=env, timeout=30)
    if time.time() - t_force >= 2.0:
        raise RuntimeError("force unload exceeded 2s")
    fst.join(timeout=15)
    if force_out.get("error"):
        raise RuntimeError(f"mux: the force-cut stream ended with a transport error: {force_out['error']}")
    _expect_cut(force_out.get("payloads", []), "mux")
    assert_no_stub_engines(home)
    assert_no_zombies(home)

    run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=env)
    health = f"https://127.0.0.1:{test_public_port(home)}/health"
    if not wait_until(lambda: http_get(health, timeout=3)[0] == 200, timeout=20):
        raise RuntimeError(f"gateway did not answer /health: {http_get(health, timeout=3)[0]}")
    run_goethe(load, env=env, timeout=120)
    curl = _curl_stream(home, token)
    try:
        if not wait_until(lambda: _in_flight(goethe, env, "a") >= 1, timeout=10.0):
            raise RuntimeError("gateway force stream never started")
        run_goethe([str(goethe), "unload", "a", "--force"], env=env, timeout=30)
        out, err = curl.communicate(timeout=30)
    finally:
        if curl.poll() is None:
            curl.kill()
            curl.wait(timeout=10)
    if curl.returncode != 0 or "http_version=2" not in out:
        raise RuntimeError(f"gateway: curl exit {curl.returncode} (92 HTTP/2 reset, 56 receive failure): {err[-300:]}")
    _expect_cut([ln[5:].strip().encode() for ln in out.splitlines() if ln.startswith("data:")], "gateway")
    assert_no_stub_engines(home)
    assert_no_zombies(home)

    run_goethe(load, env=env, timeout=120)
    t_idle = time.time()
    run_goethe([str(goethe), "unload", "a"], env=env, timeout=30)
    if time.time() - t_idle >= 2.0:
        raise RuntimeError("idle unload exceeded 2s")
    assert_no_stub_engines(home)


if __name__ == "__main__":
    main_exit(main)
