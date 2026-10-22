#!/usr/bin/env python3
"""Stage 5 nginx gateway and security gate."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lib.goethe_test import (
    _acquire_gpu_flocks,
    ensure_nginx,
    home_goethe_env,
    home_process_tree,
    http_get,
    http_post,
    main_exit,
    make_tls_certs,
    proc_cmdline,
    read_sse,
    require_cli,
    require_gpus,
    run_goethe,
    seed_installed_model,
    sha256_hex,
    sqlite_query,
    sse_events,
    start_goethed,
    stub_launch,
    stub_requests,
    temp_home,
    test_mux_port,
    test_public_port,
    wait_until,
    write_card,
)

ASSERTS = 0
SOFT: list[str] = []
TOKEN_INTERVAL = "0.2"
HOL_TOKENS = 24


def check(cond: bool, msg: str) -> None:
    """Record one assertion."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def soft(cond: bool, msg: str) -> None:
    """Record an assertion reported at the end without stopping the remaining checks."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        SOFT.append(msg)
        print(f"SOFT FAIL: {msg}", flush=True)


def _token(out: str) -> str:
    return next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))


def _engine_key_checks(home: Path, model_dir: Path, client_tokens: list[str]) -> None:
    """Contract 3: the engine runs with --api-key and only ever sees Bearer <engine_key>, never client keys."""
    row = sqlite_query(home / "state.db", "SELECT engine_key FROM loaded_instances WHERE model_id='ma' LIMIT 1")
    engine_key = row[0][0] if row else None
    soft(bool(engine_key) and len(str(engine_key)) == 32, f"loaded_instances.engine_key is 32 hex: {engine_key!r}")
    soft(stub_launch(model_dir)["parsed"].get("api_key") == engine_key, "engine launched with --api-key engine_key")
    chats = [r for r in stub_requests(model_dir) if r["path"].startswith("/v1/chat/completions")]
    check(bool(chats), "engine saw the routed chat requests")
    seen = json.dumps([r["headers"] for r in stub_requests(model_dir)])
    check(not any(t in seen for t in client_tokens), "client API keys never reach the engine")
    soft(all(r["headers"].get("authorization") == f"Bearer {engine_key}" for r in chats), "mux sends engine key")


def _stop_nginx(home: Path) -> None:
    """Fast-stop the gateway master goethed started (TERM lets it take its workers down)."""
    pid_path = home / "nginx" / "nginx.pid"
    check(pid_path.is_file(), "nginx.pid present while the gateway runs")
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    check(str(home) in proc_cmdline(pid), "nginx master belongs to this temp home")
    os.kill(pid, signal.SIGTERM)
    check(
        wait_until(lambda: not any("nginx" in proc_cmdline(p) for p in home_process_tree(home)), timeout=10),
        "nginx master and workers gone",
    )


def main() -> None:
    """Run default-deny gateway checks with stub engines and TLS."""
    goethe = require_cli("goethe")
    gpus = require_gpus(1)
    gpu = gpus[0]
    g2 = gpus[1] if len(gpus) > 1 else gpu
    release_gpu = _acquire_gpu_flocks(sorted({gpu, g2}))
    try:
        with temp_home(stub_python=True) as home:
            _gateway(goethe, home, gpu, g2)
    finally:
        release_gpu()
    if SOFT:
        raise RuntimeError(f"{len(SOFT)} contract checks failed: " + " | ".join(SOFT))
    print(f"stage5_gateway ok assertions={ASSERTS}")


def _gateway(goethe: Path, home: Path, gpu: int, g2: int) -> None:
    """All gateway scenarios against one goethed and its nginx."""
    base = f"https://127.0.0.1:{test_public_port(home)}"
    ensure_nginx(home)
    make_tls_certs(home)
    ma, mb = home / "models" / "org__ma", home / "models" / "org__mb"
    for model_dir, mid in ((ma, "ma"), (mb, "mb")):
        write_card(model_dir, mid, 512, 2, env={"STUB_TOKEN_INTERVAL": TOKEN_INTERVAL})
        seed_installed_model(home, mid, f"models/{model_dir.name}", 5.0, 512, 512, 2, 1)
    goethed = start_goethed(home)
    env = home_goethe_env(home)

    def goethe_ok(*args: str, timeout: float = 120) -> str:
        return run_goethe([str(goethe), *args], env=env, timeout=timeout).stdout

    try:
        goethe_ok("load", "ma", "--gpus", str(gpu), "--yes", timeout=300)
        goethe_ok("load", "mb", "--gpus", str(g2), "--yes", timeout=300)
        check(stub_launch(ma)["knobs"].get("STUB_TOKEN_INTERVAL") == TOKEN_INTERVAL, "token interval knob active")
        check(wait_until(lambda: http_get(f"{base}/v1/models")[0] != 0, timeout=20), "gateway listening")
        code, _ = http_get(f"{base}/v1/models")
        check(code == 403, f"empty allowlist denies with 403, got {code}")
        goethe_ok("allowlist", "add", "127.0.0.1/32")
        code, body = http_get(f"{base}/health")
        check(code == 200 and json.loads(body.decode("utf-8")).get("status") == "ok", "health no key")
        check(http_get(f"{base}/health/ready?model=ma")[0] == 200, "health ready running")
        check(http_get(f"{base}/health/ready?model=not-loaded-xyz")[0] == 503, "health ready missing")
        check(http_get(f"{base}/health/ready")[0] == 400, "health ready no param")
        check(http_get(f"{base}/v1/models")[0] == 401, "no key unauthorized")
        check(http_get(f"{base}/v1/models", headers={"Authorization": "Bearer goethe_sk_bad"})[0] == 401, "wrong key")
        token = _token(goethe_ok("key", "create", "--name", "t", "--raw"))
        keys_conf = home / "nginx" / "api_keys.conf"
        check("goethe_sk_" not in keys_conf.read_text(encoding="utf-8"), "api_keys.conf no raw token")
        check((keys_conf.stat().st_mode & 0o777) == 0o600, "api_keys.conf mode 0600")
        key_rows = sqlite_query(home / "state.db", "SELECT key_id, token_hash, raw_prefix FROM api_keys")
        key_id, thash, prefix = key_rows[0]
        check(thash == sha256_hex(token), "sha256 stored")
        check(prefix != token and len(str(prefix)) < len(token), "raw_prefix not full key")
        auth = {"Authorization": f"Bearer {token}"}
        code, body = http_get(f"{base}/v1/models", headers=auth)
        check(code == 200 and {m.get("id") for m in json.loads(body).get("data", [])} == {"ma", "mb"}, "models list")
        adm_token = _token(goethe_ok("key", "create", "--name", "adm", "--admin", "--raw"))
        code, body = http_get(f"{base}/health/admin", headers={"Authorization": f"Bearer {adm_token}"})
        check(code == 200, "admin health")
        check({"gpus", "instances", "inactive_models"} <= set(json.loads(body.decode("utf-8"))), "admin health keys")
        check(http_get(f"{base}/health/admin", headers=auth)[0] == 404, "admin health denied")
        sc_token = _token(goethe_ok("key", "create", "--name", "sc", "--raw", "--models", "ma"))
        code, body = http_get(f"{base}/v1/models", headers={"Authorization": f"Bearer {sc_token}"})
        check({m.get("id") for m in json.loads(body.decode("utf-8")).get("data", [])} == {"ma"}, "scoped models")

        def chat(model: str, tok: str, max_tokens: int = 2) -> tuple[int, bytes]:
            msg = {"model": model, "messages": [{"role": "user", "content": "x"}], "max_tokens": max_tokens}
            hdr = {"Authorization": f"Bearer {tok}"}
            return http_post(f"{base}/v1/chat/completions", json.dumps(msg).encode(), hdr)

        code, cbody = chat("ma", token)
        check(code == 200 and b"model=ma" in cbody, "route ma")
        code, cbody = chat("mb", token)
        check(code == 200 and b"model=mb" in cbody, "route mb")
        code, cbody = chat("mb", sc_token, 1)
        check(code == 404 and b"Model Not Found" in cbody, "scoped model 404")
        _engine_key_checks(home, ma, [token, adm_token, sc_token])
        goethe_ok("alias", "set", "chat", "ma")
        code, cbody = chat("chat", token)
        check(code == 200 and b"model=ma" in cbody, "alias routes")
        goethe_ok("alias", "remove", "chat")
        check(chat("missing-model-xyz", token, 1)[0] == 404, "unknown model")
        rpm_token = _token(goethe_ok("key", "create", "--name", "rpm3", "--raw", "--rpm", "3"))
        for _ in range(3):
            check(chat("ma", rpm_token, 1)[0] == 200, "rpm chat under limit")
        check(chat("ma", rpm_token, 1)[0] == 429, "rpm chat 429")
        token2 = _token(goethe_ok("key", "create", "--name", "burst", "--raw"))
        with ThreadPoolExecutor(max_workers=16) as pool:
            burst = list(
                pool.map(lambda _: http_get(f"{base}/v1/models", {"Authorization": f"Bearer {token2}"})[0], range(120))
            )
        check(429 in burst, "rate limit 429")
        stream_token = _token(goethe_ok("key", "create", "--name", "stream", "--raw"))
        goethe_ok("key", "revoke", str(key_id))
        check(http_get(f"{base}/v1/models", headers=auth)[0] == 401, "revoked key")
        check(http_get(f"http://127.0.0.1:{test_public_port(home)}/v1/models")[0] == 0, "plain http fails")
        big = b"x" * (11 * 1024 * 1024)
        check(http_post(f"{base}/v1/chat/completions", big, {"Authorization": f"Bearer {token2}"})[0] == 413, "413")
        _streams(base, stream_token, home, env, goethe)
        _stop_nginx(home)
        down_token = _token(goethe_ok("key", "create", "--name", "down", "--raw"))
        check("goethe_sk_" not in keys_conf.read_text(encoding="utf-8"), "hash-only api_keys")
        code, _ = http_get(
            f"http://127.0.0.1:{test_mux_port(home)}/v1/models", headers={"Authorization": f"Bearer {down_token}"}
        )
        check(code == 200, "mux auth with gateway down")
    finally:
        run_goethe([str(goethe), "unload", "--all"], env=env, timeout=120)
        goethed.stop()
    check(wait_until(lambda: not home_process_tree(home), timeout=10), "no engine or nginx process left")


def _streams(base: str, stream_token: str, home: Path, env: dict[str, str], goethe: Path) -> None:
    """Concurrent streams to two engines overlap (no head-of-line blocking); a stream survives nginx reload."""
    hdr = {"Authorization": f"Bearer {stream_token}"}
    spans: dict[str, list[float]] = {}

    def _stream(model: str) -> None:
        body = json.dumps({"model": model, "messages": [], "stream": True, "max_tokens": HOL_TOKENS}).encode()
        events = sse_events(f"{base}/v1/chat/completions", body, headers=hdr, max_chunks=HOL_TOKENS + 4)
        if len(events) >= HOL_TOKENS:
            spans[model] = [events[0][0], events[HOL_TOKENS - 1][0]]

    threads = [threading.Thread(target=_stream, args=(m,)) for m in ("ma", "mb")]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    check(set(spans) == {"ma", "mb"}, f"both {HOL_TOKENS}-token streams completed: {sorted(spans)}")
    duration = min(spans["ma"][1] - spans["ma"][0], spans["mb"][1] - spans["mb"][0])
    overlap = min(spans["ma"][1], spans["mb"][1]) - max(spans["ma"][0], spans["mb"][0])
    check(duration > 3.0, f"streams are slow enough to expose blocking ({duration:.1f}s)")
    check(overlap > 0.8 * duration, f"streams overlap {overlap:.1f}s of {duration:.1f}s (head-of-line blocking)")
    times = read_sse(
        f"{base}/v1/chat/completions",
        json.dumps({"model": "ma", "messages": [], "stream": True, "max_tokens": 5}).encode(),
        headers=hdr,
    )
    gaps = [times[i + 1] - times[i] for i in range(min(4, len(times) - 1))]
    check(len(gaps) >= 2 and all(g >= 0.1 for g in gaps), f"incremental sse gaps {gaps}")
    done = threading.Event()
    chunks: list[float] = []

    def _long_stream() -> None:
        b = json.dumps({"model": "ma", "messages": [], "stream": True, "max_tokens": 20}).encode()
        chunks.extend(read_sse(f"{base}/v1/chat/completions", b, headers=hdr, max_chunks=30))
        done.set()

    th = threading.Thread(target=_long_stream)
    th.start()
    time.sleep(0.5)
    run_goethe([str(goethe), "key", "create", "--name", "reload", "--raw"], env=env)
    run_goethe([str(goethe), "allowlist", "add", "127.0.0.2/32"], env=env)
    th.join(timeout=60)
    check(done.is_set() and len(chunks) >= 22, f"stream completed across nginx reloads ({len(chunks)} events)")


if __name__ == "__main__":
    main_exit(main)
