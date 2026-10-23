#!/usr/bin/env python3
"""Stage 5 gateway hardening gate: allowlist, key cache, metering, upstream auth and JSON errors."""

from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lib.goethe_test import (
    _acquire_gpu_flocks,
    ensure_nginx,
    home_goethe_env,
    main_exit,
    read_sse,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sha256_hex,
    spawn,
    sqlite_query,
    stub_launch,
    stub_requests,
    temp_home,
    test_mux_port,
    test_public_port,
    wait_goethed_socket,
    wait_until,
)

MODEL = "org/hm"
ASSERTS = 0


def check(cond: bool, msg: str) -> None:
    """Record one assertion."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def request(
    port: int,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    tls: bool = True,
    source: str = "127.0.0.1",
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP(S) request from a chosen loopback source address; returns status, headers, body."""
    if tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            "127.0.0.1", port, context=ctx, timeout=30, source_address=(source, 0)
        )
    else:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30, source_address=(source, 0))
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def json_error(status: int, headers: dict[str, str], body: bytes, want: int, label: str) -> dict:
    """Assert an OpenAI-style JSON error body with the expected status."""
    check(status == want, f"{label}: status {status} != {want} body={body[:200]!r}")
    check(headers.get("content-type", "").startswith("application/json"), f"{label}: content-type {headers}")
    payload = json.loads(body.decode("utf-8"))
    err = payload.get("error")
    check(isinstance(err, dict) and bool(err.get("message")) and "type" in err, f"{label}: error shape {payload}")
    return err


def metrics_text(home: Path) -> str:
    """Scrape the loopback mux /metrics."""
    status, _, body = request(test_mux_port(home), "GET", "/metrics", tls=False)
    check(status == 200, f"metrics status {status}")
    return body.decode("utf-8")


def metric_value(text: str, pattern: str) -> float:
    """Value of the first sample whose line starts with pattern, else 0."""
    for line in text.splitlines():
        if line.startswith(pattern):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def stub_rows(model_dir: Path, method: str, path: str) -> list[dict]:
    """Authorization and X-Request-Id of the requests the stub engine saw for one method and path."""
    return [
        {"authorization": r["headers"].get("authorization"), "x_request_id": r["headers"].get("x-request-id")}
        for r in stub_requests(model_dir)
        if r["method"] == method and r["path"] == path
    ]


def new_key(goethe: Path, home: Path, *extra: str) -> tuple[str, str]:
    """Create a raw key and return (token, key_id)."""
    rr = run_goethe([str(goethe), "key", "create", "--name", "hard", "--raw", *extra], env=home_goethe_env(home))
    token = [ln for ln in rr.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
    rows = sqlite_query(
        home / "state.db", "SELECT key_id, raw_prefix FROM api_keys WHERE token_hash = ?", (sha256_hex(token),)
    )
    check(len(rows) == 1, "key row stored by hash")
    key_id, prefix = str(rows[0][0]), str(rows[0][1])
    check(key_id not in token, "key id is independent of the secret")
    check(len(prefix) == len("goethe_sk_") + 4 and token.startswith(prefix), f"raw_prefix is 4 hex chars: {prefix}")
    return token, key_id


def check_allowlist(goethe: Path, home: Path, public: int) -> None:
    """Default deny, per-source allow, CIDR canonicalization and injection refusal."""
    env = home_goethe_env(home)
    status, headers, body = request(public, "GET", "/v1/models")
    err = json_error(status, headers, body, 403, "empty allowlist")
    check(err.get("code") == "forbidden", f"403 code {err}")
    run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=env)
    status, _, body = request(public, "GET", "/health")
    check(status == 200 and json.loads(body).get("status") == "ok", "allowlisted health")
    allow = home / "nginx" / "allowlist.conf"
    check("allow 127.0.0.1/32;" in allow.read_text(encoding="utf-8"), "allowlist rendered")
    check(stat.S_IMODE(allow.stat().st_mode) == 0o600, "allowlist.conf 0600")
    check(stat.S_IMODE((home / "nginx").stat().st_mode) in (0o700, 0o711), "nginx dir private")
    status, headers, body = request(public, "GET", "/health", source="127.0.0.2")
    json_error(status, headers, body, 403, "other loopback source denied")
    before = allow.read_text(encoding="utf-8")
    rr = run([str(goethe), "allowlist", "add", "10.0.0.0/8; allow all"], env=env)
    check(rr.returncode != 0 and "invalid CIDR" in (rr.stderr + rr.stdout), f"injection refused: {rr.stderr}")
    check(allow.read_text(encoding="utf-8") == before, "allowlist file untouched by injection")
    rows = sqlite_query(home / "state.db", "SELECT cidr FROM ip_allowlist")
    check(all(";" not in str(r[0]) for r in rows), "no injected row stored")
    status, headers, body = request(public, "GET", "/health", source="127.0.0.2")
    json_error(status, headers, body, 403, "still denied after injection attempt")
    status, _, _ = request(public, "GET", "/health")
    check(status == 200, "gateway still serves after injection attempt")
    run_goethe([str(goethe), "allowlist", "add", "10.1.2.3/8"], env=env)
    listed = run_goethe([str(goethe), "allowlist", "list"], env=env).stdout
    check("10.0.0.0/8" in listed and "10.1.2.3/8" not in listed, f"canonical CIDR stored: {listed}")
    run_goethe([str(goethe), "allowlist", "remove", "10.1.2.3/8"], env=env)
    listed = run_goethe([str(goethe), "allowlist", "list"], env=env).stdout
    check("10.0.0.0/8" not in listed, "remove matches the canonical form")


def check_models_and_revocation(goethe: Path, home: Path, public: int) -> str:
    """Model listing fields, bearer leniency, model lookup, and revocation within one second."""
    token, _ = new_key(goethe, home)
    status, _, body = request(public, "GET", "/v1/models", headers={"Authorization": f"bearer   {token}  "})
    check(status == 200, f"lenient bearer: {status}")
    data = json.loads(body).get("data", [])
    check(len(data) == 1 and data[0].get("id") == MODEL, f"models list {data}")
    fields_ok = data[0].get("owned_by") == "goethe" and int(data[0].get("created", 0)) > 1_600_000_000
    check(fields_ok, f"model fields {data}")
    for path in (f"/v1/models/{MODEL}", "/v1/models/org%2Fhm"):
        status, _, body = request(public, "GET", path, headers={"Authorization": f"Bearer {token}"})
        check(status == 200 and json.loads(body).get("id") == MODEL, f"GET {path}: {status} {body[:120]!r}")
    status, headers, body = request(public, "GET", "/v1/models/nope", headers={"Authorization": f"Bearer {token}"})
    json_error(status, headers, body, 404, "unknown model id")
    victim, victim_id = new_key(goethe, home)
    status, _, _ = request(public, "GET", "/v1/models", headers={"Authorization": f"Bearer {victim}"})
    check(status == 200, "fresh key works on its first request")
    run_goethe([str(goethe), "key", "revoke", victim_id], env=home_goethe_env(home))
    t0 = time.time()
    refused_after = None
    while time.time() - t0 < 2.0:
        if request(public, "GET", "/v1/models", headers={"Authorization": f"Bearer {victim}"})[0] == 401:
            refused_after = time.time() - t0
            break
        time.sleep(0.1)
    check(refused_after is not None and refused_after <= 1.0, f"revoked key refused within 1s ({refused_after})")
    return token


def check_metering(home: Path, public: int, token: str) -> None:
    """Streaming tokens and full duration land in /metrics; unknown models get no label."""
    before = metrics_text(home)
    label = f'{{model="{MODEL}",type="completion"}}'
    tokens_before = metric_value(before, f"goethe_tokens_total{label}")
    dur_before = metric_value(before, f'goethe_request_duration_seconds_sum{{model="{MODEL}"}}')
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "s"}], "stream": True, "max_tokens": 6})
    chunks = read_sse(
        f"https://127.0.0.1:{public}/v1/chat/completions",
        body.encode(),
        headers={"Authorization": f"Bearer {token}"},
    )
    check(len(chunks) >= 6, f"stream chunks {len(chunks)}")
    text = ""
    for _ in range(20):
        text = metrics_text(home)
        if metric_value(text, f"goethe_tokens_total{label}") >= tokens_before + 6:
            break
        time.sleep(0.1)
    check(metric_value(text, f"goethe_tokens_total{label}") >= tokens_before + 6, "streamed tokens metered")
    dur = metric_value(text, f'goethe_request_duration_seconds_sum{{model="{MODEL}"}}') - dur_before
    check(dur >= 0.2, f"stream duration covers the whole body ({dur:.3f}s)")
    check(metric_value(text, f'goethe_time_to_first_token_seconds_count{{model="{MODEL}"}}') >= 1, "ttft recorded")
    evil = 'evil"} 1\ninjected_metric 42'
    payload = json.dumps({"model": evil, "messages": [], "max_tokens": 1}).encode()
    status, headers, body = request(
        public, "POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, body=payload
    )
    err = json_error(status, headers, body, 404, "unknown model")
    check(err.get("message") == "Model Not Found", f"404 message {err}")
    text = metrics_text(home)
    check("evil" not in text and "injected_metric" not in text, "client model string never becomes a label")
    check('model="_unknown"' in text, "unknown models are bucketed")
    for line in text.splitlines():
        well_formed = line.startswith("#") or re.match(r"^[a-z_]+(\{.*\})? [0-9.e+-]+$", line) is not None
        check(well_formed, f"bad line {line!r}")


def check_errors(home: Path, public: int, token: str) -> None:
    """OpenAI-style JSON errors from nginx and from the mux."""
    auth = {"Authorization": f"Bearer {token}"}
    status, headers, body = request(public, "GET", "/v1/models")
    check(json_error(status, headers, body, 401, "no key").get("code") == "invalid_api_key", "401 code")
    status, headers, body = request(public, "GET", "/v1/chat/completions", headers=auth)
    json_error(status, headers, body, 405, "wrong method")
    check(headers.get("allow") == "POST", f"Allow header {headers}")
    status, headers, body = request(public, "GET", "/nope")
    json_error(status, headers, body, 404, "unknown path via nginx")
    status, headers, body = request(public, "GET", "/v1/nope", headers=auth)
    json_error(status, headers, body, 404, "unknown /v1 path")
    status, headers, body = request(public, "POST", "/v1/chat/completions", headers=auth, body=b"{not json")
    json_error(status, headers, body, 400, "invalid JSON")
    big = b"x" * (11 * 1024 * 1024)
    status, headers, body = request(public, "POST", "/v1/chat/completions", headers=auth, body=big)
    json_error(status, headers, body, 413, "body too large")
    conn = http.client.HTTPConnection("127.0.0.1", test_mux_port(home), timeout=30)
    try:
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Authorization", auth["Authorization"])
        conn.putheader("Content-Length", str(11 * 1024 * 1024))
        conn.endheaders()
        resp = conn.getresponse()
        json_error(resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read(), 413, "mux body limit")
    finally:
        conn.close()
    status, headers, body = request(public, "GET", "/health/ready")
    json_error(status, headers, body, 400, "ready without model")


def check_upstream_auth(home: Path, public: int, model_dir: Path, token: str) -> None:
    """The engine runs with --api-key and sees exactly that key, never the client key; X-Request-Id passes through."""
    chat = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "k"}], "max_tokens": 2}).encode()
    auth = {"Authorization": f"Bearer {token}"}
    engine_key = sqlite_query(home / "state.db", "SELECT engine_key FROM loaded_instances WHERE model_id = ?", (MODEL,))
    current = str(engine_key[0][0] or "")
    check(re.fullmatch(r"[0-9a-f]{32}", current) is not None, f"engine_key is 32 hex: {current!r}")
    check(stub_launch(model_dir)["parsed"].get("api_key") == current, "engine launched with --api-key engine_key")
    headers = {**auth, "X-Request-Id": "hardening-rid-1"}
    status, resp_headers, _ = request(public, "POST", "/v1/chat/completions", headers=headers, body=chat)
    check(status == 200, f"chat {status}")
    check(resp_headers.get("x-request-id") == "hardening-rid-1", f"request id echoed {resp_headers}")
    rows = stub_rows(model_dir, "POST", "/v1/chat/completions")
    check(bool(rows) and rows[-1].get("authorization") == f"Bearer {current}", f"engine key forwarded {rows[-1:]}")
    check(rows[-1].get("x_request_id") == "hardening-rid-1", f"request id forwarded {rows[-1]}")
    status, resp_headers, _ = request(public, "POST", "/v1/chat/completions", headers=auth, body=chat)
    check(status == 200 and resp_headers.get("x-request-id", "").startswith("req-"), "request id minted")
    seen = [r.get("authorization") or "" for r in stub_rows(model_dir, "POST", "/v1/chat/completions")]
    check(all("goethe_sk_" not in a for a in seen), "client key never reaches the engine")
    time.sleep(1.1)
    status, _, body = request(public, "GET", "/health/ready?model=org%2Fhm")
    check(status == 200 and json.loads(body).get("status") == "ready", f"ready with encoded id: {status} {body!r}")
    probes = stub_rows(model_dir, "GET", "/v1/models")
    check(any(p.get("authorization") == f"Bearer {current}" for p in probes), "readiness probe uses the engine key")
    status, _, _ = request(public, "GET", "/health/ready?model=not-loaded")
    check(status == 503, "ready for an unknown model")


def check_ip_rate_limit(public: int) -> None:
    """A burst from one address is limited per IP at the server level, /health included, with a JSON 429."""
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: request(public, "GET", "/health"), range(120)))
    limited = [r for r in results if r[0] == 429]
    check(bool(limited), "per-IP limit reached on /health")
    json_error(*limited[0], 429, "rate limited")


def main() -> None:
    """Run the hardening checks against goethed, nginx and a header-recording stub engine."""
    goethe = require_cli("goethe")
    goethed_bin = require_cli("goethed")
    gpu = roomiest_test_gpu(15.0)
    with temp_home(stub_python=True, tag="harden") as home:
        ensure_nginx(home)
        model_dir = home / "models" / "org__hm"
        model_dir.mkdir(parents=True)
        (model_dir / "goethe_card.toml").write_text(
            f'model_id="{MODEL}"\ncontext_length=512\nmax_running_requests=2\n\n[env]\nSTUB_TOKEN_INTERVAL="0.05"\n',
            encoding="utf-8",
        )
        goethed = spawn([str(goethed_bin)], env=home_goethe_env(home), label="goethed")
        release = _acquire_gpu_flocks([gpu])
        try:
            wait_goethed_socket(home)
            seed_installed_model(home, MODEL, str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1)
            env = {**home_goethe_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
            run_goethe([str(goethe), "load", MODEL, "--gpus", str(gpu), "--yes"], env=env, timeout=300)
            public = test_public_port(home)
            check(wait_until(lambda: (home / "nginx" / "nginx.pid").is_file(), timeout=10), "gateway up")
            check_allowlist(goethe, home, public)
            token = check_models_and_revocation(goethe, home, public)
            check_metering(home, public, token)
            check_errors(home, public, token)
            check_upstream_auth(home, public, model_dir, token)
            check_ip_rate_limit(public)
            check(stat.S_IMODE((home / "nginx" / "api_keys.conf").stat().st_mode) == 0o600, "api_keys.conf 0600")
            check(os.access(home / "nginx" / "master.json", os.R_OK), "nginx master identity recorded")
        finally:
            run([str(goethe), "unload", "--all"], env=home_goethe_env(home), timeout=120)
            release()
            goethed.stop()
    print(f"stage5_gateway_hardening ok assertions={ASSERTS}")


if __name__ == "__main__":
    main_exit(main)
