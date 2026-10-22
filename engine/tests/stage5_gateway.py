#!/usr/bin/env python3
"""Stage 5 nginx gateway and security gate."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lib.goethe_test import (
    _acquire_gpu_flocks,
    ensure_nginx,
    http_get,
    http_post,
    main_exit,
    read_sse,
    require_cli,
    require_gpus,
    run_goethe,
    home_goethe_env,
    seed_installed_model,
    sha256_hex,
    spawn,
    sqlite_query,
    temp_home,
    test_mux_port,
    test_public_port,
)

ASSERTS = 0


def check(cond: bool, msg: str) -> None:
    """Record one assertion."""
    global ASSERTS
    ASSERTS += 1
    if not cond:
        raise RuntimeError(msg)


def _tls_certs(home: Path) -> None:
    """Generate self-signed TLS material under GOETHE_HOME."""
    cert = home / "nginx" / "certs"
    cert.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(cert / "server.key"),
            "-out",
            str(cert / "server.crt"),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=goethe-test",
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    """Run default-deny gateway checks with stub engines and TLS."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    gpus = require_gpus(1)
    gpu = gpus[0]
    with temp_home(stub_python=True) as home:
        public_port = test_public_port(home)
        base = f"https://127.0.0.1:{public_port}"
        ensure_nginx(home)
        _tls_certs(home)
        ma = home / "models" / "org__ma"
        mb = home / "models" / "org__mb"
        ma.mkdir(parents=True)
        mb.mkdir(parents=True)
        (ma / "goethe_card.toml").write_text('model_id="ma"\ncontext_length=512\nmax_running_requests=2\n', encoding="utf-8")
        (mb / "goethe_card.toml").write_text('model_id="mb"\ncontext_length=512\nmax_running_requests=2\n', encoding="utf-8")
        seed_installed_model(home, "ma", str(ma.relative_to(home)), 5.0, 512, 512, 2, 1)
        seed_installed_model(home, "mb", str(mb.relative_to(home)), 5.0, 512, 512, 2, 1)
        goethed = spawn([str(require_cli("goethed"))], env=home_goethe_env(home), label="goethed")
        env = {**home_goethe_env(home), "CUDA_VISIBLE_DEVICES": str(gpu), "STUB_TOKEN_INTERVAL": "0.04"}
        g2 = gpus[1] if len(gpus) > 1 else gpu
        release_gpu = _acquire_gpu_flocks(sorted({gpu, g2}))
        try:
            run_goethe([str(goethe), "load", "ma", "--gpus", str(gpu)], env=env, timeout=300)
            run_goethe([str(goethe), "load", "mb", "--gpus", str(g2)], env=env, timeout=300)
            code, _ = http_get(f"{base}/v1/models")
            check(code in (403, 401), "empty allowlist denies")
            run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=home_goethe_env(home))
            code, body = http_get(f"{base}/health")
            check(code == 200 and json.loads(body.decode("utf-8")).get("status") == "ok", "health no key")
            code, _ = http_get(f"{base}/health/ready?model=ma")
            check(code == 200, "health ready running")
            code, _ = http_get(f"{base}/health/ready?model=not-loaded-xyz")
            check(code == 503, "health ready missing")
            code, _ = http_get(f"{base}/health/ready")
            check(code == 400, "health ready no param")
            code, _ = http_get(f"{base}/v1/models")
            check(code == 401, "no key unauthorized")
            code, _ = http_get(f"{base}/v1/models", headers={"Authorization": "Bearer goethe_sk_bad"})
            check(code == 401, "wrong key")
            rr_key = run_goethe([str(goethe), "key", "create", "--name", "t", "--raw"], env=home_goethe_env(home))
            token = [ln for ln in rr_key.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            check(token.startswith("goethe_sk_"), "raw key prefix")
            keys_text = (home / "nginx" / "api_keys.conf").read_text(encoding="utf-8")
            check("goethe_sk_" not in keys_text, "api_keys.conf no raw token")
            mode = (home / "nginx" / "api_keys.conf").stat().st_mode
            check((mode & 0o777) == 0o600, "api_keys.conf mode 0600")
            row = sqlite_query(home / "state.db", "SELECT key_id, token_hash, raw_prefix FROM api_keys LIMIT 1")[0]
            key_id, thash, prefix = row[0], row[1], row[2]
            check(thash == sha256_hex(token), "sha256 stored")
            check(prefix != token and len(str(prefix)) < len(token), "raw_prefix not full key")
            code, body = http_get(f"{base}/v1/models", headers={"Authorization": f"Bearer {token}"})
            check(code == 200, "valid key")
            models = json.loads(body.decode("utf-8")).get("data", [])
            ids = {m.get("id") for m in models}
            check(ids == {"ma", "mb"} or ids == {"org__ma", "org__mb"} or len(ids) == 2, "models list")
            rr_adm = run_goethe(
                [str(goethe), "key", "create", "--name", "adm", "--admin", "--raw"],
                env=home_goethe_env(home),
            )
            adm_token = [ln for ln in rr_adm.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            code, body = http_get(
                f"{base}/health/admin",
                headers={"Authorization": f"Bearer {adm_token}"},
            )
            check(code == 200, "admin health")
            adm = json.loads(body.decode("utf-8"))
            check(
                "gpus" in adm and "instances" in adm and "inactive_models" in adm,
                "admin health keys",
            )
            code, _ = http_get(
                f"{base}/health/admin",
                headers={"Authorization": f"Bearer {token}"},
            )
            check(code == 404, "admin health denied")
            rr_sc = run_goethe(
                [str(goethe), "key", "create", "--name", "sc", "--raw", "--models", "ma"],
                env=home_goethe_env(home),
            )
            sc_token = [ln for ln in rr_sc.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            code, body = http_get(f"{base}/v1/models", headers={"Authorization": f"Bearer {sc_token}"})
            sc_ids = {m.get("id") for m in json.loads(body.decode("utf-8")).get("data", [])}
            check(sc_ids == {"ma"} or sc_ids == {"org__ma"}, "scoped models list")
            chat_ma = json.dumps({"model": "ma", "messages": [{"role": "user", "content": "x"}], "max_tokens": 2}).encode()
            code, cbody = http_post(f"{base}/v1/chat/completions", chat_ma, headers={"Authorization": f"Bearer {token}"})
            check(code == 200 and b"model=ma" in cbody, "route ma")
            chat_mb = json.dumps({"model": "mb", "messages": [{"role": "user", "content": "y"}], "max_tokens": 2}).encode()
            code, cbody = http_post(f"{base}/v1/chat/completions", chat_mb, headers={"Authorization": f"Bearer {token}"})
            check(code == 200 and b"model=mb" in cbody, "route mb")
            chat_mb_sc = json.dumps(
                {"model": "mb", "messages": [{"role": "user", "content": "y"}], "max_tokens": 1}
            ).encode()
            code, cbody = http_post(
                f"{base}/v1/chat/completions",
                chat_mb_sc,
                headers={"Authorization": f"Bearer {sc_token}"},
            )
            check(code == 404 and b"Model Not Found" in cbody, "scoped model 404")
            run_goethe([str(goethe), "alias", "set", "chat", "ma"], env=home_goethe_env(home))
            chat_alias = json.dumps(
                {"model": "chat", "messages": [{"role": "user", "content": "a"}], "max_tokens": 2}
            ).encode()
            code, cbody = http_post(
                f"{base}/v1/chat/completions",
                chat_alias,
                headers={"Authorization": f"Bearer {token}"},
            )
            check(code == 200 and b"model=ma" in cbody, "alias routes")
            run_goethe([str(goethe), "alias", "remove", "chat"], env=home_goethe_env(home))
            bad = json.dumps({"model": "missing-model-xyz", "messages": [{"role": "user", "content": "z"}], "max_tokens": 1}).encode()
            code, _ = http_post(f"{base}/v1/chat/completions", bad, headers={"Authorization": f"Bearer {token}"})
            check(code == 404, "unknown model")
            rr_rpm = run_goethe(
                [str(goethe), "key", "create", "--name", "rpm3", "--raw", "--rpm", "3"],
                env=home_goethe_env(home),
            )
            rpm_token = [ln for ln in rr_rpm.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            chat_rpm = json.dumps(
                {"model": "ma", "messages": [{"role": "user", "content": "r"}], "max_tokens": 1}
            ).encode()
            for _ in range(3):
                code, _ = http_post(
                    f"{base}/v1/chat/completions",
                    chat_rpm,
                    headers={"Authorization": f"Bearer {rpm_token}"},
                )
                check(code == 200, "rpm chat under limit")
            code, _ = http_post(
                f"{base}/v1/chat/completions",
                chat_rpm,
                headers={"Authorization": f"Bearer {rpm_token}"},
            )
            check(code == 429, "rpm chat 429")
            rr_key2 = run_goethe([str(goethe), "key", "create", "--name", "burst", "--raw"], env=home_goethe_env(home))
            token2 = [ln for ln in rr_key2.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            with ThreadPoolExecutor(max_workers=16) as pool:
                burst_codes = list(
                    pool.map(
                        lambda _: http_get(f"{base}/v1/models", headers={"Authorization": f"Bearer {token2}"})[0],
                        range(120),
                    )
                )
            check(429 in burst_codes, "rate limit 429")
            rr_stream = run_goethe(
                [str(goethe), "key", "create", "--name", "stream", "--raw"],
                env=home_goethe_env(home),
            )
            stream_token = [ln for ln in rr_stream.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            run_goethe([str(goethe), "key", "revoke", str(key_id)], env=home_goethe_env(home))
            code, _ = http_get(f"{base}/v1/models", headers={"Authorization": f"Bearer {token}"})
            check(code == 401, "revoked key")
            plain, _ = http_get(f"http://127.0.0.1:{public_port}/v1/models")
            check(plain == 0, "plain http fails")
            big = b"x" * (11 * 1024 * 1024)
            code, _ = http_post(f"{base}/v1/chat/completions", big, headers={"Authorization": f"Bearer {token2}"})
            check(code == 413, "body too large")
            first: dict[str, float] = {}

            def _stream(model: str) -> None:
                body = json.dumps({"model": model, "messages": [], "stream": True, "max_tokens": 6}).encode()
                times = read_sse(
                    f"{base}/v1/chat/completions",
                    body,
                    headers={"Authorization": f"Bearer {stream_token}"},
                )
                if times:
                    first[model] = times[0]

            t_a = threading.Thread(target=_stream, args=("ma",))
            t_b = threading.Thread(target=_stream, args=("mb",))
            t_a.start()
            t_b.start()
            t_a.join(timeout=30)
            t_b.join(timeout=30)
            check("ma" in first and "mb" in first, "both streams started")
            check(abs(first["ma"] - first["mb"]) < 2.0, "concurrent streams interleave")
            body_sse = json.dumps({"model": "ma", "messages": [], "stream": True, "max_tokens": 5}).encode()
            times = read_sse(
                f"{base}/v1/chat/completions",
                body_sse,
                headers={"Authorization": f"Bearer {stream_token}"},
            )
            gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)] if len(times) > 1 else []
            check(any(g >= 0.02 for g in gaps), "incremental sse gaps")
            stream_done = threading.Event()
            chunks: list[float] = []

            def _long_stream() -> None:
                b = json.dumps({"model": "ma", "messages": [], "stream": True, "max_tokens": 20}).encode()
                chunks.extend(
                    read_sse(
                        f"{base}/v1/chat/completions",
                        b,
                        headers={"Authorization": f"Bearer {stream_token}"},
                        max_chunks=20,
                    )
                )
                stream_done.set()

            th = threading.Thread(target=_long_stream)
            th.start()
            time.sleep(0.15)
            run_goethe([str(goethe), "key", "create", "--name", "reload", "--raw"], env=home_goethe_env(home))
            run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=home_goethe_env(home))
            th.join(timeout=60)
            check(stream_done.is_set() and len(chunks) >= 3, "zero-downtime reload stream")
            ngx_pid_path = home / "nginx" / "nginx.pid"
            if ngx_pid_path.is_file():
                os.kill(int(ngx_pid_path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            rr_down = run_goethe(
                [str(goethe), "key", "create", "--name", "down", "--raw"],
                env=home_goethe_env(home),
            )
            check(rr_down.returncode == 0, "key create rc with gateway down")
            down_token = [ln for ln in rr_down.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
            check("goethe_sk_" not in (home / "nginx" / "api_keys.conf").read_text(encoding="utf-8"), "hash-only api_keys")
            mux = test_mux_port(home)
            code, _ = http_get(
                f"http://127.0.0.1:{mux}/v1/models",
                headers={"Authorization": f"Bearer {down_token}"},
            )
            check(code == 200, "mux auth gateway down")
        finally:
            run_goethe([str(goethe), "unload", "--all"], env=home_goethe_env(home), timeout=120)
            release_gpu()
            goethed.stop()
    print(f"stage5_gateway ok assertions={ASSERTS}")


if __name__ == "__main__":
    main_exit(main)
