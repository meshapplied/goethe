#!/usr/bin/env python3
"""Stage 6 install.sh end-to-end gate: signed CDN into a throwaway home, gateway answers, idempotent re-install."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    Proc,
    Skip,
    clean_test_gpu,
    free_port,
    home_goethe_env,
    home_process_tree,
    http_get,
    http_post,
    main_exit,
    run,
    run_goethe,
    spawn,
    temp_home,
    test_gpus,
    test_public_port,
    wait_until,
    write_stub_python,
)

_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
}


def _install(install_sh: Path, env: dict[str, str]) -> float:
    """Run install.sh like `curl | bash` would (stdin closed) and return its duration."""
    t0 = time.time()
    rr = subprocess.run(
        ["bash", str(install_sh)], env=env, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL
    )
    if rr.returncode != 0 or "Goethe install complete" not in rr.stdout:
        raise RuntimeError(f"install.sh failed rc={rr.returncode}: {(rr.stdout + rr.stderr)[-1500:]}")
    return time.time() - t0


def _port_busy(port: int) -> bool:
    """True when 127.0.0.1:port cannot be bound."""
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _stop_installed_daemon(home: Path, env: dict[str, str]) -> None:
    """goethe stop, then fail (after killing them) if processes of this home's own goethed binary remain."""
    run([str(home / "bin" / "goethe"), "stop"], env=env, timeout=60)
    goethed = str(home / "bin" / "goethed")
    if wait_until(lambda: not home_process_tree(home), timeout=15):
        return
    leftovers = sorted(home_process_tree(home))
    for pid in leftovers:
        try:
            if os.readlink(f"/proc/{pid}/exe") == goethed:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    raise RuntimeError(f"processes left under {home} after goethe stop: {leftovers}")


def _log_tails(home: Path) -> str:
    """Last lines of goethed.log and the newest engine log, for failures in a home that is about to vanish."""
    logs = sorted((home / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    picks = [p for p in logs if p.name == "goethed.log"] + [p for p in logs if p.name != "goethed.log"][-1:]
    return "\n".join(f"== {p.name}\n" + "\n".join(p.read_text(errors="replace").splitlines()[-15:]) for p in picks)


def main() -> None:
    """Install from the signed CDN, serve a chat through the gateway with a stub engine, re-install in place."""
    if os.environ.get("GOETHE_TEST_E2E", "0") != "1":
        raise Skip("set GOETHE_TEST_E2E=1 for installer e2e gate")
    cdn = Path(os.environ.get("GOETHE_TEST_CDN", f"{GOETHE_PKG}/dist/cdn"))
    if not (cdn / "install.sh").is_file() or not (cdn / "SHA256SUMS.sig").is_file():
        raise RuntimeError(f"GOETHE_TEST_E2E=1 needs a signed CDN at {cdn} (packaging/build_cdn.sh)")
    port = free_port()
    srv: Proc = spawn(
        [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=cdn.parent,
        label="cdn-http",
    )
    pkg_url = f"http://127.0.0.1:{port}/{cdn.name}"
    gpu = clean_test_gpu()
    gpu = gpu if gpu is not None else test_gpus()[-1]
    timings: list[str] = []
    try:
        if not wait_until(lambda: http_get(f"{pkg_url}/VERSION")[0] == 200, timeout=10):
            raise RuntimeError("local CDN server did not come up")
        with temp_home(tag="e2e", pyenv=False) as home:
            env = {
                **os.environ, **home_goethe_env(home), "GOETHE_PKG_URL": pkg_url, "GOETHE_NO_LINK": "1",
                "HOME": str(home), "UV_CACHE_DIR": str(Path.home() / ".cache" / "uv"),
                "UV_PYTHON_INSTALL_DIR": str(Path.home() / ".local" / "share" / "uv" / "python"),
            }
            env.pop("GOETHE_PYTHON", None)
            try:
                timings.append(f"install={_install(cdn / 'install.sh', env):.0f}s")
                prefix = run([str(home / "py-env" / "bin" / "python"), "-c", "import sys; print(sys.prefix)"]).stdout
                if Path(prefix.strip()).resolve() != (home / "py-env").resolve():
                    raise RuntimeError(f"installer py-env is not a venv in the temp home: {prefix.strip()}")
                probe = "import sglang, torch; print(torch.__version__)"
                imp = run([str(home / "py-env" / "bin" / "python"), "-c", probe])
                if imp.returncode != 0 or "+cu130" not in imp.stdout:
                    raise RuntimeError(f"py-env lacks the hash-locked torch/sglang: {imp.stderr[-400:]}")
                for sub in ("bin/goethe", "bin/goethed", "bin/uv", "nginx/certs/server.key", "hardware_state.json"):
                    if not (home / sub).exists():
                        raise RuntimeError(f"missing {sub} after install")
                bin_dir = home / "py-env" / "bin"
                (bin_dir / "python-real").symlink_to(os.readlink(bin_dir / "python"))
                write_stub_python(bin_dir, bin_dir / "python-real")
                goethe = str(home / "bin" / "goethe")
                src = home / "src-model"
                src.mkdir()
                (src / "config.json").write_text(json.dumps(_CFG), encoding="utf-8")
                (src / "model.safetensors").write_bytes(b"x" * 256)
                run_goethe([goethe, "install", str(src), "--name", "e2e/tiny", "--no-load"], env=env, timeout=120)
                run_goethe([goethe, "load", "e2e/tiny", "--gpus", str(gpu), "--yes"], env=env, timeout=300)
                out = run_goethe([goethe, "key", "create", "--name", "e2e", "--raw"], env=env).stdout
                token = next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))
                run_goethe([goethe, "allowlist", "add", "127.0.0.1/32"], env=env)
                base = f"https://127.0.0.1:{test_public_port(home)}"
                auth = {"Authorization": f"Bearer {token}"}
                if not wait_until(lambda: http_get(f"{base}/v1/models", headers=auth)[0] == 200, timeout=20):
                    raise RuntimeError(f"public API after install returned {http_get(f'{base}/v1/models', auth)[0]}")
                msg = {"model": "e2e/tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
                body = json.dumps(msg)
                code, reply = http_post(f"{base}/v1/chat/completions", body.encode(), headers=auth)
                if code != 200 or b"choices" not in reply:
                    raise RuntimeError(f"completion through the gateway failed: code={code} body={reply[:200]!r}")
                run_goethe([goethe, "unload", "--all"], env=env)
                _stop_installed_daemon(home, env)
                (home / "nginx" / "mux_port").unlink(missing_ok=True)
                busy = _port_busy(30000)
                reinstall_env = {k: v for k, v in env.items() if k not in ("GOETHE_MUX_PORT", "GOETHE_SOCK")}
                timings.append(f"reinstall={_install(cdn / 'install.sh', reinstall_env):.0f}s")
                mux = int((home / "nginx" / "mux_port").read_text(encoding="utf-8").strip())
                if (busy and not 30100 <= mux <= 30199) or (not busy and mux != 30000):
                    raise RuntimeError(f"mux port {mux} after re-install (30000 busy={busy})")
                if run_goethe([goethe, "ps", "--json"], env=reinstall_env).stdout.strip() not in ("[]", ""):
                    raise RuntimeError("re-install brought back an unloaded model")
            except Exception:
                print(_log_tails(home), file=sys.stderr)
                raise
            finally:
                if (home / "bin" / "goethe").is_file():
                    _stop_installed_daemon(home, env)
    finally:
        srv.stop()
    print(f"stage6_install_e2e ok gpu={gpu} {' '.join(timings)}")


if __name__ == "__main__":
    main_exit(main)
