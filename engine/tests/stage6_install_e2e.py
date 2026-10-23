#!/usr/bin/env python3
"""Stage 6 install.sh end-to-end gate: signed CDN into a throwaway home, gateway answers, idempotent re-install."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
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
    roomiest_test_gpu,
    run,
    run_goethe,
    run_temp_prefix,
    sha256_file,
    spawn,
    temp_home,
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


def _install(install_sh: Path, env: dict[str, str], late: list[str]) -> float:
    """Run install.sh like `curl | bash` would (stdin closed), note a failed recommend, return the duration."""
    t0 = time.time()
    rr = subprocess.run(
        ["bash", str(install_sh)], env=env, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL
    )
    if rr.returncode != 0 or "Goethe install complete" not in rr.stdout:
        raise RuntimeError(f"install.sh failed rc={rr.returncode}: {(rr.stdout + rr.stderr)[-1500:]}")
    if "goethe recommend failed" in rr.stdout + rr.stderr:
        late.append(f"the installer's recommend failed: {rr.stderr[-300:]}")
    return time.time() - t0


def _check_goethe_env(home: Path, goethe: str, env: dict[str, str], pkg_url: str, late: list[str]) -> None:
    """goethe.env pins the local CDN and its registry (0600), and the CLI honours it without those variables."""
    path = home / "goethe.env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    want = [f"GOETHE_PKG_URL={pkg_url}", f"GOETHE_REGISTRY_URL={pkg_url}/v1/registry.json"]
    if any(w not in lines for w in want) or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise RuntimeError(f"goethe.env must hold {want} with mode 0600: {lines}")
    bare = {k: v for k, v in env.items() if k not in ("GOETHE_PKG_URL", "GOETHE_REGISTRY_URL")}
    rec = run([goethe, "recommend"], env=bare, timeout=120)
    if rec.returncode != 0 or not rec.stdout.strip():
        late.append(f"recommend with only goethe.env failed (exit {rec.returncode}): {rec.stderr[-300:]}")


def _cdn_with_ternary(src: Path, root: Path) -> Path:
    """A copy of the signed CDN plus a tiny ternary pack, re-signed with a throwaway key embedded in its install.sh."""
    cdn = root / "cdn"
    shutil.copytree(src, cdn)
    pack = root / "pack" / "ternary-kernels"
    pack.mkdir(parents=True)
    (pack / "README").write_text("e2e ternary pack\n", encoding="utf-8")
    (cdn / "goethe").mkdir(exist_ok=True)
    tarball = cdn / "goethe" / "ternary-kernels.tar.zst"
    subprocess.run(["tar", "--zstd", "-C", str(pack.parent), "-cf", str(tarball), "ternary-kernels"], check=True)
    keys, packaging = root / "keys", GOETHE_PKG / "packaging"
    steps = [
        ["bash", str(packaging / "keygen.sh"), str(keys)],
        ["bash", str(packaging / "subst_install_pubkey.sh"), str(GOETHE_PKG / "cdn" / "install.sh"),
         str(keys / "release_pub.pem"), str(cdn / "install.sh")],
        ["bash", str(packaging / "sign_cdn.sh"), str(cdn), str(keys / "release.key")],
    ]
    for cmd in steps:
        rr = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if rr.returncode != 0:
            raise RuntimeError(f"{Path(cmd[1]).name} failed: {rr.stderr[-300:]}")
    return cdn


def _check_ternary(home: Path, goethe: str, env: dict[str, str], cdn: Path, late: list[str]) -> None:
    """The installer unpacked the ternary pack and recorded its signed sha256, which update --check compares."""
    want = sha256_file(cdn / "goethe" / "ternary-kernels.tar.zst")
    record = home / "goethe" / "ternary-kernels.sha256"
    got = record.read_text(encoding="utf-8").strip() if record.is_file() else None
    if got != want or not (home / "goethe" / "ternary-kernels" / "README").is_file():
        raise RuntimeError(f"ternary pack not unpacked or its sha256 not recorded: {got} != {want}")
    upd = run([goethe, "update", "--check", "--ternary"], env=env, timeout=120)
    if f"ternary pack {want[:12]} is up to date" not in upd.stdout:
        late.append(f"update --check --ternary must see the recorded pack: {(upd.stdout + upd.stderr)[-300:]}")


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
    src = Path(os.environ.get("GOETHE_TEST_CDN", f"{GOETHE_PKG}/dist/cdn"))
    if not (src / "install.sh").is_file() or not (src / "SHA256SUMS.sig").is_file():
        raise RuntimeError(f"GOETHE_TEST_E2E=1 needs a signed CDN at {src} (packaging/build_cdn.sh)")
    root = Path(tempfile.mkdtemp(prefix=run_temp_prefix("e2ecdn"), dir="/tmp"))
    try:
        cdn = _cdn_with_ternary(src, root)
        _serve_and_install(cdn)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _serve_and_install(cdn: Path) -> None:
    """Serve cdn over local HTTP and run the install, gateway, ternary and re-install checks against it."""
    port = free_port()
    srv: Proc = spawn(
        [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=cdn.parent,
        label="cdn-http",
    )
    pkg_url = f"http://127.0.0.1:{port}/{cdn.name}"
    gpu = clean_test_gpu()
    gpu = gpu if gpu is not None else roomiest_test_gpu(8.0)
    timings: list[str] = []
    late: list[str] = []
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
            env.pop("GOETHE_REGISTRY_URL", None)
            try:
                timings.append(f"install={_install(cdn / 'install.sh', env, late):.0f}s")
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
                _check_goethe_env(home, goethe, env, pkg_url, late)
                _check_ternary(home, goethe, env, cdn, late)
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
                timings.append(f"reinstall={_install(cdn / 'install.sh', reinstall_env, late):.0f}s")
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
    if late:
        raise RuntimeError(" | ".join(late))
    print(f"stage6_install_e2e ok gpu={gpu} {' '.join(timings)}")


if __name__ == "__main__":
    main_exit(main)
