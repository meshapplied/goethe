#!/usr/bin/env python3
"""Stage 6 CDN signature and checksum verification gate (packaging keygen/sign scripts and install.sh)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    Proc,
    free_port,
    http_get,
    main_exit,
    run_temp_prefix,
    sign_cdn_tree,
    spawn,
    temp_home,
    wait_until,
)

ARCH = "aarch64" if platform.machine() in ("aarch64", "arm64") else platform.machine()
LOCK = "requirements.aarch64.lock" if ARCH == "aarch64" else "requirements.lock"
PACKAGING = GOETHE_PKG / "packaging"


def _sh(*cmd: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=60)


def _mini_cdn(cdn: Path, pub: Path) -> None:
    """A CDN tree with real pyenv files, stub binaries and an install.sh embedding pub."""
    shutil.copytree(GOETHE_PKG / "cdn" / "v1", cdn / "v1")
    (cdn / "pyenv").mkdir()
    for name in ("requirements.lock", "requirements.aarch64.lock", "bootstrap.sh"):
        shutil.copy2(GOETHE_PKG / "pyenv" / name, cdn / "pyenv" / name)
    bindir = cdn / "bin" / f"linux-{ARCH}"
    bindir.mkdir(parents=True)
    for name in ("goethe", "goethed", "uv"):
        (bindir / name).write_bytes(b"#!/bin/sh\necho stub\n")
        (bindir / name).chmod(0o755)
    rr = _sh("bash", PACKAGING / "subst_install_pubkey.sh", GOETHE_PKG / "cdn" / "install.sh", pub, cdn / "install.sh")
    if rr.returncode != 0:
        raise RuntimeError(f"subst_install_pubkey failed: {rr.stderr}")


def _mid_release(tampered: Path, good: Path, url: str, home: Path) -> None:
    """A stale file that the CDN replaces during the install is retried until it matches."""
    boot = tampered / "pyenv" / "bootstrap.sh"
    stale = boot.read_text()
    timer = threading.Timer(2.0, lambda: shutil.copy2(good / "pyenv" / "bootstrap.sh", boot))
    timer.start()
    try:
        out = _install(tampered / "install.sh", url, home, {"GOETHE_VERIFY_ONLY": "1", "GOETHE_FETCH_RETRIES": "10",
                                                            "GOETHE_FETCH_RETRY_SECS": "1"})
    finally:
        timer.join()
        boot.write_text(stale)
    if not out.startswith("rc=0") or "retrying every 30 s" not in out or "verified CDN artifacts" not in out:
        raise RuntimeError(f"mid-release retry: {out[-500:]}")


def _install(install_sh: Path, pkg_url: str, home: Path, extra: dict[str, str] | None = None) -> str:
    """Run install.sh against pkg_url into home; return rc-prefixed combined output."""
    env = {
        "GOETHE_HOME": str(home / "g"), "GOETHE_PKG_URL": pkg_url, "GOETHE_NO_LINK": "1",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
        "GOETHE_FETCH_RETRIES": "1", "GOETHE_FETCH_RETRY_SECS": "0", **(extra or {}),
    }
    rr = subprocess.run(["bash", str(install_sh)], env=env, capture_output=True, text=True, timeout=120,
                        stdin=subprocess.DEVNULL)
    return f"rc={rr.returncode}\n{rr.stdout}{rr.stderr}"


def _variant(root: Path, name: str, cdn: Path) -> Path:
    """Copy of the signed CDN under root/name for a negative case."""
    out = root / name
    shutil.copytree(cdn, out)
    return out


def _answers(url: str) -> bool:
    """True once the local HTTP server serves url."""
    try:
        return http_get(url, timeout=2)[0] == 200
    except OSError:
        return False


def main() -> None:
    """Exercise keygen, sign_cdn.sh and the signed-CDN verification paths in install.sh."""
    root = Path(f"/tmp/{run_temp_prefix('integrity')}root")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    srv: Proc | None = None
    try:
        keys, other = root / "keys", root / "other-keys"
        for d in (keys, other):
            if _sh("bash", PACKAGING / "keygen.sh", d).returncode != 0:
                raise RuntimeError("keygen.sh failed")
        if _sh("bash", PACKAGING / "keygen.sh", keys).returncode == 0:
            raise RuntimeError("keygen.sh overwrote an existing key")
        if (keys / "release.key").stat().st_mode & 0o077:
            raise RuntimeError("release.key is readable by group or others")
        cdn = root / "cdn"
        _mini_cdn(cdn, keys / "release_pub.pem")
        wrong = _variant(root, "cdn-wrongkey", cdn)
        rr = _sh("bash", PACKAGING / "sign_cdn.sh", wrong, other / "release.key")
        if rr.returncode == 0 or "does not match" not in rr.stderr:
            raise RuntimeError(f"sign_cdn.sh accepted a key that differs from install.sh's: {rr.stderr}")
        rr = _sh("bash", PACKAGING / "sign_cdn.sh", cdn, keys / "release.key")
        if rr.returncode != 0:
            raise RuntimeError(f"sign_cdn.sh failed: {rr.stderr}")
        tampered = _variant(root, "cdn-tamper", cdn)
        marker = root / "bootstrap.ran"
        boot = tampered / "pyenv" / "bootstrap.sh"
        boot.write_text(boot.read_text() + f'\necho ran >"{marker}"\n')
        badsums = _variant(root, "cdn-badsums", cdn)
        text = (badsums / "SHA256SUMS").read_text()
        (badsums / "SHA256SUMS").write_text(text.replace("a", "b", 1))
        nolock = _variant(root, "cdn-nolock", cdn)
        (nolock / "pyenv" / LOCK).unlink()
        sign_cdn_tree(nolock, keys / "release.key")
        unsigned = _variant(root, "cdn-unsigned", cdn)
        for name in ("SHA256SUMS", "SHA256SUMS.sig", "VERSION"):
            (unsigned / name).unlink()
        shutil.copy2(GOETHE_PKG / "cdn" / "install.sh", unsigned / "install.sh")
        port = free_port()
        srv = spawn(
            [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=root,
            label="integrity-cdn",
        )
        base = f"http://127.0.0.1:{port}"
        if not wait_until(lambda: _answers(f"{base}/"), timeout=30):
            raise RuntimeError(f"local CDN server on port {port} did not come up")
        verify = {"GOETHE_VERIFY_ONLY": "1"}
        cases = [
            ("valid", cdn, verify, "rc=0", "verified CDN artifacts"),
            ("tampered", tampered, {}, "checksum mismatch", None),
            ("bad sums", badsums, verify, "signature verification failed", None),
            ("missing lock", nolock, verify, f"CDN missing pyenv/{LOCK}", None),
            ("unsigned", unsigned, verify, "unsigned install.sh", None),
            ("insecure", unsigned, {**verify, "GOETHE_INSECURE": "1"}, "rc=0", "GOETHE_INSECURE=1 skips"),
        ]
        with temp_home(stub_python=False, tag="integrity", pyenv=False) as home:
            for name, tree, extra, want, also in cases:
                out = ""
                for _ in range(20):
                    out = _install(tree / "install.sh", f"{base}/{tree.name}", home / name.replace(" ", "-"), extra)
                    if "Connection refused" not in out and "Couldn't connect" not in out:
                        break
                if want not in out or (also and also not in out) or (want != "rc=0" and out.startswith("rc=0")):
                    raise RuntimeError(f"{name}: expected {want!r}: {out[-500:]}")
            if marker.exists():
                raise RuntimeError("bootstrap executed after checksum failure")
            _mid_release(tampered, cdn, f"{base}/{tampered.name}", home / "mid-release")
    finally:
        if srv is not None:
            srv.stop()
        shutil.rmtree(root, ignore_errors=True)
    print("stage6_integrity ok")


if __name__ == "__main__":
    main_exit(main)
