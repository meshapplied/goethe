#!/usr/bin/env python3
"""Stage 6 CDN signature and checksum verification gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from lib.goethe_test import GOETHE_PKG, Proc, Skip, free_port, main_exit, spawn, temp_home


def _sign_tree(cdn: Path, key: Path) -> None:
    """Write VERSION, SHA256SUMS, and SHA256SUMS.sig for cdn."""
    ver = (GOETHE_PKG / "crates" / "goethe" / "Cargo.toml").read_text().split('version = "', 1)[1].split('"', 1)[0]
    (cdn / "VERSION").write_text(ver + "\n")
    lines: list[str] = []
    for path in sorted(p for p in cdn.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sig")):
        rel = path.relative_to(cdn).as_posix()
        digest = subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
        lines.append(f"{digest}  {rel}")
    (cdn / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(cdn / "SHA256SUMS.sig"), str(cdn / "SHA256SUMS")],
        check=True,
    )


def _mini_cdn(root: Path, pub: Path) -> Path:
    """Build a minimal CDN directory under root and return its path."""
    cdn = root / "cdn"
    cdn.mkdir(parents=True)
    shutil.copytree(GOETHE_PKG / "cdn" / "v1", cdn / "v1")
    (cdn / "pyenv").mkdir()
    shutil.copy2(GOETHE_PKG / "pyenv" / "requirements.lock", cdn / "pyenv" / "requirements.lock")
    shutil.copy2(GOETHE_PKG / "pyenv" / "bootstrap.sh", cdn / "pyenv" / "bootstrap.sh")
    bindir = cdn / "bin" / "linux-x86_64"
    bindir.mkdir(parents=True)
    for name in ("goethe", "goethed", "uv"):
        exe = bindir / name
        exe.write_bytes(b"#!/bin/sh\necho stub\n")
        exe.chmod(0o755)
    subst = GOETHE_PKG / "packaging" / "subst_install_pubkey.sh"
    subprocess.run(["bash", str(subst), str(GOETHE_PKG / "cdn" / "install.sh"), str(pub), str(cdn / "install.sh")], check=True)
    return cdn


def _run_install(install_sh: Path, pkg_url: str, home: Path, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run install.sh against pkg_url into home."""
    env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": pkg_url, "GOETHE_NO_LINK": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if extra:
        env.update(extra)
    return subprocess.run(["bash", str(install_sh)], env=env, capture_output=True, text=True, timeout=120)


def main() -> None:
    """Exercise signed CDN verification paths in install.sh."""
    keygen = GOETHE_PKG / "packaging" / "keygen.sh"
    if not keygen.is_file():
        raise Skip("packaging/keygen.sh missing")
    root = Path(f"/tmp/goethe-test-{os.environ.get('GOETHE_GATE_RUN_ID', os.getpid())}-integrity")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    subprocess.run(["bash", str(keygen), str(root)], check=True)
    key = root / "release.key"
    pub = root / "release_pub.pem"
    cdn = _mini_cdn(root, pub)
    _sign_tree(cdn, key)
    port = free_port()
    srv: Proc = spawn(
        [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(root),
        label="integrity-cdn",
    )
    pkg_url = f"http://127.0.0.1:{port}/cdn"
    marker = cdn / "pyenv" / "bootstrap.ran"
    time.sleep(0.3)
    try:
        with temp_home(stub_python=False, tag="integrity") as home:
            rr = _run_install(cdn / "install.sh", pkg_url, home, {"GOETHE_VERIFY_ONLY": "1"})
            if rr.returncode != 0:
                raise RuntimeError(f"valid verify failed: {rr.stderr[-400:]}")
        tampered = root / "cdn-tamper"
        shutil.copytree(cdn, tampered)
        boot = tampered / "pyenv" / "bootstrap.sh"
        boot.write_text(boot.read_text() + '\necho ran >"' + str(marker) + '"\n')
        with temp_home(stub_python=False, tag="integrity-tamper") as home:
            rr = _run_install(tampered / "install.sh", f"http://127.0.0.1:{port}/cdn-tamper", home)
            if rr.returncode == 0:
                raise RuntimeError("tampered bootstrap should fail")
            if "checksum mismatch" not in (rr.stderr + rr.stdout):
                raise RuntimeError(f"expected checksum mismatch: {rr.stderr[-400:]}")
            if marker.exists():
                raise RuntimeError("bootstrap executed after checksum failure")
        badsums = root / "cdn-badsums"
        shutil.copytree(cdn, badsums)
        text = (badsums / "SHA256SUMS").read_text()
        (badsums / "SHA256SUMS").write_text(text.replace("a", "b", 1))
        with temp_home(stub_python=False, tag="integrity-sig") as home:
            rr = _run_install(badsums / "install.sh", f"http://127.0.0.1:{port}/cdn-badsums", home, {"GOETHE_VERIFY_ONLY": "1"})
            if rr.returncode == 0 or "signature verification failed" not in (rr.stderr + rr.stdout):
                raise RuntimeError(f"expected signature failure: {rr.stderr[-400:]}")
        unsigned = root / "cdn-unsigned"
        shutil.copytree(cdn, unsigned)
        for name in ("SHA256SUMS", "SHA256SUMS.sig", "VERSION"):
            (unsigned / name).unlink(missing_ok=True)
        subst = GOETHE_PKG / "packaging" / "subst_install_pubkey.sh"
        subprocess.run(["bash", str(subst), str(GOETHE_PKG / "cdn" / "install.sh"), str(pub), str(unsigned / "install.sh")], check=True)
        with temp_home(stub_python=False, tag="integrity-unsigned") as home:
            rr = _run_install(unsigned / "install.sh", f"http://127.0.0.1:{port}/cdn-unsigned", home, {"GOETHE_VERIFY_ONLY": "1"})
            if rr.returncode == 0:
                raise RuntimeError("unsigned install should fail without GOETHE_INSECURE")
        with temp_home(stub_python=False, tag="integrity-insecure") as home:
            rr = _run_install(
                unsigned / "install.sh",
                f"http://127.0.0.1:{port}/cdn-unsigned",
                home,
                {"GOETHE_INSECURE": "1", "GOETHE_VERIFY_ONLY": "1"},
            )
            if rr.returncode != 0:
                raise RuntimeError(f"GOETHE_INSECURE should proceed: {rr.stderr[-400:]}")
    finally:
        srv.stop()
        shutil.rmtree(root, ignore_errors=True)
    print("stage6_integrity ok")


if __name__ == "__main__":
    main_exit(main)
