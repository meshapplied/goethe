#!/usr/bin/env python3
"""Signed package update tests (local CDN)."""

from __future__ import annotations

import platform
import shutil
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lib.goethe_test import main_exit, make_release_key, require_cli, run, run_temp_prefix, sign_cdn_tree, temp_home

ARCH = "aarch64" if platform.machine() in ("aarch64", "arm64") else platform.machine()


def _binaries(cdn: Path, tag: bytes) -> None:
    """Place fake goethe and goethed binaries for this arch into the CDN tree."""
    bindir = cdn / f"bin/linux-{ARCH}"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "goethe").write_bytes(b"goethe-test-bin-" + tag)
    (bindir / "goethed").write_bytes(b"goethed-test-bin-" + tag)


def main() -> None:
    """update --check, tamper rejection, verified binary swap, bad signature and same-version no-op."""
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("pkg"), dir="/tmp"))
    srv: ThreadingHTTPServer | None = None
    try:
        key, pub = make_release_key(work / "keys")
        cdn = work / "cdn"
        _binaries(cdn, b"v2")
        sign_cdn_tree(cdn, key, "9.9.9-test")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(cdn)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with temp_home(stub_python=True, tag="upd") as home:
            shutil.copy(pub, home / "release_pub.pem")
            env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
            rr = run([str(goethe), "update", "--check", "--yes"], env=env, timeout=60)
            if rr.returncode != 0 or "update available" not in rr.stdout + rr.stderr:
                raise RuntimeError(rr.stdout + rr.stderr)
            (cdn / f"bin/linux-{ARCH}/goethe").write_bytes(b"tampered")
            if run([str(goethe), "update", "--yes"], env=env, timeout=60).returncode == 0:
                raise RuntimeError("tampered binary should fail")
            if (home / "bin" / "goethe").is_file() and (home / "bin" / "goethe").read_bytes() == b"tampered":
                raise RuntimeError("tampered binary was installed")
            _binaries(cdn, b"v2")
            sign_cdn_tree(cdn, key)
            rr_ok = run([str(goethe), "update", "--yes"], env=env, timeout=120)
            if rr_ok.returncode != 0:
                raise RuntimeError(rr_ok.stderr or rr_ok.stdout)
            if (home / "bin" / "goethe").read_bytes() != b"goethe-test-bin-v2":
                raise RuntimeError("goethe binary not updated")
        (cdn / "SHA256SUMS.sig").write_bytes(b"broken")
        with temp_home(stub_python=True, tag="upd2") as home2:
            shutil.copy(pub, home2 / "release_pub.pem")
            env2 = {"GOETHE_HOME": str(home2), "GOETHE_PKG_URL": base}
            if run([str(goethe), "update", "--check", "--yes"], env=env2).returncode == 0:
                raise RuntimeError("bad signature should fail")
        sign_cdn_tree(cdn, key, "0.1.0")
        with temp_home(stub_python=True, tag="upd3") as home3:
            shutil.copy(pub, home3 / "release_pub.pem")
            env3 = {"GOETHE_HOME": str(home3), "GOETHE_PKG_URL": base}
            rr_same = run([str(goethe), "update", "--check", "--yes"], env=env3)
            if rr_same.returncode != 0 or "up to date" not in rr_same.stdout:
                raise RuntimeError(f"same version check should no-op: {rr_same.stdout} {rr_same.stderr}")
        print("stage3_update ok")
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
