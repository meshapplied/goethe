#!/usr/bin/env python3
"""Signed package update tests (local CDN)."""

from __future__ import annotations

import platform
import re
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


def _same_version(goethe: Path, cdn: Path, key: Path, pub: Path, base: str) -> None:
    """At the binary's own version: the CDN's binaries are up to date, another BUILDINFO commit is another build."""
    found = re.search(r"goethe (\S+) \(commit ([^,)\s]+)", run([str(goethe), "--version"]).stdout)
    if not found:
        raise RuntimeError("goethe --version does not name its version and commit")
    version, commit = found.group(1), found.group(2)
    sign_cdn_tree(cdn, key, version)
    with temp_home(stub_python=True, tag="upd3") as home:
        shutil.copy(pub, home / "release_pub.pem")
        for name in ("goethe", "goethed"):
            shutil.copy2(cdn / f"bin/linux-{ARCH}" / name, home / "bin" / name)
        env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
        rr = run([str(goethe), "update", "--check", "--yes"], env=env)
        if rr.returncode != 0 or "up to date" not in rr.stdout:
            raise RuntimeError(f"the CDN's binaries at the same version must be up to date: {rr.stdout} {rr.stderr}")
        published = "0123456789abcdef0123456789abcdef01234567"[: max(7, len(commit.removesuffix("-dirty")))]
        (cdn / "BUILDINFO").write_text(f"built_at=x\nengine_commit={published}\nengine_dirty=0\n", encoding="utf-8")
        _binaries(cdn, b"v3")
        sign_cdn_tree(cdn, key, version)
        mine = f"{version} (commit {commit})" if commit != "unknown" else version
        want = f"update available: goethe {mine} -> {version} (commit {published})"
        rr = run([str(goethe), "update", "--check", "--yes"], env=env)
        if rr.returncode != 0 or want not in rr.stdout:
            raise RuntimeError(f"another build at the same version must read {want!r}: {rr.stdout} {rr.stderr}")


def main() -> None:
    """update --check, tamper rejection, verified binary swap, bad signature, same version and another build."""
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
        _same_version(goethe, cdn, key, pub, base)
        print("stage3_update ok")
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
