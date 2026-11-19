#!/usr/bin/env python3
"""Signed package update tests (local CDN)."""

from __future__ import annotations

import os
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
    found = re.search(r"goethe (\S+) \(commit ([^,)\s]+)", run([str(goethe), "--build-info"]).stdout)
    if not found:
        raise RuntimeError("goethe --version does not name its version and commit")
    version, commit = found.group(1), found.group(2)
    sign_cdn_tree(cdn, key, version)
    with temp_home(stub_python=True, tag="upd3") as home:
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
        if commit == "unknown":
            return
        dirty = int(commit.endswith("-dirty"))
        buildinfo = f"built_at=y\nengine_commit={commit.removesuffix('-dirty')}\nengine_dirty={dirty}\n"
        (cdn / "BUILDINFO").write_text(buildinfo, encoding="utf-8")
        _binaries(cdn, b"v4")
        sign_cdn_tree(cdn, key, version)
        want = f"update available: goethe {mine} -> {version} (commit {commit}, another build)"
        rr = run([str(goethe), "update", "--check", "--yes"], env=env)
        if rr.returncode != 0 or want not in rr.stdout:
            raise RuntimeError(f"another build of the same commit must read {want!r}: {rr.stdout} {rr.stderr}")


def _local_newer(goethe: Path, cdn: Path, key: Path, pub: Path, base: str) -> None:
    """A build ahead of the release keeps its binaries without an error; only a bare --allow-downgrade swaps them."""
    _binaries(cdn, b"old")
    (cdn / "wheels").mkdir(exist_ok=True)
    (cdn / "wheels/LATEST").write_text("sglang-test.whl\n", encoding="utf-8")
    (cdn / "wheels/sglang-test.whl").write_bytes(b"wheel")
    sign_cdn_tree(cdn, key, "0.0.1")
    cases = (([], False), (["--check"], False), (["engine"], False), (["engine", "--allow-downgrade"], False))
    for flags, downgrade in (*cases, (["--allow-downgrade"], True)):
        with temp_home(stub_python=True, tag="upd4") as home:
            mine = {name: b"my-newer-" + name.encode() for name in ("goethe", "goethed")}
            for name, data in mine.items():
                (home / "bin" / name).write_bytes(data)
            (home / "pyenv").mkdir(exist_ok=True)
            (home / "pyenv/LATEST").write_text("sglang-test.whl\n", encoding="utf-8")
            (home / "pyenv/sglang-test.whl").write_bytes(b"wheel")
            env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
            rr = run([str(goethe), "update", *flags, "--yes"], env=env, timeout=120)
            out = rr.stdout + rr.stderr
            if "engine" in flags and "SGLang" not in out:
                raise RuntimeError(f"update {flags}: the engine layers are still looked at: {out}")
            if downgrade:
                if rr.returncode != 0 or (home / "bin" / "goethe").read_bytes() != b"goethe-test-bin-old":
                    raise RuntimeError(f"update {flags}: the bare --allow-downgrade must install the older ones: {out}")
                continue
            note = "is newer than the published 0.0.1; keeping it"
            if rr.returncode != 0 or note not in out or "--allow-downgrade to install" in out:
                raise RuntimeError(f"update {flags}: a newer build is a note, not an error: {rr.returncode} {out}")
            if any((home / "bin" / name).read_bytes() != data for name, data in mine.items()):
                raise RuntimeError(f"update {flags}: the newer binaries were replaced")
            if (home / "VERSION").exists():
                raise RuntimeError(f"update {flags}: VERSION was written although the binaries were kept")


def main() -> None:
    """update --check, tamper rejection, verified binary swap, bad signature, same version and another build."""
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("pkg"), dir="/tmp"))
    srv: ThreadingHTTPServer | None = None
    try:
        key, pub = make_release_key(work / "keys")
        os.environ["GOETHE_RELEASE_PUB"] = str(pub)
        cdn = work / "cdn"
        _binaries(cdn, b"v2")
        sign_cdn_tree(cdn, key, "9.9.9-test")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(cdn)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with temp_home(stub_python=True, tag="upd") as home:
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
            env2 = {"GOETHE_HOME": str(home2), "GOETHE_PKG_URL": base}
            if run([str(goethe), "update", "--check", "--yes"], env=env2).returncode == 0:
                raise RuntimeError("bad signature should fail")
        _same_version(goethe, cdn, key, pub, base)
        _local_newer(goethe, cdn, key, pub, base)
        print("stage3_update ok")
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
