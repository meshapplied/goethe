#!/usr/bin/env python3
"""Signed package update tests (local CDN)."""

from __future__ import annotations

import os
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
    """Fake goethe and goethed binaries for this arch, and the SGLang wheel the stub home has, in the CDN tree."""
    bindir = cdn / f"bin/linux-{ARCH}"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "goethe").write_bytes(b"goethe-test-bin-" + tag)
    (bindir / "goethed").write_bytes(b"goethed-test-bin-" + tag)
    (cdn / "wheels").mkdir(exist_ok=True)
    (cdn / "wheels/LATEST").write_text("sglang-test.whl\n", encoding="utf-8")
    (cdn / "wheels/sglang-test.whl").write_bytes(b"wheel")


def _wheel_installed(home: Path) -> Path:
    """Record the CDN's SGLang wheel as installed in the stub home, so only the binaries differ."""
    (home / "pyenv").mkdir(exist_ok=True)
    (home / "pyenv/LATEST").write_text("sglang-test.whl\n", encoding="utf-8")
    (home / "pyenv/sglang-test.whl").write_bytes(b"wheel")
    return home


def _same_version(goethe: Path, cdn: Path, key: Path, pub: Path, base: str) -> None:
    """At the binary's own version: the CDN's binaries are up to date, other binaries of it are another build."""
    version = run([str(goethe), "--version"]).stdout.split()[1]
    sign_cdn_tree(cdn, key, version)
    with temp_home(stub_python=True, tag="upd3") as home:
        _wheel_installed(home)
        for name in ("goethe", "goethed"):
            shutil.copy2(cdn / f"bin/linux-{ARCH}" / name, home / "bin" / name)
        env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
        rr = run([str(goethe), "update", "--check", "--yes"], env=env)
        if rr.returncode != 0 or "everything is up to date" not in rr.stdout:
            raise RuntimeError(f"the CDN's binaries at the same version must be up to date: {rr.stdout} {rr.stderr}")
        _binaries(cdn, b"v3")
        sign_cdn_tree(cdn, key, version)
        want = f"update available: goethe {version} -> {version} (another build)"
        rr = run([str(goethe), "update", "--check", "--yes"], env=env)
        if rr.returncode != 0 or want not in rr.stdout:
            raise RuntimeError(f"another build at the same version must read {want!r}: {rr.stdout} {rr.stderr}")


def _local_newer(goethe: Path, cdn: Path, key: Path, pub: Path, base: str) -> None:
    """A build ahead of the release keeps its binaries and ask model; only a bare --allow-downgrade swaps them."""
    _binaries(cdn, b"old")
    (cdn / "ask").mkdir(exist_ok=True)
    (cdn / "ask/ask-model-v1.bin").write_bytes(b"older ask model")
    sign_cdn_tree(cdn, key, "0.0.1")
    cases = (([], False), (["--check"], False), (["engine"], False))
    for flags, downgrade in (*cases, (["--allow-downgrade"], True)):
        with temp_home(stub_python=True, tag="upd4") as home:
            mine = {name: b"my-newer-" + name.encode() for name in ("goethe", "goethed")}
            for name, data in mine.items():
                (home / "bin" / name).write_bytes(data)
            (home / "ask").mkdir(exist_ok=True)
            (home / "ask/ask-model-v1.bin").write_bytes(b"my newer ask model")
            _wheel_installed(home)
            env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
            rr = run([str(goethe), "update", *flags, "--yes"], env=env, timeout=120)
            out = rr.stdout + rr.stderr
            if "engine" in flags and "SGLang" not in out:
                raise RuntimeError(f"update {flags}: the engine layers are still looked at: {out}")
            ask = (home / "ask/ask-model-v1.bin").read_bytes()
            if downgrade:
                swapped = (home / "bin" / "goethe").read_bytes() == b"goethe-test-bin-old" and ask == b"older ask model"
                if rr.returncode != 0 or not swapped:
                    raise RuntimeError(f"update {flags}: the bare --allow-downgrade must install the older ones: {out}")
                continue
            note = "is newer than the published 0.0.1; keeping it"
            if rr.returncode != 0 or note not in out or "--allow-downgrade to install" in out:
                raise RuntimeError(f"update {flags}: a newer build is a note, not an error: {rr.returncode} {out}")
            if any((home / "bin" / name).read_bytes() != data for name, data in mine.items()):
                raise RuntimeError(f"update {flags}: the newer binaries were replaced")
            if (home / "VERSION").exists():
                raise RuntimeError(f"update {flags}: VERSION was written although the binaries were kept")
            if ask != b"my newer ask model" or (flags != ["--check"] and "are not installed over yours" not in out):
                raise RuntimeError(f"update {flags}: an older package must not replace the ask model: {out}")


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
            _wheel_installed(home)
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
