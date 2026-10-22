#!/usr/bin/env python3
"""Signed package update tests (local CDN)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run, run_temp_prefix, temp_home

ARCH = subprocess.check_output(["uname", "-m"], text=True).strip()
if ARCH == "arm64":
    ARCH = "aarch64"


def _sign_tree(dist: Path, key: Path) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    ver = "9.9.9-test"
    (dist / "VERSION").write_text(f"{ver}\n", encoding="utf-8")
    bindir = dist / f"bin/linux-{ARCH}"
    if bindir.is_dir():
        shutil.rmtree(bindir)
    bindir.mkdir(parents=True)
    (bindir / "goethe").write_bytes(b"goethe-test-bin-v2")
    (bindir / "goethed").write_bytes(b"goethed-test-bin-v2")
    subprocess.check_call(
        f"cd '{dist}' && find . -type f ! -name 'SHA256SUMS*' ! -name 'SHA256SUMS' "
        "| sed 's|^\\./||' | sort | while read -r f; do "
        'sha256sum "$f" | awk -v p="$f" \'{print $1 "  " p}\'; '
        "done > SHA256SUMS",
        shell=True,
    )
    subprocess.check_call(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            str(key),
            "-out",
            str(dist / "SHA256SUMS.sig"),
            str(dist / "SHA256SUMS"),
        ]
    )


def main() -> None:
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("pkg"), dir="/tmp"))
    try:
        key = work / "key.pem"
        pub = work / "pub.pem"
        subprocess.check_call(["openssl", "ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", str(key)])
        subprocess.check_call(["openssl", "ec", "-in", str(key), "-pubout", "-out", str(pub)])
        cdn = work / "cdn"
        _sign_tree(cdn, key)
        port_holder: list[int] = []

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(cdn), **kw)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port_holder.append(srv.server_address[1])
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port_holder[0]}"
        with temp_home(stub_python=True, tag="upd") as home:
            shutil.copy(pub, home / "release_pub.pem")
            (home / "bin").mkdir(exist_ok=True)
            env = {"GOETHE_HOME": str(home), "GOETHE_PKG_URL": base}
            rr = run([str(goethe), "update", "--check", "--yes"], env=env, timeout=60)
            if rr.returncode != 0 or "update available" not in (rr.stdout + rr.stderr):
                raise RuntimeError(rr.stdout + rr.stderr)
            bad = cdn / f"bin/linux-{ARCH}/goethe"
            bad.write_bytes(b"tampered")
            rr_bad = run([str(goethe), "update", "--yes"], env=env, timeout=60)
            if rr_bad.returncode == 0:
                raise RuntimeError("tampered binary should fail")
            _sign_tree(cdn, key)
            rr_ok = run([str(goethe), "update", "--yes"], env=env, timeout=120)
            if rr_ok.returncode != 0:
                raise RuntimeError(rr_ok.stderr or rr_ok.stdout)
            if (home / "bin" / "goethe").read_bytes() != b"goethe-test-bin-v2":
                raise RuntimeError("goethe binary not updated")
        tampered_sig = cdn / "SHA256SUMS.sig"
        tampered_sig.write_bytes(b"broken")
        with temp_home(stub_python=True, tag="upd2") as home2:
            shutil.copy(pub, home2 / "release_pub.pem")
            rr_sig = run([str(goethe), "update", "--check", "--yes"], env={"GOETHE_HOME": str(home2), "GOETHE_PKG_URL": base})
            if rr_sig.returncode == 0:
                raise RuntimeError("bad signature should fail")
        same_ver = cdn / "VERSION"
        same_ver.write_text("0.1.0\n", encoding="utf-8")
        with temp_home(stub_python=True, tag="upd3") as home3:
            shutil.copy(pub, home3 / "release_pub.pem")
            subprocess.check_call(
                f"cd '{cdn}' && find . -type f ! -name 'SHA256SUMS*' ! -name 'SHA256SUMS' "
                "| sed 's|^\\./||' | sort | while read -r f; do "
                'sha256sum "$f" | awk -v p="$f" \'{print $1 "  " p}\'; '
                "done > SHA256SUMS",
                shell=True,
            )
            subprocess.check_call(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-sign",
                    str(key),
                    "-out",
                    str(cdn / "SHA256SUMS.sig"),
                    str(cdn / "SHA256SUMS"),
                ]
            )
            rr_same = run([str(goethe), "update", "--check", "--yes"], env={"GOETHE_HOME": str(home3), "GOETHE_PKG_URL": base})
            if rr_same.returncode != 0:
                raise RuntimeError("same version check should noop")
            if "up to date" not in rr_same.stdout:
                raise RuntimeError(rr_same.stdout)
        srv.shutdown()
        print("stage3_update ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
