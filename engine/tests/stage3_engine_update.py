#!/usr/bin/env python3
"""goethe update --engine stub bootstrap env."""

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


def _sign(dist: Path, key: Path) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "VERSION").write_text("9.9.9-test\n", encoding="utf-8")
    pyenv = dist / "pyenv"
    pyenv.mkdir(parents=True)
    (pyenv / "bootstrap.sh").write_text(
        "#!/bin/bash\n"
        "echo \"GOETHE_SGLANG_WHEEL=$GOETHE_SGLANG_WHEEL\" >\"$GOETHE_HOME/pyenv/bootstrap.env\"\n"
        "sha256sum \"$GOETHE_SGLANG_WHEEL\" >>\"$GOETHE_HOME/pyenv/bootstrap.env\"\n",
        encoding="utf-8",
    )
    (pyenv / "requirements.lock").write_text("lock\n", encoding="utf-8")
    wheel = pyenv / "wheel.whl"
    wheel.write_bytes(b"wheel-bytes")
    (dist / "wheels" / "LATEST").parent.mkdir(parents=True, exist_ok=True)
    (dist / "wheels" / "LATEST").write_text("wheel.whl\n", encoding="utf-8")
    (dist / "wheels" / "wheel.whl").write_bytes(b"wheel-bytes")
    bindir = dist / f"bin/linux-{ARCH}"
    bindir.mkdir(parents=True)
    (bindir / "goethe").write_bytes(b"goethe")
    (bindir / "goethed").write_bytes(b"goethed")
    subprocess.check_call(
        f"cd '{dist}' && find . -type f ! -name 'SHA256SUMS*' | sed 's|^\\./||' | sort | while read f; do sha256sum \"$f\" | awk -v p=\"$f\" '{{print $1 \"  \" p}}'; done > SHA256SUMS",
        shell=True,
    )
    subprocess.check_call(
        ["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(dist / "SHA256SUMS.sig"), str(dist / "SHA256SUMS")]
    )


def main() -> None:
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("eng"), dir="/tmp"))
    try:
        key = work / "key.pem"
        subprocess.check_call(["openssl", "ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", str(key)])
        pub = work / "pub.pem"
        subprocess.check_call(["openssl", "ec", "-in", str(key), "-pubout", "-out", str(pub)])
        cdn = work / "cdn"
        _sign(cdn, key)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (SimpleHTTPRequestHandler,), {"__init__": lambda s, *a, **k: SimpleHTTPRequestHandler.__init__(s, *a, directory=str(cdn), **k)}))
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with temp_home(stub_python=True, tag="engine") as home:
            shutil.copy(pub, home / "release_pub.pem")
            rr = run(
                [str(goethe), "update", "--engine", "--yes"],
                env={"GOETHE_HOME": str(home), "GOETHE_PKG_URL": f"http://127.0.0.1:{port}"},
                timeout=120,
            )
            if rr.returncode != 0:
                raise RuntimeError(rr.stderr or rr.stdout)
            wheel = home / "pyenv" / "wheel.whl"
            if not wheel.is_file():
                raise RuntimeError("wheel missing")
            envf = home / "pyenv" / "bootstrap.env"
            if not envf.is_file():
                raise RuntimeError("bootstrap not run")
            text = envf.read_text(encoding="utf-8")
            if "GOETHE_SGLANG_WHEEL=" not in text:
                raise RuntimeError("bootstrap env missing wheel")
            if str(wheel) not in text:
                raise RuntimeError("bootstrap env missing wheel path")
        srv.shutdown()
        print("stage3_engine_update ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
