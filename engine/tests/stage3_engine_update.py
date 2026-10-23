#!/usr/bin/env python3
"""goethe update --engine fetches the verified wheel and hands it to bootstrap.sh."""

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


def _cdn_tree(dist: Path) -> None:
    """Minimal CDN whose bootstrap.sh records the wheel it was given instead of installing anything."""
    pyenv = dist / "pyenv"
    pyenv.mkdir(parents=True)
    (pyenv / "bootstrap.sh").write_text(
        "#!/bin/bash\n"
        'echo "GOETHE_SGLANG_WHEEL=$GOETHE_SGLANG_WHEEL" >"$GOETHE_HOME/pyenv/bootstrap.env"\n'
        'sha256sum "$GOETHE_SGLANG_WHEEL" >>"$GOETHE_HOME/pyenv/bootstrap.env"\n',
        encoding="utf-8",
    )
    (pyenv / "requirements.lock").write_text("lock\n", encoding="utf-8")
    (dist / "wheels").mkdir()
    (dist / "wheels" / "LATEST").write_text("wheel.whl\n", encoding="utf-8")
    (dist / "wheels" / "wheel.whl").write_bytes(b"wheel-bytes")
    bindir = dist / f"bin/linux-{ARCH}"
    bindir.mkdir(parents=True)
    (bindir / "goethe").write_bytes(b"goethe")
    (bindir / "goethed").write_bytes(b"goethed")


def main() -> None:
    """Run update --engine against a signed local CDN and check bootstrap received the verified wheel."""
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("eng"), dir="/tmp"))
    srv: ThreadingHTTPServer | None = None
    try:
        key, pub = make_release_key(work / "keys")
        cdn = work / "cdn"
        _cdn_tree(cdn)
        sign_cdn_tree(cdn, key, "9.9.9-test")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(cdn)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with temp_home(stub_python=True, tag="engine") as home:
            shutil.copy(pub, home / "release_pub.pem")
            rr = run(
                [str(goethe), "update", "--engine", "--yes"],
                env={"GOETHE_HOME": str(home), "GOETHE_PKG_URL": f"http://127.0.0.1:{srv.server_address[1]}"},
                timeout=120,
            )
            if rr.returncode != 0:
                raise RuntimeError(rr.stderr or rr.stdout)
            wheel = home / "pyenv" / "wheel.whl"
            if not wheel.is_file() or wheel.read_bytes() != b"wheel-bytes":
                raise RuntimeError("verified wheel missing from GOETHE_HOME/pyenv")
            envf = home / "pyenv" / "bootstrap.env"
            if not envf.is_file():
                raise RuntimeError("bootstrap not run")
            text = envf.read_text(encoding="utf-8")
            if f"GOETHE_SGLANG_WHEEL={wheel}" not in text:
                raise RuntimeError(f"bootstrap did not receive the wheel path: {text!r}")
        print("stage3_engine_update ok")
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
