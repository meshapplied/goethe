#!/usr/bin/env python3
"""goethe uninstall in an isolated fake HOME tree."""

from __future__ import annotations

from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run, run_temp_prefix, temp_home
import shutil
import tempfile


def main() -> None:
    goethe = require_cli("goethe")
    root = Path(tempfile.mkdtemp(prefix=run_temp_prefix("uninst-root"), dir="/tmp"))
    try:
        user_home = root / "userhome"
        user_home.mkdir()
        goethe_inst = user_home / ".goethe"
        goethe_inst.mkdir()
        (goethe_inst / "state.db").write_bytes(b"x")
        (goethe_inst / "bin").mkdir()
        (goethe_inst / "bin" / "goethe").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        rr = run(
            [str(goethe), "uninstall", "--yes"],
            env={"GOETHE_HOME": str(user_home), "HOME": str(user_home)},
        )
        if rr.returncode == 0:
            raise RuntimeError("should refuse when GOETHE_HOME equals HOME")
        rr_tty = run(
            [str(goethe), "uninstall"],
            env={"GOETHE_HOME": str(goethe_inst), "HOME": str(user_home)},
        )
        if rr_tty.returncode == 0:
            raise RuntimeError("needs --yes without tty")
        models_only = root / "models-only"
        models_only.mkdir()
        (models_only / "models").mkdir()
        (models_only / "models" / "keep").write_text("x", encoding="utf-8")
        rr_purge = run(
            [str(goethe), "uninstall", "--purge-models", "--yes"],
            env={"GOETHE_HOME": str(models_only), "HOME": str(user_home)},
        )
        if rr_purge.returncode != 0:
            raise RuntimeError(rr_purge.stderr or rr_purge.stdout)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("stage3_uninstall ok")


if __name__ == "__main__":
    main_exit(main)
