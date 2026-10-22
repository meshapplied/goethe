#!/usr/bin/env python3
"""mesh uninstall in an isolated fake HOME tree."""

from __future__ import annotations

from pathlib import Path

from lib.mesh_test import main_exit, require_cli, run, run_temp_prefix, temp_home
import shutil
import tempfile


def main() -> None:
    mesh = require_cli("mesh")
    root = Path(tempfile.mkdtemp(prefix=run_temp_prefix("uninst-root"), dir="/tmp"))
    try:
        user_home = root / "userhome"
        user_home.mkdir()
        mesh_inst = user_home / ".mesh"
        mesh_inst.mkdir()
        (mesh_inst / "state.db").write_bytes(b"x")
        (mesh_inst / "bin").mkdir()
        (mesh_inst / "bin" / "mesh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        rr = run(
            [str(mesh), "uninstall", "--yes"],
            env={"MESH_HOME": str(user_home), "HOME": str(user_home)},
        )
        if rr.returncode == 0:
            raise RuntimeError("should refuse when MESH_HOME equals HOME")
        rr_tty = run(
            [str(mesh), "uninstall"],
            env={"MESH_HOME": str(mesh_inst), "HOME": str(user_home)},
        )
        if rr_tty.returncode == 0:
            raise RuntimeError("needs --yes without tty")
        models_only = root / "models-only"
        models_only.mkdir()
        (models_only / "models").mkdir()
        (models_only / "models" / "keep").write_text("x", encoding="utf-8")
        rr_purge = run(
            [str(mesh), "uninstall", "--purge-models", "--yes"],
            env={"MESH_HOME": str(models_only), "HOME": str(user_home)},
        )
        if rr_purge.returncode != 0:
            raise RuntimeError(rr_purge.stderr or rr_purge.stdout)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("stage3_uninstall ok")


if __name__ == "__main__":
    main_exit(main)
