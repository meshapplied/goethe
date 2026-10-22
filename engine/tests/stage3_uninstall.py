#!/usr/bin/env python3
"""goethe uninstall in an isolated fake HOME tree: refusals, keep-models default, --purge-models."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run, run_temp_prefix


def _fake_install(root: Path) -> Path:
    """A home that looks like a goethe install, with one model file."""
    root.mkdir(parents=True)
    (root / "state.db").write_bytes(b"x")
    (root / "bin").mkdir()
    (root / "bin" / "goethe").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    (root / "models" / "org__m").mkdir(parents=True)
    (root / "models" / "org__m" / "w.safetensors").write_text("x", encoding="utf-8")
    return root


def main() -> None:
    """Exercise uninstall guards and what it removes and keeps."""
    goethe = require_cli("goethe")
    root = Path(tempfile.mkdtemp(prefix=run_temp_prefix("uninst-root"), dir="/tmp"))
    try:
        user_home = root / "userhome"
        inst = _fake_install(user_home / ".goethe")
        env = {"HOME": str(user_home), "GOETHE_SOCK": str(root / "no.sock")}
        if run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(user_home)}).returncode == 0:
            raise RuntimeError("should refuse when GOETHE_HOME equals HOME")
        if run([str(goethe), "uninstall"], env={**env, "GOETHE_HOME": str(inst)}).returncode == 0:
            raise RuntimeError("needs --yes without tty")
        if not (inst / "state.db").is_file():
            raise RuntimeError("refused uninstall still removed files")
        rr = run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(inst)})
        if rr.returncode != 0:
            raise RuntimeError(rr.stderr or rr.stdout)
        if (inst / "state.db").exists() or (inst / "bin").exists():
            raise RuntimeError("uninstall left state.db or bin behind")
        if not (inst / "models" / "org__m" / "w.safetensors").is_file():
            raise RuntimeError("uninstall without --purge-models deleted models")
        purge = _fake_install(root / "purge")
        rr = run([str(goethe), "uninstall", "--purge-models", "--yes"], env={**env, "GOETHE_HOME": str(purge)})
        if rr.returncode != 0:
            raise RuntimeError(rr.stderr or rr.stdout)
        if (purge / "models").exists() or (purge / "state.db").exists():
            raise RuntimeError("--purge-models left models or state behind")
        models_only = root / "models-only"
        (models_only / "models").mkdir(parents=True)
        (models_only / "models" / "keep").write_text("x", encoding="utf-8")
        rr = run([str(goethe), "uninstall", "--purge-models", "--yes"], env={**env, "GOETHE_HOME": str(models_only)})
        if rr.returncode != 0 or (models_only / "models" / "keep").exists():
            raise RuntimeError(f"--purge-models on a models-only home did not purge: {rr.stderr or rr.stdout}")
        if not user_home.is_dir():
            raise RuntimeError("uninstall touched the user's HOME")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("stage3_uninstall ok")


if __name__ == "__main__":
    main_exit(main)
