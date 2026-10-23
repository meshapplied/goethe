#!/usr/bin/env python3
"""goethe uninstall in an isolated fake HOME tree: refusals, keep-models default, --purge-models."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run, run_temp_prefix


def _fake_install(root: Path) -> Path:
    """A home that looks like a goethe install (marker plus a regular bin/goethe), with one model file."""
    (root / "bin").mkdir(parents=True)
    (root / ".goethe-home").write_text("goethe home\n", encoding="utf-8")
    (root / "state.db").write_bytes(b"x")
    (root / "bin" / "goethe").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    (root / "models" / "org__m").mkdir(parents=True)
    (root / "models" / "org__m" / "w.safetensors").write_text("x", encoding="utf-8")
    return root


def _fake_tools(root: Path) -> Path:
    """systemctl and crontab stand-ins that log calls, so uninstall can never reach the real user's boot config."""
    tools = root / "fake-bin"
    tools.mkdir()
    for name in ("systemctl", "crontab"):
        (tools / name).write_text(f'#!/bin/sh\necho "{name} $*" >>"{root}/tools.log"\nexit 0\n', encoding="utf-8")
        (tools / name).chmod(0o755)
    return tools


def main() -> None:
    """Exercise uninstall guards and what it removes and keeps."""
    goethe = require_cli("goethe")
    root = Path(tempfile.mkdtemp(prefix=run_temp_prefix("uninst-root"), dir="/tmp"))
    try:
        user_home = root / "userhome"
        inst = _fake_install(user_home / ".goethe")
        env = {
            "HOME": str(user_home),
            "GOETHE_SOCK": str(root / "no.sock"),
            "GOETHE_CRONTAB_FILE": str(root / "crontab"),
            "GOETHE_SYSTEMD_USER_UNIT_DIR": str(root / "units"),
            "PATH": f"{_fake_tools(root)}:/usr/bin:/bin",
        }
        if run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(user_home)}).returncode == 0:
            raise RuntimeError("should refuse when GOETHE_HOME equals HOME")
        if run([str(goethe), "uninstall"], env={**env, "GOETHE_HOME": str(inst)}).returncode == 0:
            raise RuntimeError("needs --yes without tty")
        if not (inst / "state.db").is_file():
            raise RuntimeError("refused uninstall still removed files")
        rr = run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(inst)})
        if rr.returncode != 0:
            raise RuntimeError(rr.stderr or rr.stdout)
        if (inst / "state.db").exists() or (inst / "bin").exists() or (inst / ".goethe-home").exists():
            raise RuntimeError("uninstall left state.db, bin or the home marker behind")
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
        linked = _fake_install(root / "linked")
        (linked / "bin" / "goethe").unlink()
        (linked / "bin" / "goethe").symlink_to("/bin/true")
        for name, home in (("models-only home", models_only), ("symlinked bin/goethe", linked)):
            rr = run([str(goethe), "uninstall", "--purge-models", "--yes"], env={**env, "GOETHE_HOME": str(home)})
            if rr.returncode == 0 or "does not look like a goethe install" not in rr.stderr + rr.stdout:
                raise RuntimeError(f"{name} must be refused: rc={rr.returncode} {rr.stderr or rr.stdout}")
            if not (home / "models").is_dir():
                raise RuntimeError(f"refused uninstall of a {name} removed files")
        if not user_home.is_dir():
            raise RuntimeError("uninstall touched the user's HOME")
        log = root / "tools.log"
        calls = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
        if any(c.startswith("systemctl") and "--user" not in c for c in calls):
            raise RuntimeError(f"uninstall reached the system manager: {calls}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("stage3_uninstall ok")


if __name__ == "__main__":
    main_exit(main)
