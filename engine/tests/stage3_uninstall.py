#!/usr/bin/env python3
"""goethe uninstall in a fake HOME tree: refusals, models deleted by default, --keep-models, your own files kept."""

from __future__ import annotations

import shutil
import sqlite3
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


def _registered_models(goethe: Path, root: Path, env: dict[str, str]) -> None:
    """With a real state.db: a downloaded model is deleted, a folder of your own files and a link's target stay."""
    home = _fake_install(root / "registered")
    (home / "state.db").unlink()
    models = home / "models"
    (models / "org__m").rename(models / "org__dl")
    (models / "org__mine").mkdir()
    (models / "org__mine" / "w.safetensors").write_text("mine", encoding="utf-8")
    user_folder = root / "user-folder"
    user_folder.mkdir()
    (user_folder / "config.json").write_text("{}", encoding="utf-8")
    (models / "org__link").symlink_to(user_folder)
    con = sqlite3.connect(home / "state.db")
    con.execute(
        "CREATE TABLE installed_models (model_id TEXT PRIMARY KEY, local_path TEXT NOT NULL,"
        " architecture TEXT NOT NULL, static_vram_gb REAL NOT NULL, kv_bytes_per_token INTEGER NOT NULL,"
        " default_ctx INTEGER NOT NULL, default_max_conc INTEGER NOT NULL, default_tp INTEGER NOT NULL,"
        " installed_at DATETIME DEFAULT CURRENT_TIMESTAMP, kv_elems_per_token INTEGER, max_ctx INTEGER,"
        " num_attention_heads INTEGER, num_kv_heads INTEGER, attention TEXT, managed INTEGER NOT NULL DEFAULT 1)"
    )
    rows = (("org/dl", "models/org__dl", 1), ("org/mine", "models/org__mine", 0), ("org/link", "models/org__link", 0))
    for model_id, rel, managed in rows:
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,"
            " default_ctx, default_max_conc, default_tp, managed) VALUES (?, ?, 'x', 1, 1, 1, 1, 1, ?)",
            (model_id, rel, managed),
        )
    con.commit()
    con.close()
    rr = run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(home)})
    out = rr.stdout + rr.stderr
    if rr.returncode != 0:
        raise RuntimeError(out)
    if (models / "org__dl").exists() or (models / "org__link").exists() or (home / "state.db").exists():
        raise RuntimeError(f"downloaded model, link or state left behind: {out}")
    if not (models / "org__mine" / "w.safetensors").is_file() or not (user_folder / "config.json").is_file():
        raise RuntimeError(f"uninstall deleted your own files: {out}")
    if "delete 3 installed model(s)" not in out or f"keep your files in {user_folder}" not in out or "kept " not in out:
        raise RuntimeError(f"the plan must name what goes and what stays: {out}")


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
        if (inst / "models").exists():
            raise RuntimeError("uninstall left the installed models behind")
        keep = _fake_install(root / "keep")
        rr = run([str(goethe), "uninstall", "--keep-models", "--yes"], env={**env, "GOETHE_HOME": str(keep)})
        kept_models = (keep / "models" / "org__m" / "w.safetensors").is_file()
        if rr.returncode != 0 or not kept_models or (keep / "state.db").exists():
            raise RuntimeError(f"--keep-models must keep models and remove the rest: {rr.stderr or rr.stdout}")
        old = _fake_install(root / "old-flag")
        rr = run([str(goethe), "uninstall", "--purge-models", "--yes"], env={**env, "GOETHE_HOME": str(old)})
        if rr.returncode != 0 or (old / "models").exists():
            raise RuntimeError(f"the old --purge-models flag must still work: {rr.stderr or rr.stdout}")
        _registered_models(goethe, root, env)
        models_only = root / "models-only"
        (models_only / "models").mkdir(parents=True)
        (models_only / "models" / "keep").write_text("x", encoding="utf-8")
        linked = _fake_install(root / "linked")
        (linked / "bin" / "goethe").unlink()
        (linked / "bin" / "goethe").symlink_to("/bin/true")
        for name, home in (("models-only home", models_only), ("symlinked bin/goethe", linked)):
            rr = run([str(goethe), "uninstall", "--yes"], env={**env, "GOETHE_HOME": str(home)})
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
