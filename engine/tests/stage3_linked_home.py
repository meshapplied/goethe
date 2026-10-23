#!/usr/bin/env python3
"""Stage 3: a symlinked goethe without GOETHE_HOME finds its home from the binary location and marker (F6)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from lib.goethe_test import (
    home_goethe_env,
    main_exit,
    require_cli,
    run_goethe,
    run_temp_prefix,
    start_goethed,
    temp_home,
)


def _bare_run(cli: Path, user_home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the linked CLI like a fresh login shell would: no GOETHE_* variables and HOME somewhere else."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GOETHE_")}
    env.update({"HOME": str(user_home), "PATH": "/usr/bin:/bin"})
    return subprocess.run(
        [str(cli), *args], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
    )


def main() -> None:
    """goethed runs for home H; L/goethe -> H/bin/goethe with HOME=L must reach it and never create L/.goethe."""
    goethe = require_cli("goethe")
    goethed = require_cli("goethed")
    with temp_home(stub_python=False, tag="linked") as home:
        (home / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(goethe, home / "bin" / "goethe")
        shutil.copy2(goethed, home / "bin" / "goethed")
        (home / ".goethe-home").write_text("goethe home\n", encoding="utf-8")
        md = start_goethed(home)
        link_dir = Path(tempfile.mkdtemp(prefix=run_temp_prefix("link"), dir="/tmp"))
        try:
            run_goethe([str(goethe), "key", "create", "--name", "linked-key", "--raw"], env=home_goethe_env(home))
            linked = link_dir / "goethe"
            linked.symlink_to(home / "bin" / "goethe")
            ps = _bare_run(linked, link_dir, "ps", "--json")
            if ps.returncode != 0:
                raise RuntimeError(f"linked goethe ps found no goethed (exit {ps.returncode}): {ps.stderr[-300:]}")
            json.loads(ps.stdout.strip() or "[]")
            keys = _bare_run(linked, link_dir, "key", "list", "--json")
            names = [k.get("name") for k in json.loads(keys.stdout or "[]")] if keys.returncode == 0 else []
            if "linked-key" not in names:
                raise RuntimeError(f"linked goethe must use {home}: {keys.returncode} {names} {keys.stderr[-200:]}")
            stray = sorted(p.name for p in link_dir.iterdir() if p.name != "goethe")
            if stray:
                raise RuntimeError(f"linked goethe created {stray} under HOME instead of using {home}")
        finally:
            shutil.rmtree(link_dir, ignore_errors=True)
            md.stop()
    print("stage3_linked_home ok")


if __name__ == "__main__":
    main_exit(main)
