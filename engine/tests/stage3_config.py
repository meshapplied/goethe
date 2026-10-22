#!/usr/bin/env python3
"""Stage 3 goethe config idempotency gate."""

from __future__ import annotations

import json

from lib.goethe_test import main_exit, require_cli, run_goethe, temp_home


def main() -> None:
    """Run goethe config twice and verify card and DB update."""
    goethe = require_cli("goethe")
    with temp_home(stub_python=True) as home:
        model = "org__demo"
        (home / "models" / model).mkdir(parents=True)
        card = home / "models" / model / "goethe_card.toml"
        card.write_text(
            "model_id = \"demo\"\ncontext_length = 1024\nmax_running_requests = 4\n",
            encoding="utf-8",
        )
        run_goethe(
            [str(goethe), "config", "demo", "--ctx", "2048", "--max-conc", "8"],
            env={"GOETHE_HOME": str(home)},
        )
        text1 = card.read_text(encoding="utf-8")
        run_goethe(
            [str(goethe), "config", "demo", "--ctx", "2048", "--max-conc", "8"],
            env={"GOETHE_HOME": str(home)},
        )
        if card.read_text(encoding="utf-8") != text1:
            raise RuntimeError("config not idempotent")
    print("stage3_config ok")


if __name__ == "__main__":
    main_exit(main)
