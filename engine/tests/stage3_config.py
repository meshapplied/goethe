#!/usr/bin/env python3
"""Stage 3 goethe config gate: card and installed_models defaults update in place, idempotently."""

from __future__ import annotations

import re

from lib.goethe_test import main_exit, require_cli, run_goethe, seed_installed_model, sqlite_query, temp_home

ROW_SQL = "SELECT default_ctx, default_max_conc FROM installed_models WHERE model_id='demo'"


def _card_int(text: str, key: str) -> int | None:
    m = re.search(rf"^\s*{key}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def main() -> None:
    """Run goethe config twice and verify the card and DB carry the new defaults."""
    goethe = require_cli("goethe")
    with temp_home(stub_python=True) as home:
        card = home / "models" / "org__demo" / "goethe_card.toml"
        card.parent.mkdir(parents=True)
        card.write_text('model_id = "demo"\ncontext_length = 1024\nmax_running_requests = 4\n', encoding="utf-8")
        seed_installed_model(home, "demo", "models/org__demo", 1.0, 1000, 1024, 4, 1, max_ctx=4096)
        env = {"GOETHE_HOME": str(home)}
        run_goethe([str(goethe), "config", "demo", "--ctx", "2048", "--max-conc", "8"], env=env)
        text1 = card.read_text(encoding="utf-8")
        if _card_int(text1, "context_length") != 2048 or _card_int(text1, "max_running_requests") != 8:
            raise RuntimeError(f"card not updated in place: {text1!r}")
        if 'model_id = "demo"' not in text1 and 'model_id="demo"' not in text1:
            raise RuntimeError("config dropped model_id from the card")
        row = sqlite_query(home / "state.db", ROW_SQL)
        if row != [(2048, 8)]:
            raise RuntimeError(f"installed_models defaults not updated: {row}")
        run_goethe([str(goethe), "config", "demo", "--ctx", "2048", "--max-conc", "8"], env=env)
        if card.read_text(encoding="utf-8") != text1 or sqlite_query(home / "state.db", ROW_SQL) != [(2048, 8)]:
            raise RuntimeError("config not idempotent")
    print("stage3_config ok")


if __name__ == "__main__":
    main_exit(main)
