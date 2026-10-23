#!/usr/bin/env python3
"""Gate: model alias set, list, remove, and models vs list output."""

from __future__ import annotations

import subprocess

from lib.goethe_test import main_exit, require_cli, run, run_goethe, seed_installed_model, temp_home


def main() -> None:
    """Exercise alias commands against a seeded installed model."""
    goethe = require_cli("goethe")
    alias_name = "fast"
    model_id = "alias-m"
    with temp_home(stub_python=True, tag="alias") as home:
        env = {"GOETHE_HOME": str(home)}
        model_dir = home / "models" / "org__alias-m"
        model_dir.mkdir(parents=True)
        (model_dir / "goethe_card.toml").write_text(
            f'model_id="{model_id}"\ncontext_length=512\n',
            encoding="utf-8",
        )
        seed_installed_model(home, model_id, str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1)
        run_goethe([str(goethe), "alias", "set", alias_name, model_id], env=env)
        lst = subprocess.check_output([str(goethe), "alias", "list"], env=env, text=True)
        if f"{alias_name}\t{model_id}" not in lst:
            raise RuntimeError(f"alias list missing row: {lst!r}")
        rr_bad = run([str(goethe), "alias", "set", alias_name, "missing-model"], env=env)
        if rr_bad.returncode == 0:
            raise RuntimeError("alias set to missing model should fail")
        run_goethe([str(goethe), "alias", "remove", alias_name], env=env)
        lst2 = subprocess.check_output([str(goethe), "alias", "list"], env=env, text=True)
        if alias_name in lst2:
            raise RuntimeError(f"alias still listed: {lst2!r}")
        out_models = subprocess.check_output([str(goethe), "models"], env=env, text=True)
        out_list = subprocess.check_output([str(goethe), "list"], env=env, text=True)
        if out_models != out_list:
            raise RuntimeError("goethe list and goethe models differ")
    print("stage3_alias ok")


if __name__ == "__main__":
    main_exit(main)
