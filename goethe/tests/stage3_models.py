#!/usr/bin/env python3
"""Stage 3 goethe models and remove (offline)."""

from __future__ import annotations

import json

from lib.goethe_test import (
    main_exit,
    require_cli,
    resolve_python,
    run_goethe,
    seed_installed_model,
    signed_registry,
    sqlite_query,
    temp_home,
)


def main() -> None:
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, real_python=resolve_python(), tag="models") as home:
        reg = home / "reg.json"
        reg.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": "local/test-a",
                            "architecture": "dense",
                            "static_vram_gb": 0.1,
                            "kv_bytes_per_token": 100,
                            "min_total_vram_gb": 1,
                            "supported_sm": ["12.0"],
                            "default_ctx": 512,
                            "default_max_conc": 1,
                            "default_tp": 1,
                            "kv_cache_dtype": "fp8_e4m3",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        model_path = home / "models" / "local__test-a"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "goethe_card.toml").write_text(
            'model_id = "local/test-a"\ncontext_length = 512\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "local/test-a", "models/local__test-a", 0.1, 100, 512, 1, 1, managed=0)
        managed_path = home / "models" / "org__managed-b"
        managed_path.mkdir(parents=True)
        (managed_path / "config.json").write_text("{}", encoding="utf-8")
        seed_installed_model(home, "org/managed-b", "models/org__managed-b", 0.1, 100, 512, 1, 1, managed=1)
        env = {"GOETHE_HOME": str(home), **signed_registry(reg), "HF_HUB_OFFLINE": "1"}
        rows = json.loads(run_goethe([str(goethe), "models", "--json", "--installed"], env=env, timeout=30).stdout)
        row_a = next((r for r in rows if r.get("id") == "local/test-a"), None)
        if row_a is None or row_a.get("status") != "installed" or not row_a.get("custom"):
            raise RuntimeError("models --installed missing seeded custom model")
        rr_rm = run_goethe([str(goethe), "remove", "local/test-a", "--yes"], env=env, timeout=30)
        if not model_path.exists():
            raise RuntimeError("remove deleted the files of an unmanaged (in-place) model")
        rr_m = run_goethe([str(goethe), "remove", "org/managed-b", "--yes"], env=env, timeout=30)
        if managed_path.exists():
            raise RuntimeError(f"remove kept the files of a managed model: {rr_m.stdout}")
        if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
            raise RuntimeError("remove did not clear installed_models")
        said = rr_rm.stdout + rr_rm.stderr
        if f"your files in {model_path} stay on disk" not in said or "rm -rf" in said:
            raise RuntimeError(f"unmanaged remove must name the kept folder and suggest no delete: {said[-300:]}")

    print("stage3_models ok")


if __name__ == "__main__":
    main_exit(main)
