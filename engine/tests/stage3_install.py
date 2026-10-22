#!/usr/bin/env python3
"""Stage 3 goethe install without load gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

from lib.goethe_test import (
    gpu_mem_snapshot,
    main_exit,
    require_cli,
    resolve_python,
    run_goethe,
    spec_ddl_tables,
    sqlite_query,
    temp_home,
    test_gpus,
)


def _kv_elem_bytes(kv_cache_dtype: str, cfg: dict) -> int:
    """Element width for KV cache from card kv_cache_dtype."""
    key = kv_cache_dtype.strip().lower()
    if key in ("fp8_e4m3", "fp8_e5m2"):
        return 1
    if key in ("fp16", "float16", "bf16", "bfloat16"):
        return 2
    if key == "auto":
        dtype = str(cfg.get("torch_dtype", "float16"))
        return 2 if "16" in dtype else 4
    return 2


def _kv_bytes_per_token(cfg: dict, kv_cache_dtype: str) -> int:
    """Compute kv bytes per token using card kv_cache_dtype and config shape."""
    layers = int(cfg.get("num_hidden_layers", cfg.get("n_layer", 1)))
    kv_heads = int(cfg.get("num_key_value_heads", cfg.get("n_head", 1)))
    head_dim = int(cfg.get("head_dim", cfg.get("hidden_size", 64) // max(1, kv_heads)))
    return 2 * layers * kv_heads * head_dim * _kv_elem_bytes(kv_cache_dtype, cfg)


def _card_kv_cache_dtype(card_path: Path) -> str:
    """Read kv_cache_dtype from goethe_card.toml."""
    for line in card_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("kv_cache_dtype"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "auto"


def main() -> None:
    """Install tiny HF models via registry hit and miss paths."""
    goethe = require_cli("goethe")
    model = os.environ.get("GOETHE_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
    pre = gpu_mem_snapshot(test_gpus())
    with temp_home(stub_python=True, real_python=resolve_python()) as home:
        reg_hit = home / "registry_hit.json"
        reg_hit.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": model,
                            "architecture": "dense",
                            "static_vram_gb": 0.5,
                            "kv_bytes_per_token": 1024,
                            "min_total_vram_gb": 1.0,
                            "supported_sm": ["12.0"],
                            "default_ctx": 512,
                            "default_max_conc": 2,
                            "default_tp": 1,
                            "kv_cache_dtype": "fp8_e4m3",
                            "extra_sglang_args": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_goethe(
            [str(goethe), "install", model, "--no-load", "--ctx", "512", "--max-conc", "2"],
            env={
                "GOETHE_HOME": str(home),
                "GOETHE_REGISTRY_URL": f"file://{reg_hit}",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
            timeout=600,
        )
        post = gpu_mem_snapshot(test_gpus())
        for idx in test_gpus():
            if post[idx] - pre[idx] > 300:
                raise RuntimeError(f"VRAM grew on GPU {idx} during --no-load install")
        db = home / "state.db"
        if str(sqlite_query(db, "PRAGMA journal_mode")[0][0]).lower() != "wal":
            raise RuntimeError("state.db not WAL")
        for table, cols in spec_ddl_tables().items():
            names = [r[1] for r in sqlite_query(db, f"PRAGMA table_info({table})")]
            for col in cols:
                if col not in names:
                    raise RuntimeError(f"{table} missing {col}")
        model_dirs = list((home / "models").glob("*__*"))
        if not model_dirs:
            raise RuntimeError("model directory missing")
        cfg = json.loads((model_dirs[0] / "config.json").read_text(encoding="utf-8"))
        safes = sum(p.stat().st_size for p in model_dirs[0].glob("*.safetensors"))
        row = sqlite_query(db, "SELECT static_vram_gb, kv_bytes_per_token FROM installed_models LIMIT 1")[0]
        disk_gb = safes / (1024**3)
        if abs(float(row[0]) - disk_gb) / max(disk_gb, 1e-6) > 0.02:
            raise RuntimeError("static_vram_gb not within 2 percent of safetensors size")
        card_path = model_dirs[0] / "goethe_card.toml"
        expected_kv = _kv_bytes_per_token(cfg, _card_kv_cache_dtype(card_path))
        if abs(int(row[1]) - expected_kv) / max(expected_kv, 1) > 0.05:
            raise RuntimeError("kv_bytes_per_token mismatch vs card kv_cache_dtype")
        reg_miss = home / "registry_miss.json"
        reg_miss.write_text(json.dumps({"version": 1, "models": []}), encoding="utf-8")
        miss_model = "hf-internal-testing/tiny-random-gpt2"
        run_goethe(
            [str(goethe), "install", miss_model, "--no-load"],
            env={"GOETHE_HOME": str(home), "GOETHE_REGISTRY_URL": f"file://{reg_miss}"},
            timeout=600,
        )
        n = sqlite_query(db, "SELECT COUNT(*) FROM installed_models")[0][0]
        if int(n) < 2:
            raise RuntimeError("registry miss install did not register model")
    print("stage3_install ok")


if __name__ == "__main__":
    main_exit(main)
