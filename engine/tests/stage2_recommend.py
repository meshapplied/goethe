#!/usr/bin/env python3
"""Stage 2 goethe recommend formula and registry filtering gate."""

from __future__ import annotations

import json
from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run_goethe, temp_home, vram_per_gpu_gb


def main() -> None:
    """Run recommend against a synthetic registry and hardware_state."""
    goethe = require_cli("goethe")
    registry = {
        "version": 1,
        "models": [
            {
                "id": "meshapplied/tiny-a",
                "architecture": "dense",
                "static_vram_gb": 10.0,
                "kv_bytes_per_token": 1000,
                "min_total_vram_gb": 20.0,
                "supported_sm": ["12.0"],
                "default_ctx": 4096,
                "default_max_conc": 8,
                "default_tp": 1,
                "kv_cache_dtype": "fp8_e4m3",
                "extra_sglang_args": [],
            },
            {
                "id": "meshapplied/tiny-b",
                "architecture": "dense",
                "static_vram_gb": 200.0,
                "kv_bytes_per_token": 1000,
                "min_total_vram_gb": 400.0,
                "supported_sm": ["12.0"],
                "default_ctx": 4096,
                "default_max_conc": 8,
                "default_tp": 2,
                "kv_cache_dtype": "fp8_e4m3",
                "extra_sglang_args": [],
            },
            {
                "id": "meshapplied/old-sm",
                "architecture": "dense",
                "static_vram_gb": 5.0,
                "kv_bytes_per_token": 500,
                "min_total_vram_gb": 8.0,
                "supported_sm": ["8.0"],
                "default_ctx": 2048,
                "default_max_conc": 4,
                "default_tp": 1,
                "kv_cache_dtype": "fp8_e4m3",
                "extra_sglang_args": [],
            },
        ],
    }
    hw = {
        "profiled_at": "test",
        "driver": "test",
        "gpus": [
            {
                "index": 3,
                "uuid": "u3",
                "name": "GPU3",
                "compute_capability": "12.0",
                "vram_total_mb": 96000,
                "vram_used_mb": 50000,
                "vram_free_mb": 46000,
                "pcie_gen": 4,
                "pcie_width": 16,
                "running_processes": [{"pid": 999888, "used_vram_mb": 50000}],
            },
            {
                "index": 4,
                "uuid": "u4",
                "name": "GPU4",
                "compute_capability": "12.0",
                "vram_total_mb": 96000,
                "vram_used_mb": 0,
                "vram_free_mb": 96000,
                "pcie_gen": 4,
                "pcie_width": 16,
                "running_processes": [],
            },
        ],
        "links": [{"a": 3, "b": 4, "type": "NVLINK"}],
    }
    with temp_home(stub_python=False) as home:
        reg_path = home / "registry.json"
        reg_path.write_text(json.dumps(registry), encoding="utf-8")
        (home / "hardware_state.json").write_text(json.dumps(hw), encoding="utf-8")
        rr = run_goethe(
            [str(goethe), "recommend", "--json"],
            env={
                "GOETHE_HOME": str(home),
                "GOETHE_REGISTRY_URL": f"file://{reg_path}",
            },
        )
        out = json.loads(rr.stdout)
        rec_ids = {m["model_id"] for m in out.get("recommended", [])}
        exc = {m["model_id"]: m.get("reason", "") for m in out.get("excluded", [])}
        if "meshapplied/tiny-a" not in rec_ids:
            raise RuntimeError("expected tiny-a recommended")
        if "meshapplied/old-sm" not in exc:
            raise RuntimeError("expected old-sm excluded for supported_sm")
        if "meshapplied/tiny-b" not in exc and "meshapplied/tiny-b" not in rec_ids:
            raise RuntimeError("expected tiny-b tp=2 placement or exclusion")
        if "meshapplied/tiny-b" in exc:
            reason = exc["meshapplied/tiny-b"].lower()
            if "vram" not in reason and "memory" not in reason and "gpu" not in reason:
                raise RuntimeError(f"tiny-b exclusion reason unexpected: {reason}")
        for item in out.get("recommended", []):
            if item["model_id"] != "meshapplied/tiny-a":
                continue
            expected = vram_per_gpu_gb(10.0, 4096, 1000, 8, item.get("tp", 1))
            if abs(float(item["vram_per_gpu_gb"]) - expected) > 0.05:
                raise RuntimeError(f"vram_per_gpu_gb {item['vram_per_gpu_gb']} != {expected}")
        tp2 = next((m for m in out.get("recommended", []) if m["model_id"] == "meshapplied/tiny-b"), None)
        if tp2 is not None:
            if int(tp2.get("tp", 1)) != 2 or len(tp2.get("gpus", [])) != 2:
                raise RuntimeError("tiny-b should recommend tp=2 on two GPUs when not excluded")
        print("stage2_recommend ok")


if __name__ == "__main__":
    main_exit(main)
