#!/usr/bin/env python3
"""Gate: Prometheus /metrics on loopback mux only."""

from __future__ import annotations

import re
import urllib.request

from lib.mesh_test import home_mesh_env, main_exit, require_cli, require_gpus, run_mesh, seed_installed_model, spawn, temp_home, test_mux_port, test_public_port, _acquire_gpu_flocks


def main() -> None:
    """Scrape mux /metrics and validate expected series."""
    mesh = require_cli("mesh")
    require_cli("meshd")
    gpu = require_gpus(1)[0]
    with temp_home(stub_python=True) as home:
        model_dir = home / "models" / "org__m"
        model_dir.mkdir(parents=True)
        (model_dir / "mesh_card.toml").write_text(
            'model_id="m"\ncontext_length=512\nmax_running_requests=2\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "m", str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1)
        meshd = spawn([str(require_cli("meshd"))], env=home_mesh_env(home))
        env = {**home_mesh_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
        release = _acquire_gpu_flocks([gpu])
        try:
            run_mesh([str(mesh), "load", "m", "--gpus", str(gpu)], env=env, timeout=120)
            mux = test_mux_port(home)
            pub = test_public_port(home)
            with urllib.request.urlopen(f"http://127.0.0.1:{mux}/metrics", timeout=10) as resp:
                text = resp.read().decode("utf-8")
            if "# HELP mesh_up" not in text or "# TYPE mesh_up gauge" not in text:
                raise RuntimeError("mesh_up HELP/TYPE missing")
            for name in (
                "mesh_engine_status",
                "mesh_requests_total",
                "mesh_gpu_memory_total_bytes",
            ):
                if f"# HELP {name}" not in text:
                    raise RuntimeError(f"missing HELP for {name}")
            if not re.search(r"mesh_engine_status\{[^}]+\}\s+[01]", text):
                raise RuntimeError("mesh_engine_status sample missing")
            try:
                urllib.request.urlopen(f"https://127.0.0.1:{pub}/metrics", timeout=3)
                raise RuntimeError("nginx exposed /metrics")
            except Exception:
                pass
        finally:
            release()
            meshd.stop()
    print("stage3_metrics ok")


if __name__ == "__main__":
    main_exit(main)
