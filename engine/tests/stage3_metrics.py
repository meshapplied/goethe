#!/usr/bin/env python3
"""Gate: Prometheus /metrics on loopback mux only."""

from __future__ import annotations

import re
import urllib.request

from lib.goethe_test import home_goethe_env, main_exit, require_cli, require_gpus, run_goethe, seed_installed_model, spawn, temp_home, test_mux_port, test_public_port, _acquire_gpu_flocks


def main() -> None:
    """Scrape mux /metrics and validate expected series."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    gpu = require_gpus(1)[0]
    with temp_home(stub_python=True) as home:
        model_dir = home / "models" / "org__m"
        model_dir.mkdir(parents=True)
        (model_dir / "goethe_card.toml").write_text(
            'model_id="m"\ncontext_length=512\nmax_running_requests=2\n',
            encoding="utf-8",
        )
        seed_installed_model(home, "m", str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1)
        goethed = spawn([str(require_cli("goethed"))], env=home_goethe_env(home))
        env = {**home_goethe_env(home), "CUDA_VISIBLE_DEVICES": str(gpu)}
        release = _acquire_gpu_flocks([gpu])
        try:
            run_goethe([str(goethe), "load", "m", "--gpus", str(gpu)], env=env, timeout=120)
            mux = test_mux_port(home)
            pub = test_public_port(home)
            with urllib.request.urlopen(f"http://127.0.0.1:{mux}/metrics", timeout=10) as resp:
                text = resp.read().decode("utf-8")
            if "# HELP goethe_up" not in text or "# TYPE goethe_up gauge" not in text:
                raise RuntimeError("goethe_up HELP/TYPE missing")
            for name in (
                "goethe_engine_status",
                "goethe_requests_total",
                "goethe_gpu_memory_total_bytes",
            ):
                if f"# HELP {name}" not in text:
                    raise RuntimeError(f"missing HELP for {name}")
            if not re.search(r"goethe_engine_status\{[^}]+\}\s+[01]", text):
                raise RuntimeError("goethe_engine_status sample missing")
            try:
                urllib.request.urlopen(f"https://127.0.0.1:{pub}/metrics", timeout=3)
                raise RuntimeError("nginx exposed /metrics")
            except Exception:
                pass
        finally:
            release()
            goethed.stop()
    print("stage3_metrics ok")


if __name__ == "__main__":
    main_exit(main)
