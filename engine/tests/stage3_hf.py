#!/usr/bin/env python3
"""Stage 3 Hugging Face install against the real Hub: cache fast path, auth hints, endpoint and disk preflight."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from lib.goethe_test import (
    Skip,
    huggingface_reachable,
    main_exit,
    require_cli,
    run,
    run_goethe,
    temp_home,
)


def _registry_for(model: str) -> str:
    entry = {
        "id": model, "architecture": "dense", "static_vram_gb": 0.01, "kv_bytes_per_token": 512,
        "min_total_vram_gb": 1.0, "supported_sm": ["12.0"], "default_ctx": 128, "default_max_conc": 1,
        "default_tp": 1, "kv_cache_dtype": "fp8_e4m3", "extra_sglang_args": [],
    }
    return json.dumps({"version": 1, "models": [entry]})


def _msg(rr: object) -> str:
    return (getattr(rr, "stderr", "") or getattr(rr, "stdout", "") or "").lower()


def main() -> None:
    """Install a tiny public model, then exercise the failure paths that need huggingface.co."""
    if not huggingface_reachable(3.0):
        raise Skip("huggingface.co unreachable")
    goethe = require_cli("goethe")
    tiny = os.environ.get("GOETHE_TEST_TINY_MODEL", "hf-internal-testing/tiny-random-gpt2")
    disk_model = os.environ.get("GOETHE_TEST_RESUME_MODEL", "distilbert/distilgpt2")
    with temp_home(stub_python=True, tag="hf") as home:
        reg = home / "registry.json"
        reg.write_text(_registry_for(tiny), encoding="utf-8")
        base = {"GOETHE_HOME": str(home), "HF_HOME": str(home / "hf-home")}
        env = {**base, "GOETHE_REGISTRY_URL": f"file://{reg}"}
        t0 = time.time()
        for attempt in range(3):
            try:
                run_goethe([str(goethe), "install", tiny, "--no-load"], env=env, timeout=600)
                break
            except RuntimeError as exc:
                if attempt >= 2 or "cannot reach" not in str(exc).lower():
                    raise
                time.sleep(5.0)
        first = time.time() - t0
        mdir = next((home / "models").glob("*__*"))
        if not (mdir / "config.json").is_file() or not list(mdir.glob("*.safetensors")):
            raise RuntimeError(f"install did not place config and weights under {mdir}")
        t1 = time.time()
        run_goethe([str(goethe), "install", tiny, "--no-load"], env={**env, "HF_HUB_OFFLINE": "1"}, timeout=120)
        second = time.time() - t1
        if second > max(15.0, first * 0.5):
            raise RuntimeError(f"second install not fast enough ({second:.1f}s vs first {first:.1f}s)")
        rr = run([str(goethe), "install", "meta-llama/Llama-3.2-1B", "--no-load"], env=env, timeout=45)
        if rr.returncode == 0 or "goethe login" not in _msg(rr):
            raise RuntimeError(f"gated install must fail with a goethe login hint: {_msg(rr)[-300:]}")
        rr404 = run([str(goethe), "install", "no-such-org/no-such-model-xyz", "--no-load"], env=base, timeout=60)
        if rr404.returncode == 0 or ("check the id" not in _msg(rr404) and "goethe login" not in _msg(rr404)):
            raise RuntimeError(f"missing repo must fail with check-the-id or login hint: {_msg(rr404)[-300:]}")
        rr_net = run(
            [str(goethe), "install", tiny, "--no-load", "--force"],
            env={**env, "HF_ENDPOINT": "http://127.0.0.1:9"},
            timeout=60,
        )
        if rr_net.returncode == 0 or ("cannot reach" not in _msg(rr_net) and "network" not in _msg(rr_net)):
            raise RuntimeError(f"bad HF_ENDPOINT must fail naming the network: {_msg(rr_net)[-300:]}")
        reg2 = home / "registry_disk.json"
        reg2.write_text(_registry_for(disk_model), encoding="utf-8")
        for p in (home / "models").glob("*distil*"):
            shutil.rmtree(p, ignore_errors=True)
        rr_disk = run(
            [str(goethe), "install", disk_model, "--no-load"],
            env={**base, "GOETHE_REGISTRY_URL": f"file://{reg2}", "GOETHE_TEST_FREE_BYTES": "1"},
            timeout=120,
        )
        if rr_disk.returncode == 0 or "need" not in _msg(rr_disk) or "available" not in _msg(rr_disk):
            raise RuntimeError(f"disk preflight must show need and available GB: {_msg(rr_disk)[-300:]}")
        if any(Path(p).is_file() for p in (home / "models").rglob("*.safetensors") if "distil" in str(p)):
            raise RuntimeError("disk preflight failure still downloaded weights")
        rr_bad = run(
            [str(goethe), "login", "--token-stdin"], env=base, timeout=30, input_text="hf_not_a_real_token\n"
        )
        if rr_bad.returncode == 0 or "invalid" not in _msg(rr_bad):
            raise RuntimeError(f"bad token login must fail saying invalid: {_msg(rr_bad)[-300:]}")
        if (home / "hf_token").exists() or any(home.glob("*token*")):
            raise RuntimeError("a rejected token was stored")
    print("stage3_hf ok")


if __name__ == "__main__":
    main_exit(main)
