#!/usr/bin/env python3
"""Stage 4: install --load waits for RUNNING before returning."""

from __future__ import annotations

import json
import time
from pathlib import Path

from lib.goethe_test import (
    Proc,
    _acquire_gpu_flocks,
    assert_no_stub_engines,
    home_goethe_env,
    http_post,
    main_exit,
    require_cli,
    require_gpus,
    run_goethe,
    spawn,
    temp_home,
    test_mux_port,
    wait_goethed_socket,
)

DEV_PY = Path("/scratch/ternary-flash/goethe-home/py-env/bin/python")

_CFG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
}


def main() -> None:
    """Install with --load must serve chat immediately (stub with start delay)."""
    goethe = require_cli("goethe")
    require_cli("goethed")
    gpu = require_gpus(1)[0]
    release = _acquire_gpu_flocks([gpu])
    dev_py = DEV_PY if DEV_PY.is_file() else None
    md: Proc | None = None
    try:
        with temp_home(stub_python=True, real_python=dev_py) as home:
            src = home / "src_model"
            src.mkdir()
            (src / "config.json").write_text(json.dumps(_CFG), encoding="utf-8")
            (src / "w.safetensors").write_bytes(b"x" * 128)
            env = {
                **home_goethe_env(home),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "STUB_START_DELAY": "4",
            }
            md = spawn([str(require_cli("goethed"))], env=env, label="goethed")
            wait_goethed_socket(home)
            t0 = time.time()
            run_goethe(
                [str(goethe), "install", str(src), "--name", "inst", "--load", "--gpus", str(gpu)],
                env=env,
                timeout=120,
            )
            if time.time() - t0 < 3.5:
                raise RuntimeError("install --load returned before stub start delay elapsed")
            token = run_goethe(
                [str(goethe), "key", "create", "--name", "t", "--raw"],
                env=env,
                timeout=30,
            ).stdout.splitlines()
            token = next(ln for ln in token if ln.startswith("goethe_sk_"))
            mux = test_mux_port(home)
            body = json.dumps(
                {"model": "inst", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
            ).encode()
            code, out = http_post(
                f"http://127.0.0.1:{mux}/v1/chat/completions",
                body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=30,
            )
            if code != 200:
                raise RuntimeError(f"chat after install --load failed code={code} body={out[:120]!r}")
            run_goethe([str(goethe), "unload", "--all"], env=env, timeout=60)
            md.stop()
            assert_no_stub_engines(home)
        print("stage4_install_load_stub ok")
    finally:
        try:
            if md is not None:
                md.stop()
        except Exception:
            pass
        release()


if __name__ == "__main__":
    main_exit(main)
