#!/usr/bin/env python3
"""Stage 4: install --load waits for RUNNING before returning."""

from __future__ import annotations

import json
import time

from lib.goethe_test import (
    assert_no_stub_engines,
    clean_test_gpu,
    gpu_flock,
    home_goethe_env,
    http_post,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run_goethe,
    start_goethed,
    stub_launch,
    temp_home,
    test_mux_port,
)

START_DELAY = 4.0
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
    """Install with --load must return only once the (start-delayed) stub serves chat."""
    goethe = require_cli("goethe")
    clean = clean_test_gpu()
    gpu = clean if clean is not None else roomiest_test_gpu()
    yes = [] if clean is not None else ["--yes"]
    with gpu_flock([gpu]), temp_home(stub_python=True) as home:
        src = home / "src_model"
        src.mkdir()
        (src / "config.json").write_text(json.dumps(_CFG), encoding="utf-8")
        (src / "w.safetensors").write_bytes(b"x" * 128)
        (src / "stub_start_delay.txt").write_text(str(START_DELAY), encoding="utf-8")
        md = start_goethed(home)
        env = home_goethe_env(home)
        t0 = time.time()
        run_goethe(
            [str(goethe), "install", str(src), "--name", "inst", "--load", "--gpus", str(gpu), *yes],
            env=env,
            timeout=120,
        )
        elapsed = time.time() - t0
        engine_dir = next(d for d in (src, *(home / "models").iterdir()) if (d / "stub_launch_argv.json").is_file())
        launch = stub_launch(engine_dir)
        if launch["knobs"].get("STUB_START_DELAY") != str(START_DELAY):
            raise RuntimeError(f"stub did not pick up its start delay: {launch['knobs']}")
        if launch["listening_at"] - launch["started_at"] < START_DELAY - 0.2:
            raise RuntimeError("stub start delay did not take effect")
        if elapsed < START_DELAY - 0.5:
            raise RuntimeError(f"install --load returned after {elapsed:.1f}s, before the engine listened")
        token = next(
            ln
            for ln in run_goethe([str(goethe), "key", "create", "--name", "t", "--raw"], env=env).stdout.splitlines()
            if ln.startswith("goethe_sk_")
        )
        body = json.dumps({"model": "inst", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
        code, out = http_post(
            f"http://127.0.0.1:{test_mux_port(home)}/v1/chat/completions",
            body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if code != 200 or b"model=inst" not in out:
            raise RuntimeError(f"chat right after install --load failed code={code} body={out[:120]!r}")
        run_goethe([str(goethe), "unload", "--all"], env=env, timeout=60)
        md.stop()
        assert_no_stub_engines(home)
        print(f"stage4_install_load_stub ok gpu={gpu} install_load={elapsed:.1f}s")


if __name__ == "__main__":
    main_exit(main)
