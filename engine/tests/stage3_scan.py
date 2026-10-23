#!/usr/bin/env python3
"""Stage 3 goethe scan: register folders moved into GOETHE_HOME/models, load, remove; gated-repo guidance."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

from lib.goethe_test import (
    assert_no_stub_engines,
    clean_test_gpu,
    gpu_flock,
    home_goethe_env,
    huggingface_reachable,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    sqlite_query,
    start_goethed,
    temp_home,
)

GATED = "meta-llama/Llama-3.2-1B"
_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
    "torch_dtype": "bfloat16",
}


def _model(models: Path, name: str, safetensors: bool) -> Path:
    """A minimal model folder: config plus one valid safetensors file, or a .bin-only checkpoint."""
    folder = models / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(_CFG), encoding="utf-8")
    if safetensors:
        header = json.dumps({"w": {"dtype": "BF16", "shape": [32], "data_offsets": [0, 64]}}).encode()
        header += b" " * (-len(header) % 8)
        (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(64))
    else:
        (folder / "pytorch_model.bin").write_bytes(b"x" * 64)
    return folder


def _check_scan(goethe: Path, home: Path, env: dict[str, str]) -> tuple[Path, Path]:
    """Non-tty scan exits 3 and lists findings; --yes adds the supported folder as a managed local model."""
    models = home / "models"
    good = _model(models, "tinyscan", True)
    bad = _model(models, "binonly", False)
    proc = subprocess.run(
        [str(goethe), "scan"],
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 3 or "Model found: tinyscan (LlamaForCausalLM" not in out or "re-run with --yes" not in out:
        raise RuntimeError(f"non-tty scan should exit 3 and list the model: rc={proc.returncode} {out[-500:]}")
    if "Not supported: binonly: only safetensors checkpoints are supported" not in out:
        raise RuntimeError(f"unsupported folder not reported with its reason: {out[-500:]}")
    if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
        raise RuntimeError("scan without --yes registered a model")
    run_goethe([str(goethe), "scan", "--yes"], env=env, timeout=120)
    rows = sqlite_query(home / "state.db", "SELECT model_id, local_path, managed FROM installed_models")
    if rows != [("local/tinyscan", "models/tinyscan", 1)]:
        raise RuntimeError(f"scan --yes registered {rows}")
    if not (good / "goethe_card.toml").is_file():
        raise RuntimeError("scan did not write the card next to the weights")
    rr = run_goethe([str(goethe), "scan", "--json"], env=env, timeout=120)
    status = {Path(r["dir"]).name: (r["status"], r.get("reason")) for r in json.loads(rr.stdout)}
    if status.get("tinyscan", ("", None))[0] != "registered":
        raise RuntimeError(f"second scan should report tinyscan as registered: {status}")
    if status.get("binonly") != ("unsupported", "only safetensors checkpoints are supported"):
        raise RuntimeError(f"binonly should stay unsupported: {status}")
    return good, bad


def main() -> None:
    """Scan, load and unload with the stub engine, remove, then check the gated-repo guidance."""
    goethe = require_cli("goethe")
    clean = clean_test_gpu()
    gpu = clean if clean is not None else roomiest_test_gpu()
    yes = [] if clean is not None else ["--yes"]
    gated = "skipped (huggingface.co unreachable)"
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="scan") as home:
        env = home_goethe_env(home)
        good, bad = _check_scan(goethe, home, env)
        daemon = start_goethed(home)
        try:
            run_goethe([str(goethe), "load", "local/tinyscan", "--gpus", str(gpu), *yes], env=env, timeout=180)
            rows = json.loads(run_goethe([str(goethe), "ps", "--json"], env=env).stdout)
            if not any(r.get("model_id") == "local/tinyscan" and r.get("status") == "RUNNING" for r in rows):
                raise RuntimeError(f"scanned model is not RUNNING after load: {rows}")
            run_goethe([str(goethe), "unload", "local/tinyscan"], env=env, timeout=120)
        finally:
            daemon.stop()
        assert_no_stub_engines(home)
        run_goethe([str(goethe), "remove", "local/tinyscan", "--yes"], env=env, timeout=60)
        if good.exists():
            raise RuntimeError("remove kept the scanned (managed) model folder")
        if not bad.is_dir():
            raise RuntimeError("remove touched an unrelated folder")
        if huggingface_reachable(3.0):
            target = home / "models" / "meta-llama__Llama-3.2-1B"
            want = (
                f"{GATED} is gated (or private) on Hugging Face and goethe does not log in. Download it yourself "
                f"after accepting its license, for example: hf download {GATED} --local-dir {target} . "
                "Then run: goethe scan"
            )
            rr = run([str(goethe), "install", GATED, "--no-load"], env=env, timeout=120)
            if rr.returncode != 1 or want not in rr.stderr:
                raise RuntimeError(f"gated install guidance missing: rc={rr.returncode} {rr.stderr[-500:]}")
            if target.exists():
                raise RuntimeError("gated install created a model folder")
            gated = "ok"
    print(f"stage3_scan ok gpu={gpu} gated={gated}")


if __name__ == "__main__":
    main_exit(main)
