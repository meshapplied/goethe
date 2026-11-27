#!/usr/bin/env python3
"""Stage 3 in-place local installs: a directory of links, the user's folder untouched, remove keeps the files."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from lib.goethe_test import main_exit, require_cli, run, run_temp_prefix, sqlite_query, temp_home

FIXTURE_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "n_routed_experts": 8,
    "max_position_embeddings": 4096,
}


def main() -> None:
    """Install a local folder in place, check the link directory and the card, remove it and keep the user's files."""
    goethe = require_cli("goethe")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("registry"), dir="/tmp"))
    try:
        with temp_home(stub_python=True, tag="local") as home:
            model_src = work / "mymodel"
            model_src.mkdir()
            (model_src / "config.json").write_text(json.dumps(FIXTURE_CFG), encoding="utf-8")
            (model_src / "w.safetensors").write_bytes(b"x" * 128)
            env = {"GOETHE_HOME": str(home), "HF_HUB_OFFLINE": "1"}
            rr_inst = run(
                [
                    str(goethe), "install", str(model_src), "--name", "local/test-model",
                    "--", "--disable-shared-experts-fusion",
                ],
                env=env,
                timeout=60,
            )
            if rr_inst.returncode != 0:
                raise RuntimeError(rr_inst.stderr or rr_inst.stdout)
            link = home / "models" / "local__test-model"
            card = link / "goethe_card.toml"
            if not link.is_dir() or link.is_symlink() or not card.is_file() or card.is_symlink():
                raise RuntimeError(f"expected a directory of symlinks with goethe's own card at {link}")
            for name in ("config.json", "w.safetensors"):
                entry = link / name
                if not entry.is_symlink() or entry.resolve() != (model_src / name).resolve():
                    raise RuntimeError(f"{entry} must be a symlink to the source file")
            if (model_src / "goethe_card.toml").exists() or sorted(p.name for p in model_src.iterdir()) != [
                "config.json", "w.safetensors"
            ]:
                raise RuntimeError("in-place install modified the user's source folder")
            managed = sqlite_query(home / "state.db", "SELECT managed FROM installed_models")
            rr_rm = run([str(goethe), "remove", "local/test-model", "--yes"], env=env, timeout=30)
            if rr_rm.returncode != 0:
                raise RuntimeError(rr_rm.stderr or rr_rm.stdout)
            if not (model_src / "w.safetensors").is_file() or not (model_src / "config.json").is_file():
                raise RuntimeError("remove deleted files of a model registered in place")
            if link.exists() or link.is_symlink():
                raise RuntimeError("remove left goethe's link directory behind")
            if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
                raise RuntimeError("remove did not unregister the model")
            if managed != [(0,)]:
                raise RuntimeError(f"in-place local install must be stored managed=0, got {managed}")
            said = rr_rm.stdout + rr_rm.stderr
            if f"your files in {model_src} stay on disk" not in said or "rm -rf" in said:
                raise RuntimeError(f"remove must name the kept folder and suggest no delete: {said[-300:]}")
        print("stage3_registry ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
