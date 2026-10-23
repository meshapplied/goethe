#!/usr/bin/env python3
"""Stage 3 registry entry/add/validate offline tests and in-place (unmanaged) local installs."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    Skip,
    dev_home,
    link_dev_pyenv,
    main_exit,
    require_cli,
    run,
    run_temp_prefix,
    sqlite_query,
    temp_home,
)

FIXTURE_CFG = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "n_routed_experts": 8,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:
        return

    def _reply(self, code: int, body: bytes = b"") -> None:
        """Send a JSON response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Model info with blobs, config.json and organization probes."""
        if "/api/models/" in self.path and "blobs=true" in self.path:
            if "stub/missing" in self.path:
                self._reply(404)
                return
            info = {"gated": False, "sha": "abc123", "siblings": [{"rfilename": "model.safetensors", "size": 1000}]}
            self._reply(200, json.dumps(info).encode())
        elif self.path.endswith("config.json"):
            self._reply(200, json.dumps(FIXTURE_CFG).encode())
        elif "/api/organizations/" in self.path:
            self._reply(200, b"{}")
        else:
            self._reply(404)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def main() -> None:
    """Exercise registry tooling against a fake Hub and a temp home that uses the dev SGLang read-only."""
    goethe = require_cli("goethe")
    if not (dev_home() / "py-env" / "bin" / "python").is_file():
        raise Skip(f"dev py-env missing under {dev_home()}; --check-sglang needs the real fork")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("registry"), dir="/tmp"))
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with temp_home(tag="registry-sglang", pyenv=False) as sg_home:
            link_dev_pyenv(sg_home)
            sg_env = {"GOETHE_HOME": str(sg_home), "HF_HUB_OFFLINE": "1"}
            reg_path = GOETHE_PKG / "cdn" / "v1" / "registry.json"
            rr = run([str(goethe), "registry", "validate", f"file://{reg_path}", "--check-sglang"], env=sg_env)
            if rr.returncode != 0 or "skipped" in (rr.stdout + rr.stderr):
                raise RuntimeError(f"shipped registry must validate against the real fork: {rr.stderr or rr.stdout}")
            ids_file = work / "ids.txt"
            ids_file.write_text("stub/moe\nstub/missing\n", encoding="utf-8")
            hf_env = {"HF_ENDPOINT": endpoint, "HF_HUB_OFFLINE": "0", "GOETHE_HOME": str(sg_home)}
            entry_cmd = [str(goethe), "registry", "entry", "--ids-file", str(ids_file)]
            if run([*entry_cmd, "--strict"], env=hf_env, timeout=60).returncode == 0:
                raise RuntimeError("strict bulk should fail when any id fails")
            rr_bulk = run(entry_cmd, env=hf_env, timeout=60)
            if rr_bulk.returncode != 0 or len(json.loads(rr_bulk.stdout).get("models", [])) < 1:
                raise RuntimeError(f"bulk entry produced no models: {rr_bulk.stderr}")
            reg_file = _write(work / "registry.json", {"version": 1, "models": []})
            overlay = _write(
                work / "overlay.json",
                {"version": 1, "models": [{"id": "stub/moe", "description": "test overlay", "supported_sm": ["12.0"]}]},
            )
            add = [str(goethe), "registry", "add", "stub/moe", "--file", str(reg_file), "--overlay", str(overlay)]
            rr_add = run([*add, "--dry-run"], env=hf_env, timeout=60)
            if rr_add.returncode != 0:
                raise RuntimeError(rr_add.stderr or rr_add.stdout)
            if json.loads(reg_file.read_text(encoding="utf-8"))["models"]:
                raise RuntimeError("registry add --dry-run modified the registry file")
            bad_args = ["--reasoning-parser", "minimax_m3"]
            bad_overlay = _write(
                work / "bad_overlay.json",
                {"version": 1, "models": [{"id": "stub/moe", "extra_sglang_args": bad_args, "supported_sm": ["12.0"]}]},
            )
            reg_bad = _write(work / "reg_bad.json", {"version": 1, "models": []})
            rr_bad = run(
                [str(goethe), "registry", "add", "stub/moe", "--file", str(reg_bad), "--overlay", str(bad_overlay)],
                env=hf_env,
                timeout=60,
            )
            if rr_bad.returncode == 0 or json.loads(reg_bad.read_text(encoding="utf-8"))["models"]:
                raise RuntimeError("invalid reasoning-parser must refuse the add and leave the file untouched")
            tiny = _write(
                work / "tmp_reg_validate.json",
                {
                    "version": 1,
                    "models": [
                        {
                            "id": "stub/x", "architecture": "dense", "static_vram_gb": 1.0, "kv_elems_per_token": 512,
                            "max_ctx": 4096, "num_attention_heads": 8, "num_kv_heads": 2, "supported_sm": ["12.0"],
                            "extra_sglang_args": bad_args,
                        }
                    ],
                },
            )
            rr_val = run([str(goethe), "registry", "validate", f"file://{tiny}", "--check-sglang"], env=sg_env)
            if rr_val.returncode == 0:
                raise RuntimeError("validate should error on minimax_m3 parser")
        with temp_home(stub_python=True, tag="local") as home:
            model_src = work / "mymodel"
            model_src.mkdir()
            (model_src / "config.json").write_text(json.dumps(FIXTURE_CFG), encoding="utf-8")
            (model_src / "w.safetensors").write_bytes(b"x" * 128)
            env = {"GOETHE_HOME": str(home), "HF_HUB_OFFLINE": "1"}
            rr_inst = run(
                [
                    str(goethe), "install", str(model_src), "--name", "local/test-model",
                    "--sglang-arg", "--disable-shared-experts-fusion",
                ],
                env=env,
                timeout=60,
            )
            if rr_inst.returncode != 0:
                raise RuntimeError(rr_inst.stderr or rr_inst.stdout)
            link = home / "models" / "local__test-model"
            if not link.is_symlink() or not (model_src / "config.json").is_file():
                raise RuntimeError(f"expected an in-place symlink at {link} with the source intact")
            managed = sqlite_query(home / "state.db", "SELECT managed FROM installed_models")
            rr_rm = run([str(goethe), "remove", "local/test-model", "--yes"], env=env, timeout=30)
            if rr_rm.returncode != 0:
                raise RuntimeError(rr_rm.stderr or rr_rm.stdout)
            if not (model_src / "w.safetensors").is_file():
                raise RuntimeError("remove deleted files of a model registered in place")
            if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
                raise RuntimeError("remove did not unregister the model")
            if managed != [(0,)]:
                raise RuntimeError(f"in-place local install must be stored managed=0, got {managed}")
            if f"rm -rf '{model_src}'" not in rr_rm.stdout + rr_rm.stderr:
                raise RuntimeError(f"remove must print rm -rf '<path>' for the user: {rr_rm.stdout[-300:]}")
        print("stage3_registry ok")
    finally:
        srv.shutdown()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
