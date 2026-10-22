#!/usr/bin/env python3
"""Stage 3 registry entry/add/validate offline tests."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from lib.mesh_test import MESH_PKG, main_exit, require_cli, run, run_temp_prefix, temp_home

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

    def do_GET(self) -> None:
        if "/api/models/" in self.path and "blobs=true" in self.path:
            if "stub/missing" in self.path:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(
                {
                    "gated": False,
                    "sha": "abc123",
                    "siblings": [{"rfilename": "model.safetensors", "size": 1000}],
                }
            ).encode()
        elif self.path.endswith("config.json"):
            body = json.dumps(FIXTURE_CFG).encode()
        elif "/api/organizations/" in self.path:
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    mesh = require_cli("mesh")
    work = Path(tempfile.mkdtemp(prefix=run_temp_prefix("registry"), dir="/tmp"))
    try:
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        port = srv.server_address[1]
        import threading

        threading.Thread(target=srv.serve_forever, daemon=True).start()
        endpoint = f"http://127.0.0.1:{port}"
        reg_path = MESH_PKG / "cdn" / "v1" / "registry.json"
        rr = run(
            [str(mesh), "registry", "validate", f"file://{reg_path}", "--check-sglang"],
            env={"HF_HUB_OFFLINE": "1"},
        )
        if rr.returncode != 0:
            raise RuntimeError(rr.stderr or rr.stdout)
        ids_file = work / "ids.txt"
        ids_file.write_text("stub/moe\nstub/missing\n", encoding="utf-8")
        rr_strict = run(
            [str(mesh), "registry", "entry", "--ids-file", str(ids_file), "--strict"],
            env={"HF_ENDPOINT": endpoint, "HF_HUB_OFFLINE": "0"},
            timeout=60,
        )
        if rr_strict.returncode == 0:
            raise RuntimeError("strict bulk should fail when any id fails")
        rr_bulk = run(
            [str(mesh), "registry", "entry", "--ids-file", str(ids_file)],
            env={"HF_ENDPOINT": endpoint, "HF_HUB_OFFLINE": "0"},
            timeout=60,
        )
        if rr_bulk.returncode != 0:
            raise RuntimeError(rr_bulk.stderr or rr.stdout)
        out = json.loads(rr_bulk.stdout)
        if len(out.get("models", [])) < 1:
            raise RuntimeError("bulk entry produced no models")
        reg_file = work / "registry.json"
        reg_file.write_text(json.dumps({"version": 1, "models": []}), encoding="utf-8")
        overlay = work / "overlay.json"
        overlay.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": "stub/moe",
                            "description": "test overlay",
                            "supported_sm": ["12.0"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rr_add = run(
            [
                str(mesh),
                "registry",
                "add",
                "stub/moe",
                "--file",
                str(reg_file),
                "--overlay",
                str(overlay),
                "--dry-run",
            ],
            env={"HF_ENDPOINT": endpoint},
            timeout=60,
        )
        if rr_add.returncode != 0:
            raise RuntimeError(rr_add.stderr or rr_add.stdout)
        bad_overlay = work / "bad_overlay.json"
        bad_overlay.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": "stub/moe",
                            "extra_sglang_args": ["--reasoning-parser", "minimax_m3"],
                            "supported_sm": ["12.0"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        reg_bad = work / "reg_bad.json"
        reg_bad.write_text(json.dumps({"version": 1, "models": []}), encoding="utf-8")
        mesh_home = os.environ.get("MESH_HOME", "/scratch/ternary-flash/mesh-home")
        py_env = Path(mesh_home) / "py-env" / "bin" / "python"
        if py_env.is_file():
            rr_bad = run(
                [
                    str(mesh),
                    "registry",
                    "add",
                    "stub/moe",
                    "--file",
                    str(reg_bad),
                    "--overlay",
                    str(bad_overlay),
                ],
                env={"HF_ENDPOINT": endpoint, "MESH_HOME": mesh_home},
                timeout=60,
            )
            if rr_bad.returncode == 0:
                raise RuntimeError("invalid reasoning-parser should refuse add")
            tiny = work / "tmp_reg_validate.json"
            tiny.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": [
                            {
                                "id": "stub/x",
                                "architecture": "dense",
                                "static_vram_gb": 1.0,
                                "kv_elems_per_token": 512,
                                "max_ctx": 4096,
                                "num_attention_heads": 8,
                                "num_kv_heads": 2,
                                "supported_sm": ["12.0"],
                                "extra_sglang_args": ["--reasoning-parser", "minimax_m3"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rr_val = run(
                [str(mesh), "registry", "validate", f"file://{tiny}", "--check-sglang"],
                env={"MESH_HOME": mesh_home},
            )
            if rr_val.returncode == 0:
                raise RuntimeError("validate should error on minimax_m3 parser")
        with temp_home(stub_python=True, tag="local") as home:
            model_src = work / "mymodel"
            model_src.mkdir()
            (model_src / "config.json").write_text(json.dumps(FIXTURE_CFG), encoding="utf-8")
            (model_src / "w.safetensors").write_bytes(b"x" * 128)
            rr_inst = run(
                [
                    str(mesh),
                    "install",
                    str(model_src),
                    "--name",
                    "local/test-model",
                    "--sglang-arg",
                    "--disable-shared-experts-fusion",
                ],
                env={"MESH_HOME": str(home), "HF_HUB_OFFLINE": "1"},
                timeout=60,
            )
            if rr_inst.returncode != 0:
                raise RuntimeError(rr_inst.stderr or rr_inst.stdout)
            link = home / "models" / "local__test-model"
            if not link.is_symlink():
                raise RuntimeError(f"expected symlink at {link}")
            if not (Path(model_src) / "config.json").is_file():
                raise RuntimeError("local install broke source dir")
            rr_rm = run(
                [str(mesh), "remove", "local/test-model", "--yes"],
                env={"MESH_HOME": str(home)},
                timeout=30,
            )
            if rr_rm.returncode != 0:
                raise RuntimeError(rr_rm.stderr or rr_rm.stdout)
            if not model_src.is_dir():
                raise RuntimeError("remove deleted user model dir")
        srv.shutdown()
        print("stage3_registry ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main_exit(main)
