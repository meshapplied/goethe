#!/usr/bin/env python3
"""Stage 3 goethe models, remove, and registry tooling (offline)."""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    main_exit,
    require_cli,
    resolve_python,
    run,
    run_goethe,
    run_temp_prefix,
    seed_installed_model,
    sqlite_query,
    temp_home,
)


class _StubHandler(BaseHTTPRequestHandler):
    model_json = {
        "gated": False,
        "siblings": [
            {
                "rfilename": "config.json",
                "size": 512,
            },
            {
                "rfilename": "model.safetensors",
                "size": 4096,
                "lfs": {"sha256": "ab" * 32},
            },
        ],
    }
    config_json = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
    }

    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/api/models/stub/tiny"):
            body = json.dumps(self.model_json).encode("utf-8")
        elif self.path.endswith("/config.json"):
            body = json.dumps(self.config_json).encode("utf-8")
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def _start_stub() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, f"http://127.0.0.1:{port}"


def main() -> None:
    goethe = require_cli("goethe")
    good = GOETHE_PKG / "cdn" / "v1" / "registry.json"
    bad_dir = Path(tempfile.mkdtemp(prefix=run_temp_prefix("reg"), dir="/tmp"))
    import shutil
    bad = bad_dir / "bad-registry.json"
    try:
        bad.write_text(
            json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": "dup",
                        "architecture": "dense",
                        "static_vram_gb": 1,
                        "kv_bytes_per_token": 1,
                        "min_total_vram_gb": 1,
                        "supported_sm": ["12.0"],
                        "default_ctx": 1,
                        "default_max_conc": 1,
                        "default_tp": 0,
                        "kv_cache_dtype": "fp8_e4m3",
                    },
                    {
                        "id": "dup",
                        "architecture": "dense",
                        "static_vram_gb": 1,
                        "kv_bytes_per_token": 1,
                        "min_total_vram_gb": 1,
                        "supported_sm": ["12.0"],
                        "default_ctx": 1,
                        "default_max_conc": 1,
                        "default_tp": 1,
                        "kv_cache_dtype": "fp8_e4m3",
                    },
                ],
            }
        ),
            encoding="utf-8",
        )
        rr_ok = run(
            [str(goethe), "registry", "validate", "--source", f"file://{good}"],
            env={"HF_HUB_OFFLINE": "1"},
            timeout=30,
        )
        if rr_ok.returncode != 0:
            raise RuntimeError(f"good registry failed: {rr_ok.stderr}")
        rr_bad = run(
            [str(goethe), "registry", "validate", "--source", str(bad)],
            env={"HF_HUB_OFFLINE": "1"},
            timeout=30,
        )
        if rr_bad.returncode == 0:
            raise RuntimeError("bad registry should exit 1")

        stub, endpoint = _start_stub()
        try:
            rr_entry = run(
                [
                    str(goethe),
                    "registry",
                    "entry",
                    "--repo",
                    "stub/tiny",
                    "--runtime",
                    "sglang",
                    "--revision",
                    "main",
                ],
                env={"HF_ENDPOINT": endpoint, "HF_HUB_OFFLINE": "0"},
                timeout=30,
            )
            if rr_entry.returncode != 0:
                raise RuntimeError(rr_entry.stderr or rr_entry.stdout)
            entry = json.loads(rr_entry.stdout)
            models = json.loads(rr_entry.stdout)["models"]
            entry = models[0]
            for key in (
                "id",
                "architecture",
                "static_vram_gb",
                "kv_elems_per_token",
                "max_ctx",
                "num_attention_heads",
                "num_kv_heads",
                "supported_sm",
            ):
                if key not in entry:
                    raise RuntimeError(f"registry entry missing {key}")
            if "extra_sglang_args" not in entry:
                entry["extra_sglang_args"] = []
        finally:
            stub.shutdown()

        with temp_home(stub_python=True, real_python=resolve_python(), tag="models") as home:
            reg = home / "reg.json"
            reg.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": [
                            {
                                "id": "local/test-a",
                                "architecture": "dense",
                                "static_vram_gb": 0.1,
                                "kv_bytes_per_token": 100,
                                "min_total_vram_gb": 1,
                                "supported_sm": ["12.0"],
                                "default_ctx": 512,
                                "default_max_conc": 1,
                                "default_tp": 1,
                                "kv_cache_dtype": "fp8_e4m3",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model_path = home / "models" / "local__test-a"
            model_path.mkdir(parents=True)
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "goethe_card.toml").write_text(
                'model_id = "local/test-a"\ncontext_length = 512\n',
                encoding="utf-8",
            )
            seed_installed_model(home, "local/test-a", "models/local__test-a", 0.1, 100, 512, 1, 1, managed=0)
            managed_path = home / "models" / "org__managed-b"
            managed_path.mkdir(parents=True)
            (managed_path / "config.json").write_text("{}", encoding="utf-8")
            seed_installed_model(home, "org/managed-b", "models/org__managed-b", 0.1, 100, 512, 1, 1, managed=1)
            env = {"GOETHE_HOME": str(home), "GOETHE_REGISTRY_URL": f"file://{reg}", "HF_HUB_OFFLINE": "1"}
            rows = json.loads(run_goethe([str(goethe), "models", "--json", "--installed"], env=env, timeout=30).stdout)
            row_a = next((r for r in rows if r.get("id") == "local/test-a"), None)
            if row_a is None or row_a.get("status") != "installed" or not row_a.get("custom"):
                raise RuntimeError("models --installed missing seeded custom model")
            rr_rm = run_goethe([str(goethe), "remove", "--model", "local/test-a", "--yes"], env=env, timeout=30)
            if not model_path.exists():
                raise RuntimeError("remove deleted the files of an unmanaged (in-place) model")
            rr_m = run_goethe([str(goethe), "remove", "--model", "org/managed-b", "--yes"], env=env, timeout=30)
            if managed_path.exists():
                raise RuntimeError(f"remove kept the files of a managed model: {rr_m.stdout}")
            if sqlite_query(home / "state.db", "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
                raise RuntimeError("remove did not clear installed_models")
            if f"rm -rf '{model_path}'" not in rr_rm.stdout + rr_rm.stderr:
                raise RuntimeError(f"unmanaged remove must print rm -rf '<path>': {rr_rm.stdout[-300:]}")
    finally:
        shutil.rmtree(bad_dir, ignore_errors=True)

    print("stage3_models ok")


if __name__ == "__main__":
    main_exit(main)
