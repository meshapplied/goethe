#!/usr/bin/env python3
"""Stage 3 install against a local fake Hugging Face Hub: per-file resume after an interrupt, corruption repair."""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from lib.goethe_test import main_exit, require_cli, run_goethe, sha256_file, sqlite_query, temp_home, wait_until

REPO = "stub/resume"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
BIG = "model.safetensors"
RATE = 2 << 20


def _safetensors(n_bytes: int) -> bytes:
    """A valid safetensors file holding one F16 tensor of random data."""
    header = json.dumps({"w": {"dtype": "F16", "shape": [n_bytes // 2], "data_offsets": [0, n_bytes]}}).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + random.Random(7).randbytes(n_bytes)


FILES = {
    "config.json": json.dumps(
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            "vocab_size": 128,
            "torch_dtype": "float16",
        }
    ).encode(),
    BIG: _safetensors(8 << 20),
}
SHA256 = {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}
BLOB = {name: hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() for name, data in FILES.items()}


class _Hub(BaseHTTPRequestHandler):
    """Enough of the Hub API and resolve protocol for goethe preflight and hf download (plain HTTP, no xet)."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []
    served: dict[str, int] = {}
    lock = threading.Lock()

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, payload: object) -> None:
        """Send a JSON 200 response with Content-Length."""
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _siblings(self) -> list[dict[str, Any]]:
        """Sibling entries with LFS metadata for the big file."""
        out = []
        for name, data in FILES.items():
            entry: dict[str, Any] = {"rfilename": name, "size": len(data), "blobId": BLOB[name]}
            if name == BIG:
                entry["lfs"] = {"sha256": SHA256[name], "size": len(data), "pointerSize": 134}
            out.append(entry)
        return out

    def do_GET(self) -> None:
        """Model info, tree listing, organization probe and file downloads."""
        path = urlsplit(self.path).path
        with self.lock:
            self.requests.append({"method": "GET", "path": path, "range": self.headers.get("Range")})
        if path.startswith(f"/api/models/{REPO}/tree/"):
            tree = []
            for s in self._siblings():
                item: dict[str, Any] = {"type": "file", "oid": s["blobId"], "size": s["size"], "path": s["rfilename"]}
                if "lfs" in s:
                    item["lfs"] = {"oid": s["lfs"]["sha256"], "size": s["size"], "pointerSize": 134}
                tree.append(item)
            self._send_json(tree)
        elif path.startswith(f"/api/models/{REPO}"):
            self._send_json(
                {"id": REPO, "modelId": REPO, "sha": COMMIT, "private": False, "gated": False, "disabled": False,
                 "tags": [], "siblings": self._siblings()}
            )
        elif path.startswith("/api/organizations/"):
            self._send_json({})
        elif path.startswith(f"/{REPO}/resolve/"):
            self._resolve(path.rsplit("/", 1)[-1], head=False)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_HEAD(self) -> None:
        """File metadata for hf_hub_download."""
        path = urlsplit(self.path).path
        with self.lock:
            self.requests.append({"method": "HEAD", "path": path, "range": None})
        self._resolve(path.rsplit("/", 1)[-1], head=True)

    def _resolve(self, name: str, head: bool) -> None:
        """Serve a file with Range support, throttling the big one so it can be interrupted."""
        data = FILES.get(name)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes=") and not head:
            start = int(rng[6:].split("-", 1)[0] or 0)
        self.send_response(206 if start else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data) - start))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Repo-Commit", COMMIT)
        self.send_header("ETag", f'"{BLOB[name]}"')
        if name == BIG:
            self.send_header("X-Linked-Etag", f'"{SHA256[name]}"')
            self.send_header("X-Linked-Size", str(len(data)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.end_headers()
        if head:
            return
        step = 64 << 10
        try:
            for off in range(start, len(data), step):
                self.wfile.write(data[off : off + step])
                with self.lock:
                    self.served[name] = self.served.get(name, 0) + min(step, len(data) - off)
                if name == BIG:
                    time.sleep(step / RATE)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def _gets(name: str) -> list[dict[str, Any]]:
    return [r for r in _Hub.requests if r["method"] == "GET" and r["path"].endswith(f"/resolve/{COMMIT}/{name}")]


def main() -> None:
    """Interrupt an install mid-download, resume it, then corrupt the weights and require a repair."""
    goethe = require_cli("goethe")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with temp_home(stub_python=True, tag="hf-local") as home:
            reg = home / "registry.json"
            entry = {
                "id": REPO, "architecture": "dense", "static_vram_gb": 0.01, "kv_bytes_per_token": 512,
                "min_total_vram_gb": 1.0, "supported_sm": ["12.0"], "default_ctx": 128, "default_max_conc": 1,
                "default_tp": 1, "kv_cache_dtype": "fp8_e4m3", "extra_sglang_args": [],
            }
            reg.write_text(json.dumps({"version": 1, "models": [entry]}), encoding="utf-8")
            env = {
                "GOETHE_HOME": str(home),
                "GOETHE_REGISTRY_URL": f"file://{reg}",
                "HF_ENDPOINT": f"http://127.0.0.1:{srv.server_address[1]}",
                "HF_HOME": str(home / "hf-home"),
                "HF_HUB_DISABLE_XET": "1",
            }
            cmd = [str(goethe), "install", REPO, "--no-load"]
            proc = subprocess.Popen(
                cmd, env={**os.environ, **env}, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            try:
                if not wait_until(lambda: _Hub.served.get(BIG, 0) >= 1 << 20, timeout=60, interval=0.05):
                    raise RuntimeError(f"download never started: {_Hub.requests[-5:]}")
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            partial = _Hub.served[BIG]
            if partial >= len(FILES[BIG]) or len(_gets("config.json")) != 1:
                raise RuntimeError("interrupt did not land mid-way through the weights after config.json completed")
            db = home / "state.db"
            if db.is_file() and sqlite_query(db, "SELECT COUNT(*) FROM installed_models")[0][0] != 0:
                raise RuntimeError("interrupted install registered a half-downloaded model")
            big_before = len(_gets(BIG))
            run_goethe(cmd, env=env, timeout=300)
            if len(_gets("config.json")) != 1:
                raise RuntimeError("resumed install downloaded the already complete config.json again")
            if len(_gets(BIG)) != big_before + 1:
                raise RuntimeError(f"resumed install fetched the interrupted file {len(_gets(BIG)) - big_before} times")
            weights = next((home / "models").glob("*__*")) / BIG
            if sha256_file(weights) != SHA256[BIG]:
                raise RuntimeError("resumed download does not match the published sha256")
            big_before = len(_gets(BIG))
            run_goethe(cmd, env=env, timeout=120)
            if len(_gets(BIG)) != big_before:
                raise RuntimeError("re-install of a verified model downloaded the weights again")
            data = bytearray(weights.read_bytes())
            data[len(data) // 2] ^= 0xFF
            weights.write_bytes(data)
            run_goethe(cmd, env=env, timeout=300)
            if sha256_file(weights) != SHA256[BIG]:
                raise RuntimeError("re-install left a corrupted (same-size) weight file in place")
        print(f"stage3_hf_local ok interrupted_at={partial}")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main_exit(main)
