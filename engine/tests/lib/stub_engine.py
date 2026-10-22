"""Stub stand-in for sglang.launch_server in goethe stage gates."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse launch_server style flags from argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--served-model-name")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


def _stub_mode(name: str) -> str:
    """Read per-model stub mode from env or stub_<name>.txt in model dir."""
    env_key = f"STUB_{name.upper()}_MODE"
    mode = os.environ.get(env_key, os.environ.get("STUB_HEALTH_MODE" if name == "health" else "STUB_CHEAP_MODE", ""))
    model_dir = _Handler.config.get("model_dir")
    if isinstance(model_dir, Path):
        path = model_dir / f"stub_{name}_mode.txt"
        if path.is_file():
            mode = path.read_text(encoding="utf-8").strip() or mode
        delay_path = model_dir / "stub_health_delay.txt"
        if delay_path.is_file() and name == "health":
            try:
                _Handler.config["health_delay_sec"] = float(delay_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pass
    return mode


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI compatible HTTP handler for the stub engine."""

    server_version = "GoetheStub/1.0"
    config: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve health and model list endpoints."""
        if self.path == "/health":
            mode = _stub_mode("health")
            started = float(self.config.get("started_at", time.time()))
            if mode in ("late503", "warmup"):
                delay = float(self.config.get("health_delay_sec", 70))
                if time.time() - started < delay:
                    self.send_response(503)
                    self.end_headers()
                    return
            if mode == "slow":
                time.sleep(1.0)
            if mode == "503":
                self.send_response(503)
                self.end_headers()
                return
            if mode == "burst503":
                if not getattr(self.server, "health_ready_at", None):
                    setattr(self.server, "health_ready_at", time.time())
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                ready_at = float(getattr(self.server, "health_ready_at", started))
                delay = float(self.config.get("health_burst_delay", 3))
                burst = float(self.config.get("health_burst_sec", 12))
                model_dir = self.config.get("model_dir")
                if isinstance(model_dir, Path):
                    dp = model_dir / "stub_health_burst_delay.txt"
                    bp = model_dir / "stub_health_burst_sec.txt"
                    if dp.is_file():
                        delay = float(dp.read_text(encoding="utf-8").strip())
                    if bp.is_file():
                        burst = float(bp.read_text(encoding="utf-8").strip())
                now = time.time()
                if now < ready_at + delay:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                if now < ready_at + delay + burst:
                    self.send_response(503)
                    self.end_headers()
                    return
            if mode == "hang":
                time.sleep(60.0)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path in ("/v1/models", "/get_model_info"):
            cheap = _stub_mode("cheap")
            if cheap == "503" or cheap == "503_after":
                if cheap == "503_after":
                    hits = getattr(self.server, "cheap_hits", 0) + 1
                    setattr(self.server, "cheap_hits", hits)
                    after = int(os.environ.get("STUB_CHEAP_503_AFTER", "2"))
                    if hits <= after:
                        name = self.config.get("served_model_name") or "stub-model"
                        self._json(200, {"data": [{"id": name, "object": "model"}]})
                        return
                self.send_response(503)
                self.end_headers()
                return
            if cheap == "hang":
                time.sleep(60.0)
                return
            if cheap == "hang_after":
                hits = getattr(self.server, "cheap_hits", 0) + 1
                setattr(self.server, "cheap_hits", hits)
                after = int(os.environ.get("STUB_CHEAP_HANG_AFTER", "2"))
                if hits > after:
                    time.sleep(60.0)
                    return
            name = self.config.get("served_model_name") or "stub-model"
            self._json(200, {"data": [{"id": name, "object": "model"}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        """Serve chat completions with optional SSE streaming."""
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        stream = bool(req.get("stream", False))
        tokens = int(req.get("max_tokens", 3) or 3)
        name = self.config.get("served_model_name") or "stub-model"
        content_prefix = f"model={name} "
        if not stream:
            text = content_prefix + " ".join(f"t{i}" for i in range(tokens))
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": tokens,
                    "total_tokens": 3 + tokens,
                },
            }
            self._json(200, payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for i in range(tokens):
            chunk = {
                "choices": [{"delta": {"content": content_prefix if i == 0 else f" t{i}"}}]
            }
            data = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()
            time.sleep(float(os.environ.get("STUB_TOKEN_INTERVAL", "0.05")))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _spawn_tp_children(tp_size: int) -> list[int]:
    """Spawn dummy TP rank processes in their own process groups."""
    if tp_size <= 1:
        return []
    pids: list[int] = []
    for _rank in range(1, tp_size):
        child = os.fork()
        if child == 0:
            os.setsid()
            time.sleep(3600)
            sys.exit(0)
        pids.append(child)
    return pids


def _shortfall_failures(model_dir: Path) -> int:
    """Return how many startup attempts should exit with a VRAM shortfall log line."""
    path = model_dir / "stub_shortfall_fail.txt"
    if not path.is_file():
        return 0
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip()))
    except ValueError:
        return 0


def main(argv: list[str] | None = None) -> int:
    """Run the stub HTTP engine until signaled."""
    args = _parse_args(argv or sys.argv[1:])
    delay = float(os.environ.get("STUB_START_DELAY", "0"))
    if delay > 0:
        time.sleep(delay)
    model_dir = Path(args.model_path) if args.model_path else Path.cwd()
    model_dir.mkdir(parents=True, exist_ok=True)
    fail_n = _shortfall_failures(model_dir)
    if fail_n > 0:
        seen_path = model_dir / "stub_shortfall_seen.txt"
        seen = 0
        if seen_path.is_file():
            try:
                seen = int(seen_path.read_text(encoding="utf-8").strip())
            except ValueError:
                seen = 0
        if seen < fail_n:
            seen_path.write_text(str(seen + 1), encoding="utf-8")
            gb = model_dir / "stub_shortfall_gb.txt"
            extra = 4.7
            if gb.is_file():
                try:
                    extra = float(gb.read_text(encoding="utf-8").strip())
                except ValueError:
                    extra = 4.7
            msg = (
                "Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
                f"Try increasing --mem-fraction-static. total_rest_memory=-{extra} GB"
            )
            print(msg, file=sys.stderr)
            sys.stderr.flush()
            return 1
    record = {
        "argv": sys.argv,
        "parsed": vars(args),
        "env": {
            **{k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "GOETHE_HOME", "PYTHONPATH") if k in os.environ},
            **{k: os.environ[k] for k in os.environ if k.startswith("TERNARY_")},
        },
    }
    (model_dir / "stub_launch_argv.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    served = args.served_model_name or model_dir.name
    _Handler.config = {
        "served_model_name": served,
        "port": args.port,
        "model_dir": model_dir,
        "started_at": time.time(),
        "health_delay_sec": float(os.environ.get("STUB_HEALTH_DELAY_SEC", "70")),
    }
    tp_pids = _spawn_tp_children(args.tp_size)
    ignore_term = os.environ.get("STUB_IGNORE_SIGTERM", "0") == "1"

    def handle_sigterm(_signum: int, _frame: object) -> None:
        if ignore_term:
            return
        delay = float(os.environ.get("STUB_SIGTERM_DELAY", "0"))
        if delay > 0:
            time.sleep(delay)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            time.sleep(0.2)
    except SystemExit:
        pass
    finally:
        httpd.shutdown()
        for pid in tp_pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
