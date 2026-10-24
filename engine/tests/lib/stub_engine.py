"""Stub stand-in for sglang.launch_server in goethe stage gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RECORDED_ENV = (
    "CUDA_VISIBLE_DEVICES", "GOETHE_HOME", "PYTHONPATH", "GOETHE_INSTANCE_ID", "GOETHE_KERNEL_DIR", "SPT_NOENV",
)
FILE_KNOBS = ("STUB_START_DELAY", "STUB_IGNORE_SIGTERM", "STUB_SIGTERM_DELAY", "STUB_TOKEN_INTERVAL")
AUTH_EXEMPT = ("/health", "/ready", "/metrics")
_record_lock = threading.Lock()


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
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--served-model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line; the model dir may already be gone during teardown."""
    with _record_lock:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass


def _read(path: Path) -> str:
    """Read a stub control file, empty when missing."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _float_file(name: str, default: float) -> float:
    """Read a float control file from the model dir, falling back to default."""
    raw = _read(_Handler.config["model_dir"] / name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _knobs(model_dir: Path) -> dict[str, str]:
    """Active STUB_* knobs: engine env (the card [env] table) overridden by model-dir stub_<knob>.txt files."""
    knobs = {k: v for k, v in os.environ.items() if k.startswith("STUB_")}
    for name in FILE_KNOBS:
        raw = _read(model_dir / f"{name.lower()}.txt")
        if raw:
            knobs[name] = raw
    return knobs


def _stub_mode(name: str) -> str:
    """Per-model stub mode from stub_<name>_mode.txt, else STUB_<NAME>_MODE env."""
    env_key = f"STUB_{name.upper()}_MODE"
    mode = os.environ.get(env_key, os.environ.get("STUB_HEALTH_MODE" if name == "health" else "STUB_CHEAP_MODE", ""))
    return _read(_Handler.config["model_dir"] / f"stub_{name}_mode.txt") or mode


def _authorized(header: str | None, key: str) -> bool:
    """SGLang bearer check: scheme case-insensitive, constant-time token compare."""
    if not header:
        return False
    parts = header.split(" ", 1)
    return len(parts) == 2 and parts[0].lower() == "bearer" and secrets.compare_digest(parts[1], key)


class _Handler(BaseHTTPRequestHandler):
    """OpenAI compatible HTTP/1.1 handler with SGLang api-key semantics."""

    protocol_version = "HTTP/1.1"
    server_version = "GoetheStub/2.0"
    config: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _record(self, status: int) -> None:
        """Log method, path, status and headers for gateway assertions."""
        row = {
            "t": time.time(),
            "method": self.command,
            "path": self.path,
            "status": status,
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }
        _append_jsonl(self.config["model_dir"] / "stub_requests.jsonl", row)

    def _send(self, code: int, body: bytes = b"", ctype: str = "application/json") -> None:
        """Send a complete response with Content-Length so keep-alive stays usable."""
        self._record(code)
        self.send_response(code)
        if body:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"))

    def _hang(self) -> None:
        """Hold the request for 60 s, then drop the connection without a response."""
        self._record(0)
        time.sleep(60.0)
        self.close_connection = True

    def _auth_ok(self) -> bool:
        """Enforce --api-key like SGLang: health, ready and metrics paths stay open."""
        key = self.config.get("api_key")
        if not key or self.command == "OPTIONS" or self.path.startswith(AUTH_EXEMPT):
            return True
        if _authorized(self.headers.get("Authorization"), key):
            return True
        self._json(401, {"error": "Unauthorized"})
        return False

    def _read_body(self) -> bytes:
        """Read a Content-Length or chunked request body."""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))

    def _health(self) -> None:
        """Serve /health with the configured failure modes."""
        mode = _stub_mode("health")
        started = float(self.config["started_at"])
        if mode in ("late503", "warmup"):
            delay = _float_file("stub_health_delay.txt", float(self.config["health_delay_sec"]))
            if time.time() - started < delay:
                self._send(503)
                return
        if mode == "slow":
            time.sleep(1.0)
        if mode == "503":
            self._send(503)
            return
        if mode == "hang":
            self._hang()
            return
        if mode == "burst503":
            ready_at = self.config.setdefault("health_ready_at", time.time())
            delay = _float_file("stub_health_burst_delay.txt", 3.0)
            burst = _float_file("stub_health_burst_sec.txt", 12.0)
            now = time.time()
            if ready_at + delay <= now < ready_at + delay + burst:
                self._send(503)
                return
        self._send(200)

    def _cheap_fails(self) -> bool:
        """Apply cheap-probe modes to /v1/models and the model info routes; True when handled."""
        cheap = _stub_mode("cheap")
        if cheap in ("503", "503_after", "hang_after"):
            hits = self.config["cheap_hits"] = self.config.get("cheap_hits", 0) + 1
            if cheap == "503":
                self._send(503)
                return True
            if cheap == "503_after" and hits > int(os.environ.get("STUB_CHEAP_503_AFTER", "2")):
                self._send(503)
                return True
            if cheap == "hang_after" and hits > int(os.environ.get("STUB_CHEAP_HANG_AFTER", "2")):
                self._hang()
                return True
        if cheap == "hang":
            self._hang()
            return True
        return False

    def _model_info(self) -> dict[str, Any]:
        """Shape of SGLang /model_info for a generation model."""
        name = self.config["served_model_name"]
        return {
            "model_path": str(self.config["model_dir"]),
            "served_model_name": name,
            "tokenizer_path": str(self.config["model_dir"]),
            "is_generation": True,
            "preferred_sampling_params": None,
            "weight_version": "default",
            "load_format": "auto",
            "reasoning_parser": None,
            "tool_call_parser": None,
            "disaggregation_mode": "null",
            "has_image_understanding": False,
            "has_audio_understanding": False,
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }

    def do_GET(self) -> None:
        """Serve health, model list and model info endpoints."""
        if not self._auth_ok():
            return
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._health()
            return
        if path in ("/v1/models", "/get_model_info", "/model_info"):
            if self._cheap_fails():
                return
            if path == "/v1/models":
                name = self.config["served_model_name"]
                card = {
                    "id": name,
                    "object": "model",
                    "created": int(self.config["started_at"]),
                    "owned_by": "sglang",
                    "root": name,
                    "max_model_len": self.config["context_length"],
                }
                self._json(200, {"object": "list", "data": [card]})
            else:
                self._json(200, self._model_info())
            return
        self._json(404, {"detail": "Not Found"})

    def do_POST(self) -> None:
        """Serve chat completions, completions and embeddings."""
        if not self._auth_ok():
            return
        raw = self._read_body()
        try:
            req = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"object": "error", "message": "invalid JSON body", "code": 400})
            return
        path = self.path.split("?", 1)[0]
        if path == "/v1/embeddings":
            self._embeddings(req)
        elif path in ("/v1/chat/completions", "/v1/completions"):
            self._generate(req, chat=path == "/v1/chat/completions")
        elif path == "/v1/messages":
            self._message(req)
        else:
            self._json(404, {"detail": "Not Found"})

    def _embeddings(self, req: dict[str, Any]) -> None:
        """Deterministic 8-dim embeddings for a string or list input."""
        items = req.get("input", "")
        items = items if isinstance(items, list) else [items]
        data = []
        for i, item in enumerate(items):
            digest = hashlib.sha256(json.dumps(item).encode("utf-8")).digest()
            data.append({"object": "embedding", "index": i, "embedding": [b / 255.0 for b in digest[:8]]})
        usage = {"prompt_tokens": len(items), "total_tokens": len(items)}
        self._json(200, {"object": "list", "data": data, "model": self.config["served_model_name"], "usage": usage})

    def _message(self, req: dict[str, Any]) -> None:
        """Non-streaming Anthropic Messages reply naming the served model."""
        tokens = int(req.get("max_tokens") or 3)
        name = self.config["served_model_name"]
        self._json(200, {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": name,
            "content": [{"type": "text", "text": f"model={name}"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": tokens},
        })

    def _chunk(self, payload: dict[str, Any] | str) -> None:
        """Write one SSE event as an HTTP/1.1 chunk."""
        text = payload if isinstance(payload, str) else json.dumps(payload)
        data = f"data: {text}\n\n".encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _generate(self, req: dict[str, Any], chat: bool) -> None:
        """Non-streaming or chunked SSE generation of max_tokens pieces."""
        tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 3)
        name = self.config["served_model_name"]
        pieces = [f"model={name} " if i == 0 else f" t{i}" for i in range(tokens)]
        usage = {"prompt_tokens": 3, "completion_tokens": tokens, "total_tokens": 3 + tokens}
        base = {
            "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": name,
        }
        if not req.get("stream"):
            text = "".join(pieces)
            choice = {"index": 0, "finish_reason": "length"}
            choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
            self._json(200, {**base, "choices": [choice], "usage": usage})
            return
        self._record(200)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        base["object"] = "chat.completion.chunk" if chat else "text_completion"
        interval = float(self.config["knobs"].get("STUB_TOKEN_INTERVAL", "0.05"))
        try:
            for i, piece in enumerate(pieces):
                delta: dict[str, Any] = {"content": piece} if i else {"role": "assistant", "content": piece}
                choice = {"index": 0, "delta": delta} if chat else {"index": 0, "text": piece}
                self._chunk({**base, "choices": [{**choice, "finish_reason": None}]})
                time.sleep(interval)
            final = {"index": 0, "delta": {}} if chat else {"index": 0, "text": ""}
            self._chunk({**base, "choices": [{**final, "finish_reason": "length"}]})
            if (req.get("stream_options") or {}).get("include_usage"):
                self._chunk({**base, "choices": [], "usage": usage})
            self._chunk("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def _spawn_tp_ranks(tp_size: int, model_dir: Path, ignore_term: bool) -> list[int]:
    """Fork TP rank stand-ins that stay in the leader's process group and outlive a killed leader."""
    pids: list[int] = []
    for _rank in range(1, tp_size):
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN if ignore_term else signal.SIG_DFL)
            while model_dir.is_dir():
                time.sleep(0.5)
            os._exit(0)
        pids.append(child)
    return pids


def _shortfall_attempt(model_dir: Path) -> bool:
    """Count startup attempts and report whether this one must fail with a VRAM shortfall."""
    try:
        fail_n = max(0, int(_read(model_dir / "stub_shortfall_fail.txt") or 0))
        seen = int(_read(model_dir / "stub_shortfall_seen.txt") or 0)
    except ValueError:
        return False
    if seen >= fail_n:
        return False
    (model_dir / "stub_shortfall_seen.txt").write_text(str(seen + 1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """Run the stub HTTP engine until signaled or until its model dir disappears."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    model_dir = Path(args.model_path) if args.model_path else Path.cwd()
    if not model_dir.is_dir():
        print(f"stub_engine: model path {model_dir} does not exist", file=sys.stderr)
        return 1
    if _shortfall_attempt(model_dir):
        try:
            extra = float(_read(model_dir / "stub_shortfall_gb.txt") or 4.7)
        except ValueError:
            extra = 4.7
        print(
            "Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
            f"Try increasing --mem-fraction-static. total_rest_memory=-{extra} GB",
            file=sys.stderr,
            flush=True,
        )
        return 1
    knobs = _knobs(model_dir)
    ignore_term = knobs.get("STUB_IGNORE_SIGTERM", "0") == "1"
    record: dict[str, Any] = {
        "argv": sys.argv,
        "parsed": vars(args),
        "env": {
            **{k: os.environ[k] for k in RECORDED_ENV if k in os.environ},
            **{k: v for k, v in os.environ.items() if k.startswith("TERNARY_")},
        },
        "env_keys": sorted(os.environ),
        "knobs": knobs,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "started_at": time.time(),
    }
    launch = model_dir / "stub_launch_argv.json"
    launch.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def handle_sigterm(_signum: int, _frame: object) -> None:
        _append_jsonl(model_dir / "stub_signals.jsonl", {"sig": "TERM", "t": time.time(), "ignored": ignore_term})
        if ignore_term:
            return
        time.sleep(float(knobs.get("STUB_SIGTERM_DELAY", "0")))
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    tp_pids: list[int] = []
    httpd: ThreadingHTTPServer | None = None
    try:
        time.sleep(float(knobs.get("STUB_START_DELAY", "0")))
        tp_pids = _spawn_tp_ranks(args.tp_size, model_dir, ignore_term)
        _Handler.config = {
            "served_model_name": args.served_model_name or model_dir.name,
            "context_length": args.context_length,
            "model_dir": model_dir,
            "api_key": args.api_key,
            "knobs": knobs,
            "started_at": time.time(),
            "health_delay_sec": float(os.environ.get("STUB_HEALTH_DELAY_SEC", "70")),
        }
        httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
        record.update({"tp_pids": tp_pids, "listening_at": time.time()})
        launch.write_text(json.dumps(record, indent=2), encoding="utf-8")
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        while thread.is_alive() and model_dir.is_dir():
            time.sleep(0.2)
    except SystemExit:
        pass
    finally:
        if httpd is not None:
            httpd.shutdown()
        for pid in tp_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
