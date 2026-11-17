"""The world around goethe, faked: GPUs, package host, platform, Hub mirror, packs, engines and a registry."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lib.goethe_test import make_release_key, sign_cdn_tree

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent.parent
FAKE = HERE / "fake"
BIN = Path(os.environ.get("E2E_BIN", ENGINE / "target" / "debug"))
GPUS = {
    "pro6000": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition|12.0|97887|600",
    "5090": "NVIDIA GeForce RTX 5090|12.0|32607|15",
    "gtx1080": "NVIDIA GeForce GTX 1080|6.1|8192|0",
    "none": "",
}
PACK_ARCH = "sm_120a"
NODE_TOKEN = "node-token-0123456789abcdef"
HF_TOKEN = "hf_e2e_good_token"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gguf(arch: str = "llama") -> bytes:
    """A tiny llama GGUF header: 2 layers, 8 heads of 64, 2 KV heads, context 4096 (what goethe reads, no tensors)."""

    def s(v: str) -> bytes:
        return len(v).to_bytes(8, "little") + v.encode()

    kvs = [("general.architecture", 8, s(arch))]
    for k, v in (
        ("block_count", 2),
        ("attention.head_count", 8),
        ("embedding_length", 512),
        ("context_length", 4096),
        ("attention.head_count_kv", 2),
    ):
        kvs.append((f"{arch}.{k}", 4, v.to_bytes(4, "little")))
    out = b"GGUF" + (3).to_bytes(4, "little") + (0).to_bytes(8, "little") + len(kvs).to_bytes(8, "little")
    for k, t, v in kvs:
        out += s(k) + t.to_bytes(4, "little") + v
    return out + b"\0" * 64


def safetensors() -> bytes:
    header = json.dumps({"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"\0" * 32


CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
}


def build_pack(weights: list[str], fmt: str = "safetensors", arch: str = PACK_ARCH) -> tuple[bytes, bytes]:
    """A GCPACK3 container: header (manifest range and block hashes), the manifest, then the blocks."""
    blocks_data = [bytes([i + 1]) * 4096 for i in range(3)]
    manifest = json.dumps(
        {
            "model": {"weights": [{"file": w} for w in weights]},
            "variants": {"v1": {"desc": {"quant": "bf16", "format": fmt}, "weights": list(range(len(weights)))}},
            "buckets": [{"sku": f"e2e_{arch}", "requires": {"arch": arch, "sms": 170}, "maxctx": 8192}],
        }
    ).encode()
    width = 640
    base = 16 + width + len(manifest)
    blocks = [
        {"off": base + i * 4096, "len": 4096, "sha256": sha(b), "codec": "none"} for i, b in enumerate(blocks_data)
    ]
    header = json.dumps({"manifest": [0, len(manifest)], "blocks": blocks}).encode().ljust(width)
    return b"GCPACK3\n" + width.to_bytes(8, "little") + header + manifest + b"".join(blocks_data), manifest


@dataclass
class Repo:
    files: dict[str, bytes]
    gated: bool = False
    slow: float = 0.0


@dataclass
class World:
    """One server on 127.0.0.1 for the platform, packs, package host and Hub, and the files engines need."""

    root: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="goethe-e2e-")))
    repos: dict[str, Repo] = field(default_factory=dict)
    packs: dict[str, bytes] = field(default_factory=dict)
    entries: list[dict] = field(default_factory=list)
    pkg_version: str = "9.9.9"
    polls: int = 0
    heartbeats: int = 0
    requests: list[str] = field(default_factory=list)

    def start(self) -> World:
        self.key, self.pub = make_release_key(self.root / "keys")
        self.add_repos()
        self.write_registry()
        self.build_pkg()
        world = self
        self.server = Server(("127.0.0.1", 0), type("H", (Handler,), {"world": world}))
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.root, ignore_errors=True)

    def add_repos(self) -> None:
        packed = {
            "model.safetensors": safetensors(),
            "config.json": json.dumps(CONFIG).encode(),
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
        }
        self.repos["e2e/packed"] = Repo(packed)
        self.repos["e2e/packed-only"] = Repo(dict(packed))
        self.repos["nvidia/Qwen3.8-27B-NVFP4"] = Repo(dict(packed))
        self.repos["e2e/plain-gguf"] = Repo({"plain-Q8_0.gguf": gguf()})
        self.repos["e2e/packed-gguf"] = Repo({"packed-Q8_0.gguf": gguf()})
        self.repos["e2e/gated"] = Repo({"plain-Q8_0.gguf": gguf()}, gated=True)
        self.repos["e2e/slow-gguf"] = Repo({"slow-Q8_0.gguf": gguf() + b"\0" * (6 << 20)}, slow=0.05)
        for repo, arch in (
            ("e2e/packed", PACK_ARCH),
            ("e2e/packed-only", PACK_ARCH),
            ("nvidia/Qwen3.8-27B-NVFP4", "sm_100a"),
            ("e2e/packed-gguf", PACK_ARCH),
        ):
            weights = ["packed-Q8_0.gguf"] if repo.endswith("gguf") else ["model.safetensors"]
            gg = repo.endswith("gguf")
            pack, manifest = build_pack(weights, "gguf" if gg else "safetensors", arch)
            name = f"{repo.split('/')[1]}-{arch}.gcpack"
            self.packs[name] = pack
            self.entries.append(
                {
                    "arch": arch,
                    "abi_major": 3,
                    "runtime": {"kind": "gc-c", "version": 1},
                    "model": {"hf_repo": repo, "revision": "main", "format": "gguf" if gg else "safetensors"},
                    "file": name,
                    "manifest_sha256": sha(manifest),
                    "size": len(pack),
                    "min_engine": "0.1.0",
                    "sig": {"alg": "test"},
                    "aux": {"repo": "e2e/packed", "files": ["config.json", "tokenizer.json", "tokenizer_config.json"]}
                    if gg
                    else None,
                    "e2e_files": weights,
                    "e2e_variant": "v1",
                }
            )

    def index(self) -> bytes:
        return json.dumps({"schema": "gc-pack-index", "version": 3, "serial": 7, "entries": self.entries}).encode()

    def write_registry(self) -> None:
        models = [
            {
                "id": "e2e/packed-only",
                "runtime": "sglang",
                "architecture": "dense",
                "pack_only": True,
                "default_ctx": 4096,
                "max_ctx": 8192,
            },
            {
                "id": "nvidia/Qwen3.8-27B-NVFP4",
                "runtime": "sglang",
                "architecture": "dense",
                "pack_only": True,
                "default_ctx": 4096,
                "max_ctx": 8192,
            },
        ]
        body = json.dumps({"version": 1, "models": models}, indent=1)
        (self.root / "registry.json").write_text(body)
        subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                str(self.key),
                "-out",
                str(self.root / "registry.json.sig"),
                str(self.root / "registry.json"),
            ],
            check=True,
        )

    def build_pkg(self) -> None:
        cdn = self.root / "pkg"
        arch = "x86_64"
        (cdn / f"bin/linux-{arch}").mkdir(parents=True)
        shutil.copy2(FAKE / "uv", cdn / f"bin/linux-{arch}/uv")
        (cdn / "pyenv").mkdir()
        for name in ("bootstrap.sh", "requirements.base.lock", "requirements.engine.lock", "requirements.lock"):
            shutil.copy2(ENGINE / "pyenv" / name, cdn / "pyenv" / name)
        for name in ("goethe", "goethed"):
            (cdn / f"bin/linux-{arch}/{name}").write_bytes(b"e2e-" + name.encode())
        (cdn / "wheels").mkdir()
        (cdn / "wheels/LATEST").write_text("sglang-e2e.whl\n")
        (cdn / "wheels/sglang-e2e.whl").write_bytes(b"wheel")
        sign_cdn_tree(cdn, self.key, self.pkg_version)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        return


class Handler(BaseHTTPRequestHandler):
    world: World
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        return

    def reply(
        self, code: int, body: bytes | dict = b"", kind: str = "application/json", extra: dict | None = None
    ) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def blob(self, data: bytes, slow: float = 0.0) -> None:
        start = 0
        if m := re.match(r"bytes=(\d+)-", self.headers.get("Range", "")):
            start = int(m.group(1))
        part = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(part)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.end_headers()
        try:
            for i in range(0, len(part), 1 << 16):
                self.wfile.write(part[i : i + (1 << 16)])
                self.wfile.flush()
                if slow:
                    time.sleep(slow)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        w = self.world
        path = self.path.split("?")[0]
        w.requests.append(f"GET {path}")
        if path.startswith("/pkg/"):
            f = w.root / path.lstrip("/")
            return self.reply(200, f.read_bytes(), "application/octet-stream") if f.is_file() else self.reply(404)
        if path == "/packs/v2/index.json":
            return self.reply(200, w.index())
        if path == "/packs/v2/index.sig":
            return self.reply(200, sha(w.index()).encode(), "text/plain")
        if m := re.fullmatch(r"/packs/v2/pack/(.+)", path):
            return (
                self.reply(302, b"", extra={"Location": f"{w.url}/blob/{m.group(1)}"})
                if m.group(1) in w.packs
                else self.reply(404)
            )
        if m := re.fullmatch(r"/blob/(.+)", path):
            return self.blob(w.packs[m.group(1)])
        if path.startswith("/api/organizations/"):
            return self.reply(200, {})
        if m := re.fullmatch(r"/api/models/(.+?)(?:/revision/[^/]+)?", path):
            return self.model_info(m.group(1))
        if m := re.fullmatch(r"/([^/]+/[^/]+)/resolve/([^/]+)/(.+)", path):
            return self.resolve(m.group(1), m.group(3))
        self.reply(404)

    def authed(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {HF_TOKEN}"

    def model_info(self, repo: str) -> None:
        r = self.world.repos.get(repo)
        if r is None:
            return self.reply(404, {"error": "Repository not found"})
        siblings = [
            {"rfilename": n, "size": len(d), "lfs": {"sha256": sha(d), "size": len(d)}}
            if n.endswith((".gguf", ".safetensors"))
            else {"rfilename": n, "size": len(d)}
            for n, d in r.files.items()
        ]
        self.reply(200, {"id": repo, "sha": "a" * 40, "gated": "auto" if r.gated else False, "siblings": siblings})

    def resolve(self, repo: str, name: str) -> None:
        r = self.world.repos.get(repo)
        if r is None or name not in r.files:
            return self.reply(404)
        if r.gated and not self.authed():
            return self.reply(401, {"error": "gated"})
        self.blob(r.files[name], r.slow)

    def do_POST(self) -> None:
        w = self.world
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        w.requests.append(f"POST {self.path}")
        if self.path == "/api/device/auth":
            if body.get("action") == "start":
                w.polls = 0
                return self.reply(
                    200,
                    {
                        "device_code": "device-code-0123456789",
                        "user_code": "ABCD-1234",
                        "verification_uri_complete": f"{w.url}/device?code=ABCD-1234",
                        "expires_in": 600,
                        "interval": 1,
                    },
                )
            w.polls += 1
            if w.polls < 2:
                return self.reply(200, {"status": "pending"})
            return self.reply(
                200,
                {
                    "status": "approved",
                    "node_id": "node-e2e",
                    "node_token": NODE_TOKEN,
                    "workspace": {"id": "ws-e2e", "name": "E2E"},
                },
            )
        if self.path == "/api/device/heartbeat":
            w.heartbeats += 1
            return self.reply(200, {"ok": True, "interval": 600})
        if self.path == "/api/device/logout":
            return self.reply(200, {})
        self.reply(404)


@dataclass
class Result:
    code: int
    out: str
    err: str
    seconds: float
    gap: float = 0.0

    @property
    def lines(self) -> int:
        return len(
            [
                line
                for line in (self.out + self.err).splitlines()
                if line.strip() and not re.match(r"^[^ ]+: [0-9.]+ (MB|GB)[ ,]", line)
            ]
        )


class Cell:
    """One home with goethed running in it, the world around it, and goethe calls that return Result."""

    def __init__(
        self,
        world: World,
        gpu: str = "5090",
        warm: bool = True,
        login: bool = False,
        extra: dict | None = None,
        up: bool = False,
    ):
        self.world, self.gpu = world, gpu
        self.home = Path(tempfile.mkdtemp(prefix="goethe-e2e-home-", dir=os.environ.get("E2E_TMP", "/tmp")))
        self.fakes = self.home.parent / f"{self.home.name}-fake"
        self.fakes.mkdir()
        shim = self.fakes / "shim"
        shim.mkdir()
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-o", str(shim / "libnvidia-ml.so.1"), str(HERE / "nvml_shim.c")], check=True
        )
        (shim / "libnvidia-ml.so").symlink_to("libnvidia-ml.so.1")
        shutil.copytree(FAKE, self.fakes / "fake")
        ports = []
        for _ in range(2):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                ports.append(s.getsockname()[1])
        for sub in ("models", "logs", "nginx", "bin"):
            (self.home / sub).mkdir()
        self.env = {
            "PATH": f"{self.fakes / 'fake'}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "GOETHE_HOME": str(self.home),
            "GOETHE_SOCK": str(self.home / "goethe.sock"),
            "GOETHE_MUX_PORT": str(ports[0]),
            "GOETHE_PUBLIC_PORT": str(ports[1]),
            "GOETHE_NO_AUTOSTART": "1",
            "GOETHE_PKG_URL": f"{world.url}/pkg",
            "GOETHE_PLATFORM_URL": world.url,
            "GOETHE_REGISTRY_URL": f"file://{world.root / 'registry.json'}",
            "GOETHE_RELEASE_PUB": str(world.pub),
            "HF_ENDPOINT": world.url,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "LLAMA_SERVER": str(self.fakes / "fake" / "llama-server"),
            "LD_LIBRARY_PATH": str(shim),
            "FAKE_NVML": GPUS[gpu],
            "E2E_FAKE_DIR": str(self.fakes / "fake"),
            "GOETHE_SYSTEMD_USER_UNIT_DIR": str(self.home / "systemd-user"),
            "GOETHE_SYSTEMD_UNIT_DIR": str(self.home / "systemd-system"),
            **(extra or {}),
        }
        if warm:
            self.prime_pyenv()
        self.daemon = None
        self.up = up
        if up:
            done = self.run("up", "--no-boot")
            if done.code != 0:
                raise RuntimeError(f"goethe up failed: {done.out} {done.err}")
        else:
            self.daemon = subprocess.Popen(
                [str(BIN / "goethed")],
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            end = time.time() + 20
            while not (self.home / "goethe.sock").exists():
                if time.time() > end or self.daemon.poll() is not None:
                    raise RuntimeError("goethed did not start")
                time.sleep(0.05)
        if login:
            done = self.run("login")
            if done.code != 0:
                raise RuntimeError(f"login failed: {done.out} {done.err}")

    def prime_pyenv(self) -> None:
        """A py-env as the installer leaves it: base and engine layers, laid down by the fake uv bootstrap.sh calls."""
        env = {
            **self.env,
            "GOETHE_ARCH": "x86_64",
            "UV": str(self.fakes / "fake" / "uv"),
            "GOETHE_ENGINE_WHEEL_URL": f"https://x.invalid/goethe_engine-{self.engine_version()}-py3-none-any.whl",
            "GOETHE_ENGINE_WHEEL_SHA256": "0" * 64,
        }
        for layer in ("base", "engine"):
            done = subprocess.run(
                ["bash", str(ENGINE / "pyenv" / "bootstrap.sh"), layer], env=env, capture_output=True, text=True
            )
            if done.returncode:
                raise RuntimeError(f"priming the {layer} layer failed: {done.stderr[-600:]}")

    @staticmethod
    def engine_version() -> str:
        text = (ENGINE / "crates/core/src/pyenv.rs").read_text()
        return (
            re.search(r"ENGINE_VERSION: &str = \"([0-9.]+)\"", text).group(1)
            if re.search(r"ENGINE_VERSION: &str = \"([0-9.]+)\"", text)
            else "0.3.0"
        )

    def run(
        self, *args: str, timeout: float = 120, env: dict | None = None, interrupt_after: float | None = None
    ) -> Result:
        """One goethe call: exit code, both streams, wall time and the longest silence between output chunks."""
        t0 = last = time.time()
        p = subprocess.Popen(
            [str(BIN / "goethe"), *args],
            env={**self.env, **(env or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        sel = selectors.DefaultSelector()
        chunks: dict[int, list[bytes]] = {p.stdout.fileno(): [], p.stderr.fileno(): []}
        for stream in (p.stdout, p.stderr):
            sel.register(stream, selectors.EVENT_READ)
        gap, interrupted = 0.0, False
        while sel.get_map():
            now = time.time()
            if interrupt_after is not None and not interrupted and now - t0 >= interrupt_after:
                os.killpg(p.pid, signal.SIGINT)
                interrupted = True
            if now - t0 > timeout:
                os.killpg(p.pid, signal.SIGKILL)
                break
            for key, _ in sel.select(timeout=0.1):
                data = os.read(key.fd, 65536)
                if not data:
                    sel.unregister(key.fileobj)
                    continue
                chunks[key.fd].append(data)
                gap, last = max(gap, time.time() - last), time.time()
        code = p.wait()
        p.stdout.close()
        p.stderr.close()
        end = time.time()
        text = [b"".join(c).decode(errors="replace") for c in chunks.values()]
        return Result(code, text[0], text[1], end - t0, max(gap, end - last))

    def logs(self, command: str) -> list[Path]:
        return sorted((self.home / "logs").glob(f"{command}-*.log"))

    def close(self) -> None:
        self.run("unload", "--all", timeout=30)
        if self.up:
            self.run("down", timeout=30)
        else:
            try:
                os.killpg(self.daemon.pid, signal.SIGTERM)
                self.daemon.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        subprocess.run(["pkill", "-f", str(self.home)], check=False)
        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.fakes, ignore_errors=True)
