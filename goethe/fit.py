"""Fit context and lanes to the memory the engine actually has, by booting it.

NInfer has no dry-run for memory, so the only honest probe is a real boot. A boot that fails says exactly how
many bytes it wanted and how many it had, and a boot that starts says how many KV tokens it got. Both lines are
stable text in the server log; if a new NInfer version changes them, the probe raises instead of guessing.
"""
from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from goethe.resolve import Plan

# ponytail: string match on ninfer's own log lines. A version that rewords them fails loudly (ProbeError).
FATAL = re.compile(r"minimum Engine runtime reservation requires (\d+) bytes in addition to (\d+) bytes of "
                   r"automatic headroom, but only (\d+) bytes are available")
CAPACITY = re.compile(r"capacity \| KV ([\d,]+) tokens")
ATTEMPTS = 4
STEP = 16384  # the reservation grows steeply with context, so a fixed step converges faster than a ratio
PROBE_PORT = 18099  # the probe must not take the port the real server will bind


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class Fit:
    context: int
    lanes: int
    kv_tokens: int
    attempts: int


def fit(plan: Plan, argv: list[str], log: Path, *, boot: Callable[[list[str], Path], str] | None = None,
        seconds: float = 40) -> Fit:
    """Shrink the context until the engine starts. Lanes stay: fewer lanes reserve more KV, not less.

    Measured on a 5090: the reservation failure only appears past ~200k context, and the shortfall scales with it.
    """
    boot = boot or (lambda argv, log: _boot(argv, log, seconds))
    context, lanes = int(plan.value("context")), int(plan.value("lanes"))
    for attempt in range(1, ATTEMPTS + 1):
        text = boot(engine_argv(argv, context, lanes, port=PROBE_PORT), log)
        if (got := CAPACITY.search(text)):
            return Fit(context, lanes, int(got[1].replace(",", "")), attempt)
        if not _failed(text):
            raise ProbeError(f"boot {attempt} neither started nor gave the reservation error")
        context -= STEP
        if context < STEP:
            raise ProbeError(f"boot {attempt} does not fit even at the smallest context")
        time.sleep(2)  # GPU memory from the failed boot is freed within a second; don't boot on top of it
    raise ProbeError(f"still does not fit after {ATTEMPTS} boots")


def _failed(text: str) -> bool:
    """Reservation shortfall and CUDA out-of-memory are both a boot that did not start."""
    return FATAL.search(text) is not None or "cudaErrorMemoryAllocation" in text


def engine_argv(argv: list[str], context: int, lanes: int, port: int | None = None) -> list[str]:
    """The serve argv with context, lanes and, when given, the port replaced."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ("--max-context", "--max-concurrency", "--port"):
            skip = True
            continue
        out.append(arg)
    out += ["--max-context", str(context), "--max-concurrency", str(lanes)]
    return out + ["--port", str(port)] if port else out


def _boot(argv: list[str], log: Path, seconds: float = 40) -> str:
    """Start the engine, return its log once it listens or dies. Kills it either way."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # noqa: S603 argv is our own
    # reading the pipe, not a file: a block-buffered child never flushes to a file the probe can re-read
    text, poller = "", selectors.DefaultSelector()
    poller.register(proc.stdout, selectors.EVENT_READ)
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not poller.select(timeout=deadline - time.monotonic()):
                break
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            text += chunk.decode(errors="replace")
            if "listening on" in text or _failed(text):
                return text
        raise ProbeError(f"boot produced no result in {seconds:.0f}s")
    finally:
        poller.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text)


def cache_key(plan: Plan, model_file: Path) -> dict:
    gpu = plan.gpu
    size = model_file.stat().st_size if model_file.is_file() else 0
    return {"gpu": gpu.name, "vram_mib": gpu.total_mib, "driver": gpu.driver, "profile": plan.profile,
            "model": str(model_file), "model_bytes": size, "tuner": 1}


def load(path: Path, key: dict, free_mib: int) -> dict | None:
    """The cached fit, or None when the machine changed or free memory dropped below what the fit had."""
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("key") != key or free_mib < cached.get("free_mib", 0):
        return None
    if not all(k in cached for k in ("context", "lanes", "kv_tokens")):
        return None  # a hand-edited or older file is a cache miss, not a crash in serve
    return cached


def save(path: Path, key: dict, result: Fit, free_mib: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "context": result.context, "lanes": result.lanes,
                                "kv_tokens": result.kv_tokens, "free_mib": free_mib}))


def free_mib() -> int:
    code, out = __import__("goethe.hw", fromlist=["run"]).run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    try:
        return int(out.splitlines()[0])
    except (IndexError, ValueError):
        return 0  # nvidia-smi failed; treated as no free memory, so the cache is not trusted


def tune_path(env: os._Environ | dict | None = None) -> Path:
    from goethe import store
    return store.state_dir(env or os.environ) / "tune.json"
