"""Fit context and lanes to the memory the engine actually has, by booting it.

NInfer has no dry-run for memory, so the only honest probe is a real boot. A boot that fails says exactly how
many bytes it wanted and how many it had, and a boot that starts says how many KV tokens it got. Both lines are
stable text in the server log; if a new NInfer version changes them, the probe raises instead of guessing.
"""
from __future__ import annotations

import json
import os
import re
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
ALIGN = 256
PROBE_PORT = 18099  # the probe must not take the port the real server will bind


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class Fit:
    context: int
    lanes: int
    kv_tokens: int
    attempts: int


def _context(plan: Plan, value: int) -> Plan:
    plan.settings["context"].value = value
    return plan


def _lanes(plan: Plan, value: int) -> Plan:
    plan.settings["lanes"].value = value
    return plan


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
        if not (fail := FATAL.search(text)):
            raise ProbeError(f"boot {attempt} neither started nor gave the reservation error")
        need, headroom, have = (int(x) for x in fail.groups())
        smaller = max(ALIGN, int(context * have / (need + headroom)) // ALIGN * ALIGN)
        if smaller >= context:
            raise ProbeError(f"boot {attempt} is short {-have + need + headroom} bytes and the context cannot shrink")
        context = smaller
    raise ProbeError(f"still does not fit after {ATTEMPTS} boots")


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
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    with log.open("a") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)  # noqa: S603 argv is our own
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            text = log.read_text(errors="replace")
            if "listening on" in text or FATAL.search(text) or proc.poll() is not None:
                return text
            time.sleep(0.5)
        raise ProbeError(f"boot produced no result in {seconds:.0f}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


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
    return cached


def save(path: Path, key: dict, result: Fit, free_mib: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "context": result.context, "lanes": result.lanes,
                                "kv_tokens": result.kv_tokens, "free_mib": free_mib}))


def free_mib() -> int:
    code, out = __import__("goethe.hw", fromlist=["run"]).run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return int(out.splitlines()[0]) if code == 0 and out.strip() else 0


def tune_path(env: os._Environ | dict | None = None) -> Path:
    from goethe import store
    return store.state_dir(env or os.environ) / "tune.json"
