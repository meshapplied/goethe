"""Replay the bundled agent traces against a running server and score what came back.

One stream. Temperature 1. The first two requests are reported separately and are not in the mean.
Decode tok/s is taken from the engine log when it is passed in. Wall tok/s includes prefill.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from pathlib import Path

BASH = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
DONE = re.compile(r"req#\d+ done.*decode ([\d.]+) tok/s.*accepted ([\d,]+)/([\d,]+)")
POWER = re.compile(r"([\d.]+)\s*W")


def _load(path: Path) -> dict:
    if path.name.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def turns(traj_dir: Path, stride: int = 6):
    jobs = []
    for path in sorted(traj_dir.glob("*.traj.json*")):
        raw = _load(path)
        msgs = [m for m in raw["messages"] if m["role"] != "exit"]
        wire = [_wire(m) for m in msgs]
        idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for i in idxs[stride - 1::stride]:
            n = 600
            try:
                n = msgs[i]["extra"]["response"]["usage"]["completion_tokens"]
            except (KeyError, TypeError):
                pass
            jobs.append((path.name, wire[:i], min(n + 16, 1500)))
    return jobs


def _wire(m: dict) -> dict:
    out = {k: m[k] for k in ("role", "content", "tool_calls", "tool_call_id") if m.get(k) is not None}
    if m["role"] == "assistant" and m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    return out


def tool_ok(message: dict) -> tuple[int, int]:
    """(valid, invalid) tool calls. A reply with no tool call is (0, 0)."""
    calls = message.get("tool_calls") or []
    valid = invalid = 0
    for call in calls:
        try:
            args = json.loads(call["function"]["arguments"])
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                valid += 1
            else:
                invalid += 1
        except (KeyError, TypeError, json.JSONDecodeError):
            invalid += 1
    return valid, invalid


def score(rows: list[dict], powers: list[float]) -> dict:
    kept = rows[2:]
    def mean(key):
        vals = [r[key] for r in kept if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    decode = [r["decode_tok_s"] for r in kept if r.get("decode_tok_s")]
    watts = [p for p in powers if p > 50]
    out = {
        "requests": len(rows),
        "scored": len(kept),
        "warmup_dropped": min(2, len(rows)),
        "mean_prompt_tokens": mean("prompt_tokens"),
        "decode_tok_s": round(sum(decode) / len(decode), 1) if decode else None,
        "wall_tok_s": round(sum(r["completion_tokens"] for r in kept) / sum(r["wall_s"] for r in kept), 1)
        if kept else None,
        "tool_calls_valid": sum(r["tool_valid"] for r in kept),
        "tool_calls_invalid": sum(r["tool_invalid"] for r in kept),
        "acceptance": _acceptance(kept),
        "mean_power_w": round(sum(watts) / len(watts), 1) if watts else None,
    }
    if out["decode_tok_s"] and out["mean_power_w"]:
        out["tok_s_per_watt"] = round(out["decode_tok_s"] / out["mean_power_w"], 3)
    return out


def _acceptance(rows: list[dict]) -> str | None:
    a = sum(r["acc_n"] for r in rows if r.get("acc_n"))
    b = sum(r["acc_d"] for r in rows if r.get("acc_d"))
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else None


def _post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def _powers(stop: threading.Event, out: list[float]) -> None:
    import subprocess
    while not stop.is_set():
        try:
            text = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2).stdout
            m = POWER.search(text)
            if m:
                out.append(float(m.group(1)))
        except (OSError, subprocess.TimeoutExpired):
            pass
        stop.wait(0.2)


def run(traj_dir: Path, port: int = 8080, log: Path | None = None) -> dict:
    jobs = turns(traj_dir)
    stop, powers = threading.Event(), []
    threading.Thread(target=_powers, args=(stop, powers), daemon=True).start()
    log_at = log.stat().st_size if log and log.is_file() else 0
    rows = []
    try:
        for name, messages, n in jobs:
            body = {
                "model": "qwen3.8-27b", "messages": messages, "tools": [BASH],
                "max_tokens": n, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            t0 = time.perf_counter()
            resp = _post(port, body)
            wall = time.perf_counter() - t0
            msg = resp["choices"][0]["message"]
            usage = resp.get("usage") or {}
            valid, invalid = tool_ok(msg)
            rows.append({
                "session": name, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens") or 0,
                "wall_s": wall, "tool_valid": valid, "tool_invalid": invalid,
            })
    finally:
        stop.set()
    if log and log.is_file():
        text = log.read_bytes()[log_at:].decode(errors="replace")
        found = DONE.findall(text)
        for i, row in enumerate(rows):
            if i < len(found):
                row["decode_tok_s"] = float(found[i][0])
                row["acc_n"] = int(found[i][1].replace(",", ""))
                row["acc_d"] = int(found[i][2].replace(",", ""))
    return score(rows, powers)
