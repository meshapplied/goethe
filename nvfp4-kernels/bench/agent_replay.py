#!/usr/bin/env python3
"""Replay recorded agent turns against a running ninfer-serve at a fixed number of workers."""
import argparse
import glob
import gzip
import json
import threading
import time

import httpx

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("traj_dir")
ap.add_argument("workers", type=int)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--cap", type=int, default=1500)
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--think", type=int, default=1)
ap.add_argument("--model", default="qwen3.8-27b")
args = ap.parse_args()


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["messages"]


def wire(m):
    """Keep only OpenAI wire fields."""
    out = {k: m[k] for k in ("role", "content", "tool_calls", "tool_call_id") if m.get(k) is not None}
    if m["role"] == "assistant" and m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    return out


sessions = []
for path in sorted(glob.glob(f"{args.traj_dir}/*.traj.json*")):
    msgs = load(path)
    turns = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        try:
            n = m["extra"]["response"]["usage"]["completion_tokens"]
        except (KeyError, TypeError):
            n = 600
        turns.append((i, n))
    sessions.append(([wire(m) for m in msgs if m["role"] != "exit"], turns))

sessions.sort(key=lambda s: -len(s[1]))
pool = list(reversed(sessions))
results, lock = [], threading.Lock()


def worker():
    while True:
        with lock:
            if not pool:
                return
            msgs, turns = pool.pop()
        for i, n in turns[args.stride - 1 :: args.stride]:
            body = {
                "model": args.model,
                "messages": msgs[:i],
                "tools": [BASH_TOOL],
                "max_tokens": min(n + 16, args.cap),
                "chat_template_kwargs": {"enable_thinking": bool(args.think)},
            }
            t0 = time.perf_counter()
            r = httpx.post(f"http://127.0.0.1:{args.port}/v1/chat/completions", json=body, timeout=1800).json()
            usage = r["usage"]
            with lock:
                results.append((usage["prompt_tokens"], usage["completion_tokens"], time.perf_counter() - t0))


t0 = time.perf_counter()
threads = [threading.Thread(target=worker) for _ in range(args.workers)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0
out = sum(r[1] for r in results)
print(json.dumps({
    "workers": args.workers,
    "requests": len(results),
    "wall_s": round(wall, 1),
    "out_tokens": out,
    "wall_tok_s": round(out / wall, 1),
    "mean_prompt_tokens": round(sum(r[0] for r in results) / len(results)),
}))
