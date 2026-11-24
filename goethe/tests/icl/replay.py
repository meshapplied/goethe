"""Replays one recorded session through a live gateway with learning on, then asks about facts lost from context."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
tr = importlib.import_module("trace_reader")


def post(url: str, key: str, body: dict, session: str | None) -> tuple[int, dict | str]:
    """POST one chat completion."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if session:
        headers["X-Session-Id"] = session
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def note(session: str) -> dict:
    """The stored note of a session."""
    out = subprocess.run(["goethe", "learn", "--print", session, "--json"], capture_output=True, text=True)
    return json.loads(out.stdout)[0] if out.returncode == 0 else {}


def main() -> None:
    """Send every recorded request of the session in order, one output token each."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", default=str(HERE / ".work" / "traces"))
    p.add_argument("--session", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--url", default="http://127.0.0.1:30100")
    p.add_argument("--model", default="meshapplied/Qwen3.8-27B-NVFP4-NInfer")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    key = Path(a.key_file).read_text().strip()
    turns = tr.load_turns(Path(a.traces))
    ours = sorted((t for t in turns.values() if t.get("session_hint") == a.session), key=lambda t: t["started_at"])
    if a.limit:
        ours = ours[: a.limit]
    sid = f"replay-{a.session[:20]}-{int(time.time())}"
    tooled = [t for t in ours if tr.params(turns, t).get("tools")]
    real = sum(1 for t in tooled if t.get("parent") is None and t["seq"] > 0)
    print(f"replaying {len(ours)} requests, {len(tooled)} with tools, {real} history breaks as {sid}", flush=True)
    bad = 0
    started = time.time()
    last_body = None
    for i, t in enumerate(ours):
        par = {k: v for k, v in tr.params(turns, t).items() if k not in ("stream", "stream_options", "max_tokens")}
        body = {**par, "messages": tr.history(turns, t), "model": a.model, "max_tokens": 1, "stream": False}
        code, _ = post(a.url, key, body, sid)
        bad += code != 200
        if par.get("tools"):
            last_body = body
        if i % 50 == 0:
            n = note(sid)
            print(f"{i}: {time.time() - started:.0f}s bad={bad} compactions={n.get('compactions')} "
                  f"updates={n.get('updates')} waiting={n.get('waiting_messages')}", flush=True)
    time.sleep(30)
    n = note(sid)
    print(json.dumps({k: n.get(k) for k in ("requests", "compactions", "updates", "waiting_messages", "prompt_tokens",
                                            "completion_tokens")}), f"real breaks {real}, bad {bad}", flush=True)
    live = [e for e in n["entries"] if e["live"]]
    for e in live:
        print(f"  {e['id']} [{e['section']}] {e['text'][:170]}", flush=True)
    print("note in prompt (chars):", len(n.get("in_prompt") or ""), flush=True)
    print(f"cost: {n['prompt_tokens']} + {n['completion_tokens']} tokens", flush=True)
    facts = json.loads((HERE / "facts.json").read_text())
    seen = "\n".join(json.dumps(m["content"]) for t in ours for m in tr.history(turns, t) if m.get("role") == "user")
    current = json.dumps(last_body["messages"]) if last_body else ""
    score = {"with": [0, 0], "without": [0, 0], "stale": 0}
    for f in facts:
        planted = [v for v in f["versions"] if v["marker"] in seen]
        if not planted or planted[-1]["marker"] in current:
            continue
        ask = f"Quick check, in a few words, without any tool: {f['question']} Say unknown if you do not know."
        q = {**last_body, "messages": [*last_body["messages"], {"role": "user", "content": ask}], "max_tokens": 60,
             "chat_template_kwargs": {"enable_thinking": False}, "tool_choice": "none"}
        for name, s in (("with", sid), ("with", sid), ("without", None)):
            code, out = post(a.url, key, q, s)
            reply = out["choices"][0]["message"].get("content") or "" if code == 200 else str(out)[:100]
            ok = planted[-1]["answer"].lower() in reply.lower()
            score[name][0] += ok
            score[name][1] += 1
            if name == "with":
                score["stale"] += any(v["answer"].lower() in reply.lower() for v in planted[:-1])
            print(f"  {f['id']} {name}: {'right' if ok else 'wrong'}: {reply[:90]!r}", flush=True)
    print("recall of facts gone from context:", score, flush=True)


if __name__ == "__main__":
    main()
