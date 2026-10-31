"""End-to-end check of goethe trace against a running gateway with one model loaded (docs/traces.md)."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import trace_reader

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the project.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ANTHROPIC_TOOLS = [{"name": "read_file", "description": "Read a file from the project.",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                     "required": ["path"]}}]
NO_THINK = {"enable_thinking": False}


class Gateway:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def post(self, path: str, key: str, body: dict, headers: dict | None = None) -> tuple[int, Any]:
        """POST JSON; streamed replies are read to the end and returned as their list of data events."""
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=600) as r:
                if body.get("stream"):
                    events = []
                    for raw in r:
                        line = raw.decode().strip()
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            events.append(json.loads(line[5:]))
                    return r.status, events
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    def chat(self, key: str, messages: list, stream: bool = False, headers: dict | None = None, **extra) -> dict:
        """One chat turn; returns the assistant message, rebuilt from the stream when streaming."""
        body = {"model": self.model, "messages": messages, "max_tokens": extra.pop("max_tokens", 200),
                "temperature": 0.6, "stream": stream, "chat_template_kwargs": extra.pop("kwargs", NO_THINK), **extra}
        code, out = self.post("/v1/chat/completions", key, body, headers)
        if code != 200:
            raise RuntimeError(f"chat {code}: {out}")
        if not stream:
            return out["choices"][0]["message"]
        msg: dict[str, Any] = {"role": "assistant", "content": ""}
        calls: dict[int, dict] = {}
        for e in out:
            for c in e.get("choices", []):
                d = c.get("delta") or {}
                msg["content"] += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                                 "function": {"name": "", "arguments": ""}})
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["function"]["name"] += (tc.get("function") or {}).get("name") or ""
                    slot["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        return msg

    def disconnect_mid_stream(self, key: str, messages: list) -> None:
        """Start a long streamed reply, read a few chunks, then drop the connection."""
        body = {"model": self.model, "messages": messages, "max_tokens": 2000, "stream": True,
                "chat_template_kwargs": NO_THINK}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = urllib.request.urlopen(req, context=CTX, timeout=600)
        for _ in range(8):
            r.readline()
        r.close()


def goethe(*args: str) -> str:
    """Run the goethe CLI and return stdout; a failure stops the test."""
    p = subprocess.run(["goethe", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"goethe {' '.join(args)}: {p.stderr or p.stdout}")
    return p.stdout


def status() -> dict:
    return json.loads(goethe("trace", "--json"))


def canon(msg: dict) -> tuple:
    """Role, visible text and tool calls with parsed arguments: what must match between sent and rebuilt history."""
    text, calls = [], []
    content = msg.get("content")
    if isinstance(content, str):
        text.append(content.strip())
    for part in content if isinstance(content, list) else []:
        if part.get("type") == "text":
            text.append(part["text"].strip())
        elif part.get("type") == "tool_use":
            calls.append((part["name"], json.dumps(part["input"], sort_keys=True)))
        elif part.get("type") == "tool_result":
            inner = part.get("content")
            text.append("[tool_result]" + (inner if isinstance(inner, str) else json.dumps(inner)))
    for tc in msg.get("tool_calls") or []:
        args = tc["function"]["arguments"]
        try:
            args = json.dumps(json.loads(args), sort_keys=True)
        except (TypeError, ValueError):
            pass
        calls.append((tc["function"]["name"], args))
    return msg.get("role"), "\n".join(text), tuple(calls)


def agent_conversation(gw: Gateway, key: str, n: int, stream: bool, headers: dict | None = None) -> list[list]:
    """A small tool-using agent loop; returns the messages list sent at each turn."""
    sent = []
    messages = [{"role": "system", "content": "You are a coding agent. Use read_file when you need a file."},
                {"role": "user", "content": f"Task {n}: read src/main_{n}.rs and tell me in one line what it does."}]
    for step in range(4):
        sent.append([dict(m) for m in messages])
        reply = gw.chat(key, messages, stream=stream, tools=TOOLS, headers=headers, max_tokens=300)
        clean = {"role": "assistant", "content": reply.get("content") or ""}
        if reply.get("tool_calls"):
            clean["tool_calls"] = reply["tool_calls"]
        messages.append(clean)
        if not reply.get("tool_calls") or step == 3:
            messages.append({"role": "user", "content": "Thanks. Now say it in five words."})
            sent.append([dict(m) for m in messages])
            gw.chat(key, messages, stream=stream, headers=headers, max_tokens=60)
            break
        for tc in reply["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": f"fn main() {{ println!(\"server {n} on port {8000 + n}\"); }}"})
    return sent


def anthropic_conversation(gw: Gateway, key: str) -> list[list]:
    """Two Anthropic Messages turns with a tool round trip, streamed."""
    sent = []
    messages = [{"role": "user", "content": "Read README.md with the tool, then summarize it in one line."}]
    body = {"model": gw.model, "max_tokens": 300, "system": "You are a careful agent.", "tools": ANTHROPIC_TOOLS,
            "stream": True, "messages": messages}
    sent.append([dict(m) for m in messages])
    code, events = gw.post("/v1/messages", key, body)
    if code != 200:
        raise RuntimeError(f"messages {code}: {events}")
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    for e in events:
        if e.get("type") == "content_block_start":
            blocks[e["index"]] = dict(e["content_block"])
        elif e.get("type") == "content_block_delta":
            d = e["delta"]
            if d.get("type") == "text_delta":
                blocks[e["index"]]["text"] = blocks[e["index"]].get("text", "") + d["text"]
            elif d.get("type") == "input_json_delta":
                partial[e["index"]] = partial.get(e["index"], "") + d["partial_json"]
    for i, s in partial.items():
        blocks[i]["input"] = json.loads(s) if s.strip() else {}
    content = [blocks[i] for i in sorted(blocks) if blocks[i].get("type") in ("text", "tool_use")]
    messages.append({"role": "assistant", "content": content})
    uses = [b for b in content if b["type"] == "tool_use"]
    follow = [{"type": "tool_result", "tool_use_id": b["id"], "content": "# Demo\nA tiny demo project."} for b in uses]
    messages.append({"role": "user", "content": follow or "Summarize anyway."})
    sent.append([dict(m) for m in messages])
    code, _ = gw.post("/v1/messages", key, {**body, "messages": messages})
    if code != 200:
        raise RuntimeError(f"messages turn 2 {code}")
    return sent


def wait_for_turns(expected: int, timeout: float = 30) -> int:
    """Turns are written by a background thread and counted in the manifest every 2 s."""
    end = time.time() + timeout
    while time.time() < end:
        n = status()["turns"]
        if n >= expected:
            return n
        time.sleep(0.5)
    return status()["turns"]


def check(cond: bool, what: str, failures: list) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://100.70.155.114:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / ".work" / "traces"))
    ap.add_argument("--parallel", type=int, default=12)
    args = ap.parse_args()
    gw = Gateway(args.url, args.model)
    folder = Path(args.dir)
    failures: list[str] = []
    key_a = goethe("key", "--create", "trace-a", "--rpm", "0", "--raw").strip()
    key_b = goethe("key", "--create", "trace-b", "--rpm", "0", "--raw").strip()
    ids = {k["name"]: k["id"] for k in json.loads(goethe("key", "--json")) if k["name"] in ("trace-a", "trace-b")}
    expect: list[tuple[str, list]] = []
    try:
        goethe("trace", "on", "--key", "trace-a", "--dir", str(folder))
        st = status()
        check(st["recording"] and not st["all_keys"] and [k["id"] for k in st["keys"]] == [ids["trace-a"]],
              "trace on --key trace-a records only that key", failures)
        check(oct(folder.stat().st_mode & 0o777) == "0o700", "trace folder is private (0700)", failures)

        single = [{"role": "user", "content": "Name one prime number above 100. Just the number."}]
        gw.chat(key_a, single)
        expect.append(("single", single))
        gw.chat(key_b, [{"role": "user", "content": "key b must not be recorded"}])
        thinking = [{"role": "user", "content": "Is 91 prime? Answer yes or no."}]
        gw.chat(key_a, thinking, stream=True, kwargs={"enable_thinking": True}, max_tokens=400)
        expect.append(("thinking", thinking))
        for i, sent in enumerate(agent_conversation(gw, key_a, 1, stream=True)):
            expect.append((f"agent-stream-{i}", sent))
        agent_sent = agent_conversation(gw, key_a, 2, stream=False)
        for i, sent in enumerate(agent_sent):
            expect.append((f"agent-json-{i}", sent))
        regen = agent_sent[1]
        gw.chat(key_a, regen, tools=TOOLS, max_tokens=300)
        expect.append(("regenerate", regen))
        hinted = [{"role": "user", "content": "Say hi."}]
        gw.chat(key_a, hinted, headers={"X-Session-Id": "client-session-42"})
        gw.chat(key_a, [{"role": "user", "content": "Unrelated question: what is 2+2?"}],
                headers={"X-Session-Id": "client-session-42"})
        expect += [("hinted-1", hinted),
                   ("hinted-2", [{"role": "user", "content": "Unrelated question: what is 2+2?"}])]
        for i, sent in enumerate(anthropic_conversation(gw, key_a)):
            expect.append((f"anthropic-{i}", sent))
        code, _ = gw.post("/v1/completions", key_a, {"model": gw.model, "prompt": "The capital of France is",
                                                       "max_tokens": 8})
        check(code == 200, "completions call works", failures)
        gw.post("/v1/embeddings", key_a, {"model": gw.model, "input": "not traced"})
        gw.disconnect_mid_stream(key_a, [{"role": "user", "content": "Write a 1500 word essay about rivers."}])
        time.sleep(2)

        workers, errors = [], []

        def run(n):
            try:
                agent_conversation(gw, key_a, 100 + n, stream=n % 2 == 0)
            except Exception as e:
                errors.append(repr(e))

        for n in range(args.parallel):
            workers.append(threading.Thread(target=run, args=(n,)))
            workers[-1].start()
        time.sleep(3)
        goethe("trace", "on")
        gw.chat(key_b, [{"role": "user", "content": "key b while every key is on"}])
        for w in workers:
            w.join()
        check(not errors, f"{args.parallel} parallel agent conversations finished ({errors[:2]})", failures)
        goethe("trace", "off", "--key", "trace-b")
        st = status()
        check(st["all_keys"] and [k["id"] for k in st["except"]] == [ids["trace-b"]],
              "trace off --key trace-b excepts it", failures)
        gw.chat(key_b, [{"role": "user", "content": "key b excepted again"}])
        time.sleep(3)
        before = wait_for_turns(10**9, timeout=4)
        goethe("trace", "off")
        gw.chat(key_a, [{"role": "user", "content": "after off"}])
        time.sleep(3)
        after = status()
        check(not after["recording"], "trace off stops recording", failures)
        check(after["turns"] == before, f"no turns recorded after off ({before} -> {after['turns']})", failures)
        limit = (after["bytes"] + 4000) / 1e9
        goethe("trace", "on", "--key", "trace-a", "--max-gb", f"{limit:.9f}")
        for i in range(20):
            gw.chat(key_a, [{"role": "user", "content": f"filling up to the size limit {i}"}])
            time.sleep(0.5)
            if not status()["recording"]:
                break
        time.sleep(3)
        capped = status()
        last = json.loads((folder / "history.jsonl").read_text().splitlines()[-1])
        stopped = capped["stopped"] or ""
        check(not capped["recording"] and "limit was reached" in stopped and capped["turns"] > before,
              f"the size limit turns recording off ({stopped}, {capped['turns'] - before} turns first)", failures)
        check(last.get("reason") == "size limit" and last["was"]["keys"] == [ids["trace-a"]],
              "history.jsonl logs the automatic off with the scope it had", failures)
        refused = subprocess.run(["goethe", "trace", "on"], capture_output=True, text=True)
        check(refused.returncode != 0 and "over its" in refused.stderr, "turning on over the limit is refused",
              failures)
        goethe("trace", "--max-gb", "20")
        check(status()["stopped"] is None and not status()["recording"], "raising the limit alone keeps recording off",
              failures)
    finally:
        goethe("trace", "off")
        for kid in ids.values():
            goethe("key", "--revoke", kid)

    turns = trace_reader.load_turns(folder)
    st = status()
    print(json.dumps({k: st[k] for k in ("turns", "sessions", "bytes", "models", "dropped", "paused")}))
    check(len(turns) == st["turns"], f"manifest turn count matches files ({len(turns)} vs {st['turns']})", failures)
    by_key = {}
    for t in turns.values():
        by_key.setdefault(t["key_id"], []).append(t)
    b_turns = [t for t in by_key.get(ids["trace-b"], [])]
    check(len(b_turns) == 1 and "every key is on" in json.dumps(b_turns[0]["messages"]),
          "key b recorded only while every key was on", failures)
    check(not any(t["api"] not in ("chat.completions", "messages", "completions") for t in turns.values()),
          "embeddings are not traced", failures)
    check(all("after off" not in json.dumps(t.get("messages")) for t in turns.values()), "nothing after off", failures)

    rebuilt_ok, missing = 0, []
    a_turns = by_key.get(ids["trace-a"], [])
    for name, sent in expect:
        target = [canon(m) for m in sent]
        hit = [t for t in a_turns if t.get("messages") is not None
               and [canon(m) for m in trace_reader.history(turns, t)] == target]
        if hit:
            rebuilt_ok += 1
        else:
            missing.append(name)
    check(not missing, f"every sent history rebuilds from the corpus ({rebuilt_ok}/{len(expect)}; missing {missing})",
          failures)
    for t in a_turns:
        if t.get("messages") is not None:
            trace_reader.history(turns, t)
    agent = [t for t in a_turns if "src/main_1.rs" in json.dumps(trace_reader.history(turns, t)) if t.get("messages")]
    check(len({t["session_id"] for t in agent}) == 1 and len(agent) >= 2,
          f"agent loop turns share one session ({len(agent)} turns)", failures)
    check(all(t["parent"] for t in agent if t["seq"] > 0) and all(t.get("params_from_parent") is not None
          or "params" in t for t in agent), "follow-up turns point at their parent", failures)
    stored = sum(len(t["messages"]) for t in agent)
    full = sum(len(trace_reader.history(turns, t)) for t in agent)
    check(stored < full, f"follow-ups store only new messages ({stored} stored for {full} history messages)", failures)
    alts = trace_reader.alternatives(turns)
    check(any(len(g) >= 2 for g in alts), f"regeneration shows up as alternative replies ({len(alts)} groups)",
          failures)
    hinted_turns = [t for t in a_turns if t.get("session_hint") == "client-session-42"]
    check(len(hinted_turns) == 2 and len({t["session_id"] for t in hinted_turns}) == 1,
          "X-Session-Id groups unrelated requests into one session", failures)
    think = [t for t in a_turns if "Is 91 prime" in json.dumps(t.get("messages"))]
    msg = trace_reader.reply_message(think[0]) if think else {}
    check(bool(think) and bool((msg or {}).get("reasoning_content")), "streamed reasoning is kept", failures)
    tool_turns = [t for t in a_turns if (trace_reader.reply_message(t) or {}).get("tool_calls")]
    check(bool(tool_turns) and all(json.loads(c["function"]["arguments"]) is not None
          for t in tool_turns for c in trace_reader.reply_message(t)["tool_calls"]),
          f"tool calls are rebuilt with parseable arguments ({len(tool_turns)} turns)", failures)
    ant = [t for t in a_turns if t["api"] == "messages"]
    check(len(ant) == 2 and ant[0]["session_id"] == ant[1]["session_id"]
          and "system" in trace_reader.params(turns, ant[1]), "Anthropic turns link, with the system prompt in params",
          failures)
    cut = [t for t in a_turns if "1500 word essay" in json.dumps(t.get("messages"))]
    check(len(cut) == 1 and cut[0]["response"]["complete"] is False and cut[0]["reply_hash"] is None,
          "a client disconnect is recorded as incomplete", failures)
    complete = [t for t in a_turns if t not in cut]
    check(all(t["response"]["complete"] and t["response"]["status"] == 200 for t in complete),
          "every other turn is complete with status 200", failures)
    models = json.loads((folder / "models.json").read_text())
    fps = {t["model_fingerprint"] for t in turns.values()}
    check(len(fps) == 1 and fps <= set(models) and models[next(iter(fps))]["files"].get("config.json"),
          "model fingerprint recorded with file hashes in models.json", failures)
    sessions = [json.loads(line) for line in (folder / "sessions.jsonl").read_text().splitlines()]
    check({s["session_id"] for s in sessions} == {t["session_id"] for t in turns.values()},
          "sessions.jsonl lists every session", failures)
    modes = {oct(p.stat().st_mode & 0o777) for p in folder.rglob("*") if p.is_file()}
    check(modes == {"0o600"}, f"every trace file is private ({modes})", failures)
    history = [json.loads(line)["action"] for line in (folder / "history.jsonl").read_text().splitlines()]
    check(history[:2] == ["on", "on"] and history[-1] == "off" and "settings" in history,
          f"history.jsonl logs on/off and setting changes ({history})", failures)
    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
