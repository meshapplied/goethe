"""Reference reader for a Goethe trace folder: loads turns and rebuilds each turn's full history (docs/traces.md)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def load_turns(folder: Path) -> dict[str, dict[str, Any]]:
    """Every turn under folder/v1, keyed by turn_id; a torn last member (crash mid-write) ends that file's turns."""
    turns: dict[str, dict[str, Any]] = {}
    for f in sorted((folder / "v1").rglob("*.jsonl.gz")):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    t = json.loads(line)
                    t["_file"] = str(f.relative_to(folder))
                    turns[t["turn_id"]] = t
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError):
            continue
    return turns


def reply_message(turn: dict[str, Any]) -> dict[str, Any] | None:
    """The assistant message the model produced in this turn, as a client would append it."""
    body = turn["response"]["body"]
    if not isinstance(body, dict):
        return None
    if turn["api"] == "chat.completions":
        choices = body.get("choices") or [{}]
        return choices[0].get("message")
    if turn["api"] == "messages":
        return {"role": "assistant", "content": body.get("content", [])}
    return None


def reply_items(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """What a client appends to its history after this turn: one message, or the Responses output items."""
    body = turn["response"]["body"]
    if turn["api"] == "responses":
        return list(body.get("output", [])) if isinstance(body, dict) else []
    reply = reply_message(turn)
    return [reply] if reply else []


def counted(turn: dict[str, Any], items: list[dict[str, Any]]) -> int:
    """History length as messages_from counts it: Responses reasoning items are not counted."""
    if turn["api"] == "responses":
        return sum(i.get("type") != "reasoning" for i in items)
    return len(items)


def history(turns: dict[str, dict[str, Any]], turn: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages sent in this turn: the parent's history, its reply when with_reply, then the new messages."""
    chain = [turn]
    while chain[-1].get("parent"):
        chain.append(turns[chain[-1]["parent"]["turn_id"]])
    base = list(chain[-1].get("messages", []))
    for parent, child in zip(reversed(chain), reversed(chain[:-1]), strict=False):
        if child["parent"]["with_reply"]:
            base.extend(reply_items(parent))
        expected = child["messages_from"]
        if counted(child, base) != expected:
            raise ValueError(f"{child['turn_id']}: rebuilt {len(base)} history messages, expected {expected}")
        base = base + list(child.get("messages", []))
    return base


def params(turns: dict[str, dict[str, Any]], turn: dict[str, Any]) -> dict[str, Any]:
    """Request parameters (everything but messages), following params_from_parent up the chain."""
    while turn.get("params_from_parent"):
        turn = turns[turn["parent"]["turn_id"]]
    return turn.get("params", {})


def alternatives(turns: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Groups of turns that answered the same request (regenerations), oldest first."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in turns.values():
        if t.get("request_hash"):
            groups.setdefault((t["key_id"], t["request_hash"]), []).append(t)
    return [[t["turn_id"] for t in sorted(g, key=lambda t: t["started_at"])] for g in groups.values() if len(g) > 1]
