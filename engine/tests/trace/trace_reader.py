"""Reference reader for a Goethe trace folder: loads turns and rebuilds each turn's full history (docs/traces.md)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def load_turns(folder: Path) -> dict[str, dict[str, Any]]:
    """Every turn under folder/v1, keyed by turn_id; each session file is concatenated gzip members."""
    turns: dict[str, dict[str, Any]] = {}
    for f in sorted((folder / "v1").rglob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)
                t["_file"] = str(f.relative_to(folder))
                turns[t["turn_id"]] = t
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


def history(turns: dict[str, dict[str, Any]], turn: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages sent in this turn: the parent's history, its reply when with_reply, then the new messages."""
    parent = turn.get("parent")
    if not parent:
        return list(turn.get("messages", []))
    p = turns[parent["turn_id"]]
    base = history(turns, p)
    if parent["with_reply"]:
        base.append(reply_message(p))
    if len(base) != turn["messages_from"]:
        raise ValueError(f"{turn['turn_id']}: rebuilt {len(base)} history messages, expected {turn['messages_from']}")
    return base + list(turn.get("messages", []))


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
