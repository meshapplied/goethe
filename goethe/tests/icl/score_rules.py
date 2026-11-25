"""Scores the chat-only rules variant: rule compliance per turn after each rule was given, plus effort."""

from __future__ import annotations

import datetime
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
tr = importlib.import_module("trace_reader")
RUNS = HERE / ".work" / "stamina-runs"
TRACES = HERE / ".work" / "traces-stamina"
KEYS = {"off": "fec10900c742", "mem": "285b53e874f8"}
NEW_TURN = "Now I need you to make the following changes"
TOOL_WRITE = {"edit", "write", "patch", "multiedit"}


def utc(local: str) -> str:
    """Trajectory timestamps are local (CET) without zone; traces are UTC."""
    t = datetime.datetime.fromisoformat(local[:19]) - datetime.timedelta(hours=1)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def turns_of(traj: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a trajectory into benchmark turns: start time and the agent's last output of the turn."""
    out: list[dict[str, Any]] = []
    for e in traj:
        if e["type"] in ("prompt", "user_input") and (not out or NEW_TURN in (e.get("content") or "")):
            out.append({"start": utc(e["timestamp"]), "last": ""})
        elif e["type"] == "assistant" and out:
            out[-1]["last"] = e.get("content") or ""
    for a, b in zip(out, out[1:] + [{"start": "9999"}], strict=False):
        a["end"] = b["start"]
    return out


def changelog_counts(scen: Path, n: int) -> list[int]:
    """change: lines in CHANGELOG.md of the last snapshot of each turn (-1 when the turn has no snapshot)."""
    out = []
    for t in range(n):
        snaps = sorted(scen.glob(f"agent_code/turn_{t}_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        if not snaps:
            out.append(-1)
            continue
        f = snaps[-1] / "CHANGELOG.md"
        text = f.read_text(errors="replace") if f.exists() else ""
        out.append(sum(1 for line in text.splitlines() if line.strip().lower().startswith("change:")))
    return out


def test_edits(turns: dict[str, dict[str, Any]], sid: str, key: str, start: str, end: str) -> int:
    """Write or edit tool calls to test files made by the agent of one session in a time window."""
    n = 0
    for t in turns.values():
        if t.get("session_hint") != sid or t["key_id"] != key or not (start <= t["started_at"] < end):
            continue
        msg = tr.reply_message(t) or {}
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            if fn.get("name", "").lower() in TOOL_WRITE:
                args = str(fn.get("arguments", ""))
                path = re.search(r'"(?:filePath|file_path|path)"\s*:\s*"([^"]+)"', args)
                if path and re.search(r"^/workspace/(.*/)?(test[^/]*\.py|tests?/.*)$", path.group(1)):
                    n += 1
            write = r"(sed -i[^|;&\\n]*|>>?\s*|tee\s+|mv\s+|rm\s+(-\w+\s+)?)/workspace/(\S*/)?test"
            if fn.get("name", "").lower() == "bash" and re.search(write, str(fn.get("arguments", ""))):
                n += 1
    return n


def effort(turns: dict[str, dict[str, Any]], sid: str, key: str) -> dict[str, float]:
    """Model calls and tokens of one session."""
    ts = [t for t in turns.values() if t.get("session_hint") == sid and t["key_id"] == key]
    us = [(t["response"]["body"] or {}).get("usage") or {} if isinstance(t["response"]["body"], dict) else {}
          for t in ts]
    return {"calls": len(ts), "prompt_M": round(sum(u.get("prompt_tokens", 0) for u in us) / 1e6, 2),
            "output_k": round(sum(u.get("completion_tokens", 0) for u in us) / 1e3, 1),
            "model_min": round(sum(t["duration_s"] for t in ts) / 60, 1)}


def pct(xs: list[bool]) -> str:
    """Compliant over total, as text."""
    return f"{sum(xs)}/{len(xs)}"


def main() -> None:
    """Print per scenario and condition: compliance with each rule after it was given, turns passed, effort."""
    turns = tr.load_turns(TRACES)
    for cond in ("off", "mem"):
        for scen in sorted((RUNS / f"rules-{cond}").glob("scenario_*")):
            t = json.loads((scen / "trajectory.json").read_text())
            summ, parts = t["summary"], turns_of(t["trajectory"])
            sid = summ.get("session_id", "")
            r1 = [p["last"].strip().strip("`*").rstrip().endswith("STATUS: DONE") for p in parts[1:]]
            r2 = [test_edits(turns, sid, KEYS[cond], p["start"], p["end"]) == 0 for p in parts[2:]]
            cl = changelog_counts(scen, len(parts))
            r3 = [cl[i] > (cl[i - 1] if i > 3 else 0) for i in range(3, len(parts)) if cl[i] >= 0]
            det = summ.get("details", {})
            print(f"{cond:<3} {scen.name}: turns passed {det.get('turns_passed')}/{det.get('total_turns')}  "
                  f"STATUS line {pct(r1)}  tests untouched {pct(r2)}  changelog {pct(r3)}  "
                  f"{json.dumps(effort(turns, sid, KEYS[cond]))}")
            print(f"      per turn STATUS {''.join('Y' if x else '.' for x in r1)}  tests "
                  f"{''.join('Y' if x else '.' for x in r2)}  changelog {''.join('Y' if x else '.' for x in r3)}")


if __name__ == "__main__":
    main()
