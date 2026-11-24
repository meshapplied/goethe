"""Scores run_test.sh output per scenario: rules, hidden recall checks, stale recall and cost."""

from __future__ import annotations

import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE))
tr = importlib.import_module("trace_reader")
sr = importlib.import_module("score_rules")
W = HERE / ".work"
KEYS = {"off": "fec10900c742", "mem": "285b53e874f8"}


def ends(text: str, line: str) -> bool:
    """Whether a final message ends with the exact line, ignoring markup."""
    return text.strip().strip("`*").rstrip().endswith(line)


def score(cond: str, turns: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per-scenario results for one condition."""
    probes = [json.loads(x) for x in (W / f"note-{cond}.log").open()]
    out = {}
    for scen in sorted((W / "stamina-runs" / f"note-{cond}").glob("scenario_*")):
        t = json.loads((scen / "trajectory.json").read_text())
        s, parts = t["summary"], sr.turns_of(t["trajectory"])
        sid = s.get("session_id", "")
        early = [ends(p["last"], "STATUS: DONE") for p in parts[1:8]]
        late = [ends(p["last"], "RESULT: OK") for p in parts[8:]]
        stale_rule = [ends(p["last"], "STATUS: DONE") for p in parts[8:]]
        tests = [sr.test_edits(turns, sid, KEYS[cond], p["start"], p["end"]) == 0 for p in parts[2:]]
        cl = sr.changelog_counts(scen, len(parts))
        log = [cl[i] > (cl[i - 1] if i > 3 else 0) for i in range(3, len(parts)) if cl[i] >= 0]
        pr = [p for p in probes if p.get("event") == "probe" and p["session"] == sid]
        lost = [p for p in pr if p["lost_from_context"]]
        comp = sum(1 for p in probes if p.get("event") == "compaction" and p["session"] == sid)
        late_billing = [p for p in pr if p["fact"] == "billing" and p["stale"]]
        out[int(scen.name.split("_")[1])] = {
            "sid": sid, "passed": s.get("details", {}).get("turns_passed"),
            "total": s.get("details", {}).get("total_turns"),
            "early": early, "late": late, "stale_rule": stale_rule, "tests": tests, "log": log, "compactions": comp,
            "probes": len(pr), "lost": len(lost), "lost_ok": sum(p["correct"] for p in lost),
            "kept": len(pr) - len(lost), "kept_ok": sum(p["correct"] for p in pr if not p["lost_from_context"]),
            "stale_recall": sum(p["stale"] for p in pr), "billing_stale": len(late_billing),
            **sr.effort(turns, sid, KEYS[cond]),
        }
    return out


def pct(xs: list[bool]) -> str:
    """Passed over total."""
    return f"{sum(xs)}/{len(xs)}"


def main() -> None:
    """Print each condition, then the paired per-scenario differences with a bootstrap interval."""
    turns = tr.load_turns(W / "traces-stamina")
    res = {c: score(c, turns) for c in KEYS}
    for c, r in res.items():
        print("==", c)
        for k, v in r.items():
            print(f" s{k}: passed {v['passed']}/{v['total']} STATUS {pct(v['early'])} RESULT {pct(v['late'])} "
                  f"old-rule-after-change {sum(v['stale_rule'])} tests {pct(v['tests'])} changelog {pct(v['log'])} "
                  f"compactions {v['compactions']} recall-lost {v['lost_ok']}/{v['lost']} "
                  f"recall-kept {v['kept_ok']}/{v['kept']} stale {v['stale_recall']} "
                  f"calls {v['calls']} prompt {v['prompt_M']}M out {v['output_k']}k min {v['model_min']}")
        t = {k: sum(v[k] for v in r.values()) for k in ("lost", "lost_ok", "kept", "kept_ok", "stale_recall", "calls",
                                                        "prompt_M", "output_k", "model_min", "compactions")}
        rules = {k: pct([x for v in r.values() for x in v[k]]) for k in ("early", "late", "tests", "log")}
        print(" total", rules, {k: round(x, 1) for k, x in t.items()},
              "old rule after change", sum(sum(v["stale_rule"]) for v in r.values()))
    rate = {c: {k: v["lost_ok"] / max(v["lost"], 1) for k, v in r.items()} for c, r in res.items()}
    ks = sorted(res["off"])
    d = [rate["mem"][k] - rate["off"][k] for k in ks]
    random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
    print("recall of lost facts, mem minus off per scenario:", [round(x, 2) for x in d],
          "mean", round(sum(d) / len(d), 3), "95% CI", round(bs[125], 3), round(bs[4875], 3))
    for m in ("calls", "prompt_M", "output_k", "model_min"):
        pc = [(res["mem"][k][m] - res["off"][k][m]) / max(res["off"][k][m], 1e-9) for k in ks]
        bs = sorted(sum(random.choice(pc) for _ in pc) / len(pc) for _ in range(5000))
        mean = 100 * sum(pc) / len(pc)
        print(f"{m}: mem vs off per scenario mean {mean:+.0f}% CI {100 * bs[125]:+.0f}% {100 * bs[4875]:+.0f}%")
    for c in KEYS:
        by: Counter[str] = Counter()
        ok: Counter[str] = Counter()
        fails = Counter()
        for x in (W / f"note-{c}.log").open():
            p = json.loads(x)
            if p.get("event") == "probe" and p["lost_from_context"]:
                by[p["fact"]] += 1
                ok[p["fact"]] += p["correct"]
            if p.get("event") == "update_failed":
                fails[p["error"][:70]] += 1
        print(c, "lost-fact recall by fact", {f: f"{ok[f]}/{by[f]}" for f in by}, "update failures", dict(fails))


if __name__ == "__main__":
    main()
