#!/usr/bin/env python3
"""Summarise ninfer-serve --request-log-jsonl files from an agent run: context, cache hits, output length,
tokens per round, per-position acceptance, time split, and breakdowns by context and output length.

usage: agent_stats.py REQUEST_LOG [REQUEST_LOG ...]
"""
import json, sys, statistics as st

def pct(x, p):
    x = sorted(x); return x[min(len(x) - 1, int(p / 100 * len(x)))] if x else float("nan")

rows = []
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("event") != "request_done" or "result" not in r: continue
        res, sp, t = r["result"], r.get("speculative") or {}, r["timings_seconds"]
        rounds = sp.get("rounds") or 0
        rows.append(dict(
            ctx=res["prompt_tokens"], hit=res.get("prefix_cache_hit_tokens", 0), new=res["computed_prefill_tokens"],
            out=res["completion_tokens"], tools=res.get("tool_call_count", 0),
            dec=t["decode"], pre=t["prefill"], ttft=t["ttft"], tot=t["total"], rounds=rounds,
            acc=sp.get("accepted_tokens", 0), app=sp.get("accepted_per_position", []), drafted=sp.get("drafted_tokens", 0),
            fin=res["finish_reason"]))
n = len(rows); print(f"requests {n}")
if not n: sys.exit()
for k, name in [("ctx", "prompt tokens"), ("hit", "prefix-cache hit tokens"), ("new", "computed prefill tokens"),
                ("out", "completion tokens")]:
    v = [r[k] for r in rows]
    print(f"{name:26s} mean {st.mean(v):8.0f} p10 {pct(v,10):7.0f} p50 {pct(v,50):7.0f} p90 {pct(v,90):7.0f} max {max(v):7.0f}")
hit = sum(r["hit"] for r in rows) / sum(r["ctx"] for r in rows)
print(f"prefix hit fraction of prompt tokens: {hit:.3f}")
print(f"turns with tool calls: {sum(1 for r in rows if r['tools'])/n:.3f}")
tout = sum(r["out"] for r in rows); tdec = sum(r["dec"] for r in rows)
rd = sum(r["rounds"] for r in rows); ac = sum(r["acc"] for r in rows)
print(f"token-weighted decode rate per request {tout/tdec:.1f} tok/s (decode phase only, includes batching effects)")
if rd:
    print(f"tokens/round tau = {1+ac/rd:.3f}  (accepted {ac}, rounds {rd})")
    pos = len(rows[0]["app"]); tot = [sum(r["app"][i] for r in rows if len(r["app"]) > i) for i in range(pos)]
    print("per-position survival (accepted at depth i / rounds):", [round(x / rd, 3) for x in tot])
print(f"time split (sum over requests): prefill {sum(r['pre'] for r in rows):.0f}s  decode {tdec:.0f}s  total {sum(r['tot'] for r in rows):.0f}s")
# by context bucket
for lo, hi in [(0, 8000), (8000, 16000), (16000, 32000), (32000, 64000), (64000, 200000)]:
    s = [r for r in rows if lo <= r["ctx"] < hi]
    if not s: continue
    o = sum(r["out"] for r in s); d = sum(r["dec"] for r in s); rr = sum(r["rounds"] for r in s); aa = sum(r["acc"] for r in s)
    print(f"ctx {lo:6d}-{hi:6d}: n={len(s):4d} out/turn {o/len(s):6.0f} dec {o/d:6.1f} tok/s tau {1+aa/rr if rr else float('nan'):.2f} new/turn {sum(r['new'] for r in s)/len(s):6.0f} ttft {st.mean(r['ttft'] for r in s):.2f}s")
for lo, hi in [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 100000)]:
    s = [r for r in rows if lo <= r["out"] < hi]
    if not s: continue
    o = sum(r["out"] for r in s); d = sum(r["dec"] for r in s); rr = sum(r["rounds"] for r in s); aa = sum(r["acc"] for r in s)
    print(f"out {lo:5d}-{hi:6d}: n={len(s):4d} share_of_tokens {o/tout:.2f} dec {o/d:6.1f} tok/s tau {1+aa/rr if rr else float('nan'):.2f}")
