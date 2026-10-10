#!/usr/bin/env python3
"""Throughput-vs-actual-decode-batch curve from a ninfer --request-log-jsonl (request_done records).
For each request: decode interval = [done - decode_s, done]; n_bar = time-average number of requests decoding during it.
Aggregate rate at batch n ~ n * mean per-stream rate of requests with round(n_bar)==n  (token-weighted).
usage: batch_curve.py req_log.jsonl [...]"""
import json, sys, collections

reqs = []
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("event") != "request_done" or "result" not in r: continue
        t = r["timings_seconds"]; d = t["decode"]
        if d <= 0.2: continue  # too short to carry a rate
        end = r["timestamp_unix_ms"] / 1000; sp = r.get("speculative") or {}
        reqs.append(dict(s=end - d, e=end, out=r["result"]["completion_tokens"], d=d, ctx=r["result"]["prompt_tokens"],
                         rounds=sp.get("rounds", 0), acc=sp.get("accepted_tokens", 0)))
ev = sorted([(r["s"], 1) for r in reqs] + [(r["e"], -1) for r in reqs])
def nbar(r):  # time-average concurrency over [s,e]
    cur = 0; last = None; area = 0.0
    for t, k in ev:
        if last is not None and t > r["s"] and last < r["e"]:
            area += cur * (min(t, r["e"]) - max(last, r["s"])) if min(t, r["e"]) > max(last, r["s"]) else 0
        cur += k; last = t
    return area / (r["e"] - r["s"])
bins = collections.defaultdict(list)
for r in reqs: bins[max(1, round(nbar(r)))].append(r)
print("n  requests  tokens  per-stream tok/s  aggregate tok/s  tau   mean_ctx")
for n in sorted(bins):
    b = bins[n]; o = sum(r["out"] for r in b); d = sum(r["d"] for r in b); ro = sum(r["rounds"] for r in b); ac = sum(r["acc"] for r in b)
    if len(b) < 3: continue
    print(f"{n}  {len(b):6d}  {o:7d}  {o/d:8.1f}  {n*o/d:10.1f}  {1+ac/ro if ro else float('nan'):6.2f}  {sum(r['ctx'] for r in b)/len(b):8.0f}")
