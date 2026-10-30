"""Mixed prompt-length load test against a Goethe gateway: short chats under load while long prompts arrive."""

import argparse
import asyncio
import json
import os
import random
import ssl
import statistics
import time
from pathlib import Path

import aiohttp

WORDS = ("scheduler kernel request batch token cache prefill decode tensor stream gateway engine replica memory "
         "latency throughput attention expert router layer budget queue state graph").split()


def filler(n_tokens: int, rng: random.Random) -> str:
    """About n_tokens of text (1.11 tokens per word, measured); a random lead line defeats prefix caching."""
    words = [rng.choice(WORDS) for _ in range(int(n_tokens / 1.11))]
    lines = [" ".join(words[i:i + 16]) for i in range(0, len(words), 16)]
    return f"Session {rng.getrandbits(64):x}.\n" + "\n".join(lines)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


class Client:
    def __init__(self, args):
        self.url = args.url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {args.key}", "Content-Type": "application/json"}
        self.model = args.model
        self.ssl = ssl.create_default_context()
        self.ssl.check_hostname = False
        self.ssl.verify_mode = ssl.CERT_NONE
        self.session = None

    async def one(self, kind: str, prompt_tokens: int, max_tokens: int, rng: random.Random) -> dict:
        """One streamed request; records status, time to first token and every gap between streamed chunks."""
        body = {"model": self.model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": filler(prompt_tokens, rng) +
                              "\n\nSummarize the text above in detail, as a long numbered list."}]}
        rec = {"kind": kind, "prompt_target": prompt_tokens, "t0": time.time(), "status": None, "ttft": None,
               "gaps": [], "tokens": 0, "prompt_tokens": None, "error": None}
        t0 = time.perf_counter()
        last = None
        try:
            async with self.session.post(self.url, json=body, headers=self.headers, ssl=self.ssl) as r:
                rec["status"] = r.status
                if r.status != 200:
                    rec["error"] = (await r.text())[:300]
                    return rec
                async for raw in r.content:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("usage"):
                        rec["tokens"] = d["usage"].get("completion_tokens", 0)
                        rec["prompt_tokens"] = d["usage"].get("prompt_tokens")
                    if any((c.get("delta") or {}).get("content") for c in d.get("choices", [])):
                        now = time.perf_counter()
                        if last is None:
                            rec["ttft"] = now - t0
                        else:
                            rec["gaps"].append(now - last)
                        last = now
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["total"] = time.perf_counter() - t0
        return rec


async def closed_loop(client, kind, n, duration, prompt_rng, out_tokens, results, seed):
    """n workers each send one request after another until the duration ends."""
    end = time.time() + duration

    async def worker(i):
        rng = random.Random(seed * 1000 + i)
        while time.time() < end:
            rec = await client.one(kind, rng.randint(*prompt_rng), rng.randint(*out_tokens), rng)
            results.append(rec)
            if rec["status"] != 200 or rec["error"]:
                await asyncio.sleep(1.0)

    await asyncio.gather(*(worker(i) for i in range(n)))


async def injector(client, sizes, every, duration, out_tokens, results, seed):
    """Send one long prompt every `every` seconds, cycling through sizes, without waiting for replies."""
    rng = random.Random(seed)
    tasks, end, i = [], time.time() + duration, 0
    while time.time() < end:
        tasks.append(asyncio.create_task(client.one("long", sizes[i % len(sizes)], out_tokens, rng)))
        i += 1
        await asyncio.sleep(every)
    for t in tasks:
        results.append(await t)


def summarize(name, results, out_dir: Path):
    rows = []
    for kind in sorted({r["kind"] for r in results}):
        rs = [r for r in results if r["kind"] == kind]
        ok = [r for r in rs if r["status"] == 200 and not r["error"]]
        gaps = [g for r in ok for g in r["gaps"]]
        ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
        tps = [r["tokens"] / (r["total"] - r["ttft"]) for r in ok
               if r["ttft"] and r["tokens"] and r["total"] > r["ttft"]]
        codes: dict = {}
        for r in rs:
            code = "stream-error" if r["status"] == 200 and r["error"] else str(r["status"])
            codes[code] = codes.get(code, 0) + 1
        rows.append({
            "scenario": name, "kind": kind, "requests": len(rs), "ok": len(ok), "codes": codes,
            "ttft_p50": pct(ttft, .5), "ttft_p95": pct(ttft, .95), "ttft_max": max(ttft) if ttft else None,
            "gap_p50_ms": 1e3 * pct(gaps, .5) if gaps else None, "gap_p99_ms": 1e3 * pct(gaps, .99) if gaps else None,
            "gap_max_s": max(gaps) if gaps else None, "stalls_over_1s": sum(g > 1 for g in gaps),
            "decode_tok_s_p50": statistics.median(tps) if tps else None,
            "errors": sorted({r["error"] for r in rs if r["error"]})[:3],
        })
    (out_dir / f"{name}.json").write_text(json.dumps({"rows": rows, "results": results}, indent=1))
    for row in rows:
        print(json.dumps(row))
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://100.70.155.114:8080/v1")
    ap.add_argument("--key", default=os.environ.get("GOETHE_API_KEY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", required=True,
                    choices=["baseline", "mixed", "overload", "kv_pressure", "overlong"])
    ap.add_argument("--short", type=int, default=8, help="concurrent short-chat workers")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--long-sizes", default="32000,64000,100000")
    ap.add_argument("--long-every", type=float, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / ".work"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(args)
    results: list = []
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        client.session = s
        short = (1000, 3000)
        if args.scenario == "baseline":
            await closed_loop(client, "short", args.short, args.duration, short, (256, 512), results, 1)
        elif args.scenario == "mixed":
            sizes = [int(x) for x in args.long_sizes.split(",")]
            await asyncio.gather(
                closed_loop(client, "short", args.short, args.duration, short, (256, 512), results, 2),
                injector(client, sizes, args.long_every, args.duration, 128, results, 3))
        elif args.scenario == "overload":
            await closed_loop(client, "mixed", args.short, args.duration, (500, 24000), (128, 512), results, 4)
        elif args.scenario == "kv_pressure":
            await asyncio.gather(
                closed_loop(client, "short", args.short, args.duration, short, (256, 512), results, 5),
                injector(client, [max(int(x) for x in args.long_sizes.split(","))], 5, min(args.duration, 20), 512,
                         results, 6))
        elif args.scenario == "overlong":
            results.append(await client.one("overlong", 140000, 64, random.Random(7)))
            results.append(await client.one("near_limit", 127000, 4000, random.Random(8)))
    summarize(args.scenario, results, out)


if __name__ == "__main__":
    asyncio.run(main())
