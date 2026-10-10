# Results: Qwen3.8-27B NVFP4 on one RTX 5090

Decode throughput of the NInfer engine with speculative decoding, measured on replayed agent traffic.
Everything here is reproducible with the scripts in `bench/`.

## Best config by decode batch n

Aggregate tokens per second, mean context about 28k tokens, thinking on, fp8 KV cache.
`tok/round` is the mean number of tokens committed per verify round (accepted draft tokens plus one).
Aggregate = n x token-weighted per-stream rate, over requests binned by their time-averaged number of
concurrently decoding streams.

| config           | tok/round | n=1 | n=2 | n=3 | n=4 | n=5 |
|------------------|-----------|-----|-----|-----|-----|-----|
| MTP, K=3         | 2.86      | 173 | 311 | 428 | 565 | 648 |
| MTP, K=4         | 3.13      | 181 | 339 | 450 | 577 | 697 |
| DFlash2, K=5     | 3.7       | 209 | 393 | 546 | 615 | -   |
| DFlash2, K=7     | 4.0-4.15  | 239 | 404 | 569 | 686 | -   |
| DFlash2, K=9     | 4.28      | 236 | 396 | -   | -   | -   |
| DFlash2, K=12    | 4.37      | 236 | 405 | 525 | -   | -   |

"-" means fewer than 5 requests landed in that bin, or the config cannot run that many lanes (see below).
Bins with n >= 4 hold fewer than 10 requests each and are noisy. Raw per-bin tables with request counts,
token counts and tok/round are in `bench/runs/`.

How to read it:

- DFlash2 K=7 is the best default for n <= 4. K=5 is slightly worse, K=9 and K=12 gain tokens per round
  (4.28 and 4.37) but lose it back in draft and verify cost, so per-stream speed does not improve over K=7.
- For DFlash2 K=7 the fraction of rounds reaching draft position 1 to 7 is 0.83, 0.65, 0.50, 0.37, 0.28, 0.22, 0.17.
  Conditional acceptance is about 0.77 and flat from position 2, so a longer draft helps only a little.
- MTP is the only option above 4 lanes: DFlash2 needs more engine reservation memory and does not boot
  at 8 lanes with 131k context on a 32 GiB card (see Limits).
- MTP K=4 is a little faster than K=3 at every n.

## Exact run configurations

Server (`serve.sh` prints the same command):

Common flags:

```
--model-id qwen3.8-27b --kv-capacity auto --kv-dtype fp8 \
--lm-head-draft --prefill-chunk 1024
```

Per row (the model path is the first argument):

```
MTP, K=3 and K=4:        --spec mtp     --draft-tokens K  --max-context 131072 --max-concurrency 8
DFlash2, K=5,7,9,12:     --spec dflash2 --draft-tokens K  --max-context 100000 --max-concurrency 4
```

Client (`bench/agent_replay.py`, driven by `bench/run_matrix.sh`):

| parameter            | value |
|----------------------|-------|
| input                | 7 recorded mini-swe-agent trajectories on SWE-bench Lite (`bench/traj/`) |
| requests             | every 6th assistant turn of each trajectory, 37 per matrix pass |
| prompt               | full message history up to that turn, bash tool schema attached |
| max_tokens           | min(recorded completion length + 16, 1500) |
| thinking             | on (`chat_template_kwargs.enable_thinking = true`) |
| sampling             | server defaults, T=1.0, top_p=0.95, top_k=20 |
| context cache        | on (NInfer default), so only the new turn is prefilled |
| workers              | varied per run to populate n = 1 to 5. Each worker replays whole sessions from a shared queue |
| decode batch n       | not the worker count. Taken from the server log per request, see `bench/batch_curve.py` |

Replayed requests have a mean prompt of about 28k tokens. Each config was replayed once, with several worker counts pooled into one request log.

Hardware and software:

| item     | value |
|----------|-------|
| GPU      | RTX 5090, 32 GiB, driver 580.178.04, sm_120a |
| CPU      | Intel Core Ultra 7 265K, 20 threads |
| toolkit  | CUDA 13.4 (nvcc from the pip `nvidia-cuda-nvcc` cu13 packages) |
| engine   | NInfer, commit `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` |
| artifact | `neroued/Qwen3.8-27B-nvfp4-NInfer`, sha256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |

The GPU was shared: a second user's job held about 1 GiB and a few percent of compute on and off during runs.

## Live agent run (MTP K=3, 4 workers)

`bench/run_agent.sh` against a running server: mini-swe-agent on SWE-bench Lite, 4 sessions in parallel,
7 of 8 tasks ran to submission (one container error). 253 requests (`bench/runs/live_agent_run_mtp3.txt`):

| metric                          | value |
|---------------------------------|-------|
| prompt tokens, mean / p50 / p90 | 26.9k / 22.7k / 55.2k |
| prefix cache hit                | 98.1 % of prompt tokens |
| new prefill tokens per turn     | mean 503, p90 1281 |
| completion tokens, mean / p50   | 858 / 317 |
| decode rate per request         | 149.6 tok/s (token-weighted, includes batching) |
| tok/round                       | 2.87 |
| per-position survival           | 0.775, 0.606, 0.488 |
| time split (sum over requests)  | 1.6 % prefill, 90.5 % decode, rest queueing and overhead |

Decode dominates agent traffic once the context cache is warm, so the speculative decoder, not prefill,
decides the user-visible speed.

## Limits

- DFlash2 at 8 lanes and 131k context fails at startup:
  `minimum Engine runtime reservation requires 14451222016 bytes ... only 10160766976 available`.
  The DFlash2 rows therefore use 4 lanes and 100k context. See also upstream issue 385.
- One run per config. No confidence intervals. Differences under about 5 % should be treated as noise.
- mini-swe-agent has a single bash tool and short prompts. Real coding harnesses carry larger system prompts,
  more tools and longer contexts.
- Only decode speed was measured. Task success and output quality were not compared across configs.
  Speculative decoding with exact verification does not change the output distribution, but sampled
  outputs differ run to run.
