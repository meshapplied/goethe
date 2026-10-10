# nvfp4-kernels

Run Qwen3.8-27B in NVFP4 on a single RTX 5090 with speculative decoding, using the
[NInfer](https://github.com/Neroued/ninfer) engine. This directory is a thin, pinned wrapper: it builds a
known NInfer commit, fetches a known weight artifact, starts the server with measured settings, and ships the
benchmark used to choose them. It does not vendor or modify NInfer.

On a 5090 the best measured config decodes at about 240 tok/s for one stream and 570 to 690 tok/s aggregate at
3 to 4 concurrent streams, with a mean context of 28k tokens. Details and caveats are in [RESULTS.md](RESULTS.md).

## Requirements

- NVIDIA Blackwell GPU with compute capability 12.0 (RTX 5090 or RTX PRO 6000). NInfer builds for sm_120a only.
- 32 GiB of VRAM for the default profiles. Weights occupy 21.4 GiB on the GPU with the DFlash2 drafter, 19.7 GiB with MTP only.
- Linux, CUDA toolkit 13.x with `nvcc` (13.1 or newer), `cmake` 3.28+, `ninja`, `git`, a C++20 compiler.
- Python 3 with `httpx` for the replay benchmark, and the `hf` CLI (`pip install huggingface_hub`) for the download.
- About 26 GB of disk: the 23.7 GB artifact and a 1.3 GB build tree.

## Quick start

```bash
cd nvfp4-kernels
./install.sh        # clone and build NInfer at the pinned commit (about 10 minutes on 20 threads)
./fetch_model.sh    # download the 23.7 GB artifact and verify its sha256
./serve.sh          # OpenAI and Anthropic compatible server on 127.0.0.1:8080
```

Then:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a function that reverses a linked list in C."}],
  "max_tokens": 400
}'
```

Endpoints: `/v1/chat/completions`, `/v1/responses` and the Anthropic Messages API. Everything is installed under
`$NVFP4_HOME` (default `~/.cache/goethe/nvfp4`). If `nvcc` is not on `PATH`, set `CUDA_HOME`.

## Profiles

`PROFILE=<name> ./serve.sh`

| profile   | drafter | draft tokens | lanes | max context | status |
|-----------|---------|--------------|-------|-------------|--------|
| `default` | DFlash2 | 7            | 4     | 100000      | measured, fastest at 1 to 4 streams |
| `long`    | DFlash2 | 7            | 1     | 131072      | boots, not benchmarked |
| `batch`   | MTP     | 4            | 8     | 131072      | measured, needed above 4 streams |

All profiles use an fp8 KV cache, `--lm-head-draft` and `--prefill-chunk 1024`. Any value can be overridden:

```bash
SPEC=mtp K=3 MAX_CONCURRENCY=8 MAX_CONTEXT=131072 ./serve.sh
PORT=9000 REQUEST_LOG=/tmp/req.jsonl ./serve.sh --some-ninfer-flag
```

Arguments after `serve.sh` are passed to `ninfer-serve` unchanged. `REQUEST_LOG` writes one JSON line per request
with prompt and cached token counts, accepted draft tokens per position, and timings.

DFlash2 cannot start at 8 lanes and 131072 context on a 32 GiB card (engine reservation 14.45 GB against
10.2 GB free after weights), hence the 4 lane limit on the DFlash2 profiles.

## Numbers

Aggregate decode tok/s by actual decode batch n, replayed agent traffic, mean context 28k, thinking on:

| config        | tok/round | n=1 | n=2 | n=3 | n=4 | n=5 |
|---------------|-----------|-----|-----|-----|-----|-----|
| MTP, K=3      | 2.86      | 173 | 311 | 428 | 565 | 648 |
| MTP, K=4      | 3.13      | 181 | 339 | 450 | 577 | 697 |
| DFlash2, K=5  | 3.7       | 209 | 393 | 546 | 615 | -   |
| DFlash2, K=7  | 4.0-4.15  | 239 | 404 | 569 | 686 | -   |
| DFlash2, K=9  | 4.28      | 236 | 396 | -   | -   | -   |
| DFlash2, K=12 | 4.37      | 236 | 405 | 525 | -   | -   |

One run per config, bins with n >= 4 hold fewer than 10 requests. Exact flags, client parameters, hardware and
limits are in [RESULTS.md](RESULTS.md).

## Reproduce

```bash
pip install httpx
bench/run_matrix.sh "mtp:3:8:131072 mtp:4:8:131072 dflash2:7:4:100000" "1 2 4"
```

The pinned build, rerun from a fresh clone with `dflash2:7:4:100000` and workers 1 and 2, gave 242 tok/s at n=1
and 433 at n=2, against 239 and 404 in the table.

`run_matrix.sh` starts each config through `serve.sh`, replays the trajectories in `bench/traj/` at each worker
count, and prints the throughput versus decode batch curve (`bench/batch_curve.py`). To record new trajectories
from a live agent, run `bench/run_agent.sh` (needs `mini-swe-agent` and `podman`), and summarise a request log
with `bench/agent_stats.py`.

## Layout

| Path | Role |
|------|------|
| `NINFER_PIN` | Pinned NInfer commit, model repo and revision, artifact sha256 |
| `install.sh` | Clone and build NInfer at the pin |
| `fetch_model.sh` | Download the artifact and verify the checksum |
| `serve.sh` | Start the server with a named profile |
| `bench/` | Replay and analysis scripts, recorded trajectories (`traj/`), raw per-config tables (`runs/`) |
| `RESULTS.md` | Measured numbers, exact configurations, limits |
| `NOTICE` | Third-party attribution |

## Licensing and credits

The scripts and documentation in this directory are Apache-2.0, like the rest of the repository.
NInfer (Apache-2.0, Neroued) and the weight artifact are third-party and are fetched at install time, not
redistributed here. The artifact card lists Apache-2.0 for the artifact and for its sources: Qwen3.8-27B,
the unsloth NVFP4 quantization and the z-lab DFlash2 drafter. See [NOTICE](NOTICE).
