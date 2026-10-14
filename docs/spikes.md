# Spikes

Measured on greengoblin, RTX 5090, engine `81c8ce093b2c` with the down-projection patch, model file `qwen3_8_27b_nvfp4.ninfer`. One sample unless a count is given.

## Context above capacity

One lane, fp8 KV, DFlash2, 7 draft tokens, prefill chunk 1024.

| max-context | result |
| --- | --- |
| 220032 | `capacity \| KV 220,032 tokens, fp8, auto`, runtime 8.21 GiB, 1.98 GiB free |
| 240000 | FATAL. Reservation wants 9469846528 bytes plus 1073741824 of headroom, 10160766976 available after weights |
| 262144 | FATAL. Reservation wants 10201131264 bytes plus 1073741824 of headroom, 10160766976 available after weights |

The FATAL line and the `capacity | KV` line are the strings `goethe serve` matches. A CUDA out-of-memory line is `cudaMalloc failed: cudaErrorMemoryAllocation: out of memory`. That line is not the reservation FATAL. Fit treats it as a failed boot and lowers context.

## Four lanes

Same 37-request replay, temperature 0, 4 workers, `--max-concurrency 4`. Capacity `KV 117,056`. Wall 51.5 s. Preemptions 0. Replayed tokens 0. The 1-lane run of the same replay was also 0 and 0.

## Cold start

Page cache dropped. `Cached` stayed near 8.7 GiB because `Shmem` was 8.2 GiB. Weights ready 3.8 s at 5.66 GiB/s. The warm boot the same afternoon was 3.7 s at 5.78 GiB/s. Host-context pin 3.6 s, warm 2.1 s. CUDA graphs 1.6 s, warm 1.1 s. Engine ready 11.0 s, warm 8.0 s. Wall until `/health` 12.1 s.

## Download

`fetch_model.sh` is one `curl`. A range request to the pinned Hugging Face file returns 206. One 4 MiB range took 0.86 s (4.65 MiB/s). Four parallel 1 MiB ranges of the same span took 0.681 s (5.87 MiB/s). This is one sample of 4 MiB, not the 23.7 GB file.

## Claude Code and the Messages stream

A streamed `POST /v1/messages` with `max_tokens` 16 returned `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. The body contained thinking deltas. It contained no `event: ping`. `thinking.budget_tokens` of 128 was rejected: the minimum is 1024.

Claude Code 2.1.300, launched with `CLAUDE_CODE_ATTRIBUTION_HEADER=0` in its process environment, wrote a file through the server. Codex's snap reported a write that did not appear on the host filesystem.
