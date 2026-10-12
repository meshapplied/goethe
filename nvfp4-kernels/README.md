# Qwen3.8-27B NVFP4 (NInfer)

Dense hybrid in NVFP4 with MTP and DFlash2 in one artifact. Recipe card: [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer). Weights: [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) file `qwen3_8_27b_nvfp4.ninfer` (23.7 GB). Pins in [NINFER_PIN](NINFER_PIN).

Engine: [NInfer](https://github.com/Neroued/ninfer) at the commit in `NINFER_PIN`, built unmodified. Hardware: Blackwell sm_120a, 32 GiB VRAM (RTX 5090 or RTX PRO 6000). CUDA 13.x with `nvcc`, `cmake`, `ninja`, C++20.

```bash
./install.sh && ./fetch_model.sh && ./serve.sh
```

| profile | drafter | K | lanes | max context |
|---------|---------|---|-------|-------------|
| `default` | DFlash2 | 7 | 4 | 100000 |
| `long` | DFlash2 | 7 | 1 | 131072 |
| `batch` | MTP | 4 | 8 | 131072 |

`PROFILE=<name> ./serve.sh`. Overrides: `SPEC`, `K`, `MAX_CONCURRENCY`, `MAX_CONTEXT`. fp8 KV, `--lm-head-draft`, `--prefill-chunk 1024`. OpenAI Chat Completions, Responses and Anthropic Messages on port 8080 by default.

DFlash2 does not start at 8 lanes and 131072 context on 32 GiB. Use MTP for 8 lanes.

Throughput (one RTX 5090, replayed agent turns, mean context ~28k, thinking on): see [RESULTS.md](RESULTS.md). Reproduce: `pip install httpx` then `bench/run_matrix.sh "mtp:3:8:131072 mtp:4:8:131072 dflash2:7:4:100000" "1 2 4"`.

Third-party attribution: [NOTICE](NOTICE).
