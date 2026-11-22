# Our models

The NVFP4 models we publish on Hugging Face. Each one is in Goethe's catalog: install and serve it like any other model.

🤗 [View our models](https://goethe.sh/models)

```bash
goethe install meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest --load
```

| Model | What it is | Runs on |
|---|---|---|
| [`meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest) | Qwen3.8 Flash-Next with NVFP4 experts and a built-in speculative drafter | One 96 GB Blackwell GPU |
| [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer) | Qwen3.8-27B in NVFP4 | 32 GB Blackwell. Weights: [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) file `qwen3_8_27b_nvfp4.ninfer` |

`goethe models recommend` shows which of them fit your GPUs. Each model's card on Hugging Face has its full results.

The numbers below were measured on RTX PRO 6000 Blackwell GPUs (96 GB).

## Performance (Flash-Next NVFP4)

Single RTX PRO 6000 96 GB, TP1, NEXTN 3/4 + hot vocab, T=0. Details and percentiles on the Hub card.

### Agentic coding

Replay of 10 SWE-bench Pro / DeepSWE mini-swe-agent conversations × 6 turns, 256 out/turn, warm prefix cache.

| Concurrency | out tok/s | accept | TTFT ms | ITL ms |
|------------:|----------:|-------:|--------:|-------:|
| 1 | **184** | 3.13 | 192 | 4.9 |
| 4 | **334** | 3.13 | 354 | 9.9 |

### Cold prefill and long-context decode

Radix cache off + flush between lengths; random prompts, out=256, concurrency 1. Prefill ≈ `prompt_len / mean_TTFT`.

| prompt | decode tok/s | prefill tok/s | TTFT mean / p50 / p90 / p99 (ms) | ITL mean / p50 / p90 / p99 (ms) |
|------:|-------------:|--------------:|--------------------------------:|-------------------------------:|
| 1024 | 152 | ~9.4k | 109 / 98 / 139 / 154 | 5.8 / 3.7 / 14.4 / 14.9 |
| 4096 | 133 | ~19k | 214 / 203 / 319 / 320 | 6.1 / 3.8 / 14.4 / 15.0 |
| 16384 | 89 | ~21k | 769 / 763 / 1230 / 1474 | 5.9 / 3.8 / 14.4 / 15.0 |
| 32768 | 69 | ~26k | 1257 / 1141 / 2123 / 2458 | 5.8 / 3.8 / 14.3 / 15.1 |

Idle serve VRAM ≈ **89.3 / 95.6 GB**.

## License

Base model licenses on Hugging Face still apply. The Flash-Next NVFP4 model is also covered by the NVIDIA Open Model License (`LICENSE` and `NOTICE` in its repository).
