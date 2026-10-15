# Goethe

🤗 [View our models](https://meshapplied.com/models)

One command to start the models in this repo on your GPU. The `goethe` package is 1,672 lines of Python, tests not included. Those lines are not an engine. They exec NInfer or SGLang.

```bash
curl -fsSL https://raw.githubusercontent.com/meshapplied/Goethe/main/install.sh | bash
goethe doctor
goethe serve
```

`goethe serve` with no name starts Qwen3.8-27B on NInfer, at `127.0.0.1:8080`. It builds NInfer if the binary is missing and downloads the weights if the file is missing. Run `goethe doctor` first. It tells you if this GPU can do that.

A name starts a different model. `goethe serve qwen-flash-next-ternary` and `goethe serve deepseek-ternary` download a pack and the public base checkpoint, then run the SGLang command in `examples/`. `goethe serve qwen-flash-next-nvfp4` runs the Flash-Next NVFP4 command, and only if `NVFP4_MODEL_DIR` already points at the tree. Goethe does not install SGLang. If `python -m sglang.launch_server` is missing, serve stops and says so.

If the 27B build dies on a CUDA header mismatch, `goethe doctor` names the fix. The rest of the commands are in [docs/cli.md](docs/cli.md).

## Models

`goethe list` prints these ids and whether this GPU is in the tested set.

- **qwen3-8-27b**, Qwen3.8-27B NVFP4. NInfer. Measured on an RTX 5090. The GPU needs 30 GiB. [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer)
- **qwen-flash-next-ternary**, Qwen3.8 Flash-Next ternary. SGLang, two GPUs. Serve downloads the pack and the base. [`meshapplied/Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest)
- **deepseek-ternary**, DeepSeek-V4.1-Flash ternary. SGLang, four GPUs. Same download. [`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest)
- **qwen-flash-next-nvfp4**, Qwen3.8 Flash-Next NVFP4. SGLang, one 96 GB GPU. The tree has to already be on disk. [`meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest)

A ternary pack is the expert weights laid over a public base checkpoint. An NVFP4 pack is the whole model.

## Ternary runtime

`goethe serve` calls this for the two ternary ids. Install it yourself when you want `fetch` from Python.

```bash
git clone https://github.com/meshapplied/goethe
cd goethe
pip install -e ./ternary-kernels
python -c "import ternary_runtime; print(ternary_runtime.__version__)"
```

Linux, a CUDA GPU, Python ≥ 3.10, and a PyTorch build that matches the GPU. The JIT cache is `~/.cache/ternary-flash/build`, or `$TERNARY_BUILD_DIR` if you set it.

## Fetch a ternary pack

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest")
# or: meshapplied/Qwen3.8-Flash-Next-Ternary-Latest
ternary_runtime.bootstrap()
```

That sets `TERNARY_PACK_DIR`, `TERNARY_MODEL_DIR`, and `TERNARY_PROFILE` in the process. `bootstrap()` applies the SGLang patches.

## SGLang flags

`goethe serve <id>` reads the matching file and execs it. The flags are here if you want to see them, or change them.

| Model | Flags |
|-------|-------|
| Qwen Flash-Next ternary | [`examples/ternary-flash-next.toml`](examples/ternary-flash-next.toml) |
| DeepSeek ternary | [`examples/ternary-deepseek.toml`](examples/ternary-deepseek.toml) |
| Qwen Flash-Next NVFP4 | [`examples/nvfp4-qwen.toml`](examples/nvfp4-qwen.toml) |

Pack format: [docs/pack.md](docs/pack.md). The Flash-Next NVFP4 numbers and the full flag list are on the Hub card [`Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest).

## Performance (DeepSeek-V4.1 Ternary)

4× RTX PRO 6000 96 GB, **TP=EP=4**, DSpark γ=5, T=0. Details on the Hub card [`DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest).

### Agentic coding

| Concurrency | out tok/s | accept | TTFT ms | ITL ms |
|------------:|----------:|-------:|--------:|-------:|
| 1 | **194** | 3.78 | 425 | 3.5 |
| 4 | **309** | 4.08 | 474 | 9.8 |

### Cold prefill and long-context decode

| prompt | decode tok/s | prefill tok/s | TTFT mean / p50 / p90 / p99 (ms) | ITL mean / p50 / p90 / p99 (ms) |
|------:|-------------:|--------------:|--------------------------------:|-------------------------------:|
| 1024 | 75 | ~0.9k | 1080 / 1085 / 1982 / 2828 | 5.8 / 4.1 / 16.0 / 16.6 |
| 4096 | 99 | ~6.5k | 629 / 600 / 984 / 1000 | 5.7 / 4.1 / 8.5 / 16.8 |
| 16384 | 27 | ~3.7k | 4376 / 2203 / 8701 / 20339 | 5.8 / 4.1 / 8.6 / 16.6 |
| 32768 | 32 | ~9.1k | 3617 / 3301 / 6158 / 7080 | 5.5 / 4.1 / 8.3 / 16.6 |

Idle serve VRAM ≈ **86.1 / 95.6 GB** per rank (peak).

## Performance (Flash-Next Ternary)

Single RTX PRO 6000 96 GB, **TP1**, NEXTN 3/4 + hot vocab, T=0. Details on the Hub card [`Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest).

### Agentic coding

| Concurrency | out tok/s | accept | TTFT ms | ITL ms |
|------------:|----------:|-------:|--------:|-------:|
| 1 | **168** | 2.99 | 338 | 5.2 |
| 4 | **265** | 3.03 | 451 | 13.3 |

### Cold prefill and long-context decode

| prompt | decode tok/s | prefill tok/s | TTFT mean / p50 / p90 / p99 (ms) | ITL mean / p50 / p90 / p99 (ms) |
|------:|-------------:|--------------:|--------------------------------:|-------------------------------:|
| 1024 | 135 | ~4.2k | 247 / 179 / 448 / 482 | 5.7 / 4.1 / 8.3 / 16.6 |
| 4096 | 114 | ~9.2k | 444 / 402 / 701 / 820 | 5.7 / 4.2 / 8.4 / 16.8 |
| 16384 | 61 | ~11k | 1490 / 1524 / 2322 / 2521 | 5.9 / 4.2 / 8.4 / 16.9 |
| 32768 | 43 | ~13k | 2517 / 2267 / 4240 / 4861 | 5.7 / 4.2 / 8.4 / 17.0 |

Idle serve VRAM ≈ **85.2 / 95.6 GB**.

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

## Performance (Qwen3.8-27B NVFP4 NInfer)

32 GiB Blackwell, DFlash2 K=7. Details on the Hub card [`Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer).

### Agentic coding

| Concurrency | out tok/s | accept | TTFT ms | ITL ms |
|------------:|----------:|-------:|--------:|-------:|
| 1 | **255** | 4.40 | 1438 | 4.3 |
| 4 | **686** | 4.35 | 2421 | 5.4 |

### Cold prefill and long-context decode

| prompt | decode tok/s | prefill tok/s | TTFT mean / p50 / p90 / p99 (ms) | ITL mean / p50 / p90 / p99 (ms) |
|------:|-------------:|--------------:|--------------------------------:|-------------------------------:|
| 1024 | 211 | ~16k | 65 / 56 / 105 / 116 | 17.1 / 17.1 / 17.2 / 17.3 |
| 4096 | 147 | ~16k | 254 / 248 / 394 / 413 | 17.2 / 17.2 / 17.3 / 17.4 |
| 16384 | 81 | ~16k | 998 / 1010 / 1606 / 1777 | 17.4 / 17.4 / 17.7 / 17.8 |
| 32768 | 56 | ~18k | 1778 / 1586 / 3068 / 3639 | 17.6 / 17.6 / 18.0 / 18.2 |

Idle NInfer VRAM ≈ **24.7–27.0 / 32 GiB**.

## Layout

| Path | What it is |
|------|------------|
| [`nvfp4-kernels/`](nvfp4-kernels/) | Build NInfer, download the 27B file, start that server |
| [`examples/`](examples/) | SGLang flags for the other three models |
| [`ternary-kernels/ternary_runtime/`](ternary-kernels/ternary_runtime/) | Pack download, SGLang patches, CUDA kernels |
| [`docs/pack.md`](docs/pack.md) | How a ternary pack is laid out on disk |
| [`docs/HUB_CARDS.md`](docs/HUB_CARDS.md) | Canonical Hub ids and card text contract |

## License

Apache-2.0 for this repository. Base model licenses on Hugging Face still apply. The Flash-Next NVFP4 Hub tree is also covered by the NVIDIA Open Model License (`LICENSE` / `NOTICE` on that repo).
