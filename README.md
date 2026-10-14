# Goethe

Goethe is a runtime and set of model packs for fast coding and research inference on fewer GPUs. The lineup:

- **DeepSeek-V4.1-Flash Ternary** — complex coding agents and long research loops ([`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest))
- **Qwen3.8-Flash-Next NVFP4** — everyday coding and STEM chat on one 96 GB GPU ([`meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest))
- **Qwen3.8-27B NVFP4 NInfer** — personal agents ([`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer))
- **Qwen3.8-Flash-Next Ternary** — fast retrieval and data analysis ([`meshapplied/Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest))

Ternary packs are expert overlays on a public base checkpoint. NVFP4 packs are full ready-to-serve trees. For ternary models, `ternary_runtime.fetch()` downloads the pack and its base.

## Install

```bash
git clone https://github.com/meshapplied/goethe
cd goethe
pip install -e ./ternary-kernels
python -c "import ternary_runtime; print(ternary_runtime.__version__)"
```

Linux, a CUDA GPU, Python ≥ 3.10, and a matching PyTorch build are required. JIT kernels cache under `~/.cache/ternary-flash/build` (or `$TERNARY_BUILD_DIR`).

## The goethe command

```bash
curl -fsSL https://raw.githubusercontent.com/meshapplied/Goethe/main/install.sh | bash
goethe doctor    # check this machine
goethe serve     # install what is missing and start the Qwen3.8-27B NVFP4 server
```

See [docs/cli.md](docs/cli.md). It covers the NInfer models for now; the ternary packs below are still used from Python.

## Fetch a ternary pack

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest")
# or: meshapplied/Qwen3.8-Flash-Next-Ternary-Latest
ternary_runtime.bootstrap()
```

This sets `TERNARY_PACK_DIR`, `TERNARY_MODEL_DIR`, and `TERNARY_PROFILE`.

## Run

Serve commands and knobs live in-repo — open the matching file and follow it:

| Model | Start here |
|-------|------------|
| Qwen Flash-Next ternary | [`examples/ternary-flash-next.toml`](examples/ternary-flash-next.toml) |
| DeepSeek ternary | [`examples/ternary-deepseek.toml`](examples/ternary-deepseek.toml) |
| Qwen Flash-Next NVFP4 | [`examples/nvfp4-qwen.toml`](examples/nvfp4-qwen.toml) |

Pack on-disk format: [docs/pack.md](docs/pack.md). Full Flash-Next NVFP4 serve flags and benches: Hub card [`Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest).

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

## Layout

| Path | Role |
|------|------|
| [`ternary-kernels/ternary_runtime/`](ternary-kernels/ternary_runtime/) | Loader, Hub fetch, SGLang patches, CUDA kernels |
| [`examples/`](examples/) | Serve config templates |
| [`docs/pack.md`](docs/pack.md) | Expert pack contract |
| [`nvfp4-kernels/`](nvfp4-kernels/) | Optional Qwen3.8-27B NInfer helpers |

## License

Apache-2.0 for this repository. Base model licenses on Hugging Face still apply. The Flash-Next NVFP4 Hub tree is also covered by the NVIDIA Open Model License (`LICENSE` / `NOTICE` on that repo).
