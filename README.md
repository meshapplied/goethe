# Goethe

Apache-2.0 inference distribution: pinned packs or weights, engines, serve commands and measurement where we have it. MoE models use SGLang and `ternary_runtime`; Qwen3.8-27B NVFP4 uses NInfer on Blackwell.

## Catalog

| Model | Hugging Face | Engine | Documentation |
|-------|--------------|--------|-----------------|
| Qwen3.8-Flash-Next | Ternary: [`meshapplied/qwen3.8-flash-next-ternary-latest`](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest). NVFP4: upstream quants (see doc). | SGLang + `ternary_runtime` (ternary); vLLM or SGLang (NVFP4) | [docs/models/qwen38-flash-next.md](docs/models/qwen38-flash-next.md) |
| DeepSeek-V4.1-Flash | [`meshapplied/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) (ternary expert pack) | SGLang + `ternary_runtime` | [docs/models/deepseek-v41-flash.md](docs/models/deepseek-v41-flash.md) |
| Qwen3.8-27B NVFP4 | Weights: [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer). Recipe: [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer). | NInfer | [docs/models/qwen38-27b-nvfp4.md](docs/models/qwen38-27b-nvfp4.md) |

Each model page uses the same section order where it applies: Hub, upstream base, engine, hardware, install, serve, measurement, repository paths, license. Flash-Next documents **ternary** and **NVFP4** variants on one page.

## Repository layout

| Path | Role |
|------|------|
| [`docs/models/`](docs/models/) | Per-model reference |
| [`ternary-kernels/`](ternary-kernels/) | Ternary pack loader and CUDA kernels |
| [`nvfp4-kernels/`](nvfp4-kernels/) | NInfer install, fetch, serve and benchmark for Qwen3.8-27B |
| [`docs/`](docs/) | Install, pack format, architecture |
| [`examples/`](examples/) | SGLang config templates |

## Framework install (ternary MoE)

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

Requires Linux, NVIDIA GPU, Python ≥ 3.10, PyTorch with CUDA. Optional SGLang for serve. Details: [docs/install.md](docs/install.md).

Qwen3.8-27B: [docs/models/qwen38-27b-nvfp4.md](docs/models/qwen38-27b-nvfp4.md). Flash-Next NVFP4: same model page, upstream weights.

## License

Apache-2.0 for this repository. Base checkpoints and third-party artifacts keep their licenses on Hugging Face.
