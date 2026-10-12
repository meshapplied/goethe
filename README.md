# Goethe

Apache-2.0 inference distribution for four published model entries: pinned weights or packs, engines, serve commands and measurement where we have it. Two MoE models run through SGLang and `ternary_runtime`; Qwen3.8-27B NVFP4 runs through NInfer on Blackwell with a checked artifact and a separate serve recipe card.

## Catalog

| Model | Hugging Face | Engine | Documentation |
|-------|--------------|--------|-----------------|
| Qwen3.8-Flash-Next | [`meshapplied/qwen3.8-flash-next-ternary-latest`](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) | SGLang + `ternary_runtime` | [docs/models/qwen38-flash-next.md](docs/models/qwen38-flash-next.md) |
| DeepSeek-V4.1-Flash | [`meshapplied/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) | SGLang + `ternary_runtime` | [docs/models/deepseek-v41-flash.md](docs/models/deepseek-v41-flash.md) |
| Qwen3.8-27B NVFP4 artifact | [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | NInfer (download only) | [docs/models/qwen38-27b-artifact.md](docs/models/qwen38-27b-artifact.md) |
| Qwen3.8-27B NVFP4 recipe | [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer) | NInfer (pinned build) | [docs/models/qwen38-27b-recipe.md](docs/models/qwen38-27b-recipe.md) |

Each model page uses the same section order: Hub, upstream base, engine, hardware, install, serve, measurement, repository paths, license. Hugging Face README sources live under [`hub/`](hub/).

## Repository layout

| Path | Role |
|------|------|
| [`docs/models/`](docs/models/) | Per-model reference |
| [`ternary-kernels/`](ternary-kernels/) | Ternary pack loader and CUDA kernels |
| [`nvfp4-kernels/`](nvfp4-kernels/) | NInfer install, fetch, serve and benchmark for Qwen3.8-27B |
| [`hub/`](hub/) | Model card sources for Hugging Face |
| [`docs/`](docs/) | Install, pack format, architecture |
| [`examples/`](examples/) | SGLang config templates |

## Framework install (ternary models)

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

Requires Linux, NVIDIA GPU, Python ≥ 3.10, PyTorch with CUDA. Optional SGLang for serve. Details: [docs/install.md](docs/install.md).

Qwen3.8-27B does not use `ternary_runtime`; follow [docs/models/qwen38-27b-recipe.md](docs/models/qwen38-27b-recipe.md).

## License

Apache-2.0 for this repository. Base checkpoints and third-party artifacts keep their licenses on Hugging Face.
