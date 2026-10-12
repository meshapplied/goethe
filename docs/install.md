# Install

Goethe ships four model entries (see [catalog](../README.md#catalog)). Install steps depend on the engine.

## Ternary MoE models (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash)

Requirements:

- Linux x86_64, NVIDIA GPU, CUDA toolkit matching PyTorch
- Python ≥ 3.10
- Optional: [SGLang](https://github.com/sgl-project/sglang) for serve

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

JIT CUDA extensions cache under `$TERNARY_BUILD_DIR`, else `~/.cache/ternary-flash/build`. Set `TORCH_CUDA_ARCH_LIST` if autodetection is wrong.

Fetch and serve: [qwen38-flash-next.md](models/qwen38-flash-next.md), [deepseek-v41-flash.md](models/deepseek-v41-flash.md).

## Qwen3.8-27B NVFP4 (NInfer)

Requirements: Blackwell GPU (sm_120a), 32 GiB VRAM, CUDA 13.x with `nvcc`, `cmake`, `ninja`, C++20, Python 3 with `httpx` and `huggingface_hub` for fetch and benchmark.

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe/nvfp4-kernels
./install.sh && ./fetch_model.sh
```

Serve and measurement: [qwen38-27b-recipe.md](models/qwen38-27b-recipe.md). Artifact pins: [qwen38-27b-artifact.md](models/qwen38-27b-artifact.md).

## Development

```bash
pip install -e "./ternary-kernels[dev]"
```
