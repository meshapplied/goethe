# Install

Three models in the [catalog](../README.md#catalog). Steps depend on engine.

## Ternary MoE (Qwen3.8-Flash-Next ternary, DeepSeek-V4.1-Flash)

Requirements: Linux x86_64, NVIDIA GPU, CUDA toolkit matching PyTorch, Python ≥ 3.10. Optional [SGLang](https://github.com/sgl-project/sglang).

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

JIT CUDA extensions cache under `$TERNARY_BUILD_DIR`, else `~/.cache/ternary-flash/build`.

- [Qwen3.8-Flash-Next](models/qwen38-flash-next.md) (ternary variant; NVFP4 is upstream only)
- [DeepSeek-V4.1-Flash](models/deepseek-v41-flash.md)

## Qwen3.8-27B NVFP4 (NInfer)

Blackwell GPU (sm_120a), 32 GiB VRAM, CUDA 13.x with `nvcc`, `cmake`, `ninja`, C++20, Python 3 with `httpx` and `huggingface_hub`.

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe/nvfp4-kernels
./install.sh && ./fetch_model.sh
```

[docs/models/qwen38-27b-nvfp4.md](models/qwen38-27b-nvfp4.md)

## Development

```bash
pip install -e "./ternary-kernels[dev]"
```
