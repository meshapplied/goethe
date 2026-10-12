# DeepSeek-V4.1-Flash

## Hub

| Field | Value |
|-------|-------|
| Repository | [`meshapplied/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) |
| Kind | Ternary expert pack (not an NVFP4 weight dump; the Hub name is historical) |
| Runtime profile | `deepseek_v41` |

## Upstream base

[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Attention, router, shared experts and MTP stay in the base checkpoint. The pack replaces routed expert matrices only.

## Engine

[SGLang](https://github.com/sgl-project/sglang) with the Goethe `ternary_runtime` patches (`pip install -e ./ternary-kernels`). Expert layout: `w123`, specialized K 5120 and 2304. Pack contract: [pack.md](../pack.md).

## Hardware

NVIDIA GPU with CUDA matching your PyTorch build. Tensor parallel must respect Hadamard block sizes in the pack (see [pack.md](../pack.md)). No published single-GPU VRAM table in this repository.

## Install

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

## Fetch and serve

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-NVFP4")
ternary_runtime.bootstrap()
```

```bash
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 1 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code
```

Example serve template: [`examples/ternary-deepseek.toml`](../../examples/ternary-deepseek.toml).

## Measurement

No decode throughput benchmark is checked in to this repository for this model. Report tok/s with engine version, batch size, context length and sampling settings alongside any number.

## Repository paths

| Path | Role |
|------|------|
| [`ternary-kernels/`](../../ternary-kernels/) | Loader, Hub fetch, CUDA kernels, SGLang hooks |
| [`hub/deepseek-v41-flash/README.md`](../../hub/deepseek-v41-flash/README.md) | Hugging Face model card source |
| [`docs/architecture.md`](../architecture.md) | Runtime design |

## License

Apache-2.0 for Goethe code and pack metadata. The base checkpoint license on Hugging Face applies when you download and run DeepSeek-V4.1-Flash.
