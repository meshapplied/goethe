# Qwen3.8-Flash-Next

## Hub

| Field | Value |
|-------|-------|
| Repository | [`meshapplied/qwen3.8-flash-next-ternary-latest`](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) |
| Kind | Ternary expert pack (Hadamard-rotated `{-1,0,+1}` routed experts) |
| Runtime profile | `flash_next` |

## Upstream base

[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Attention, router, shared experts and MTP stay in the base checkpoint. The pack replaces routed expert matrices only.

## Engine

[SGLang](https://github.com/sgl-project/sglang) with the Goethe `ternary_runtime` patches (`pip install -e ./ternary-kernels`). Expert layout: `gate_up_down`, specialized K 2560 and 640. Pack contract: [pack.md](../pack.md).

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
m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
```

```bash
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 1 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code
```

Example serve template: [`examples/ternary-flash-next.toml`](../../examples/ternary-flash-next.toml).

## Measurement

No decode throughput benchmark is checked in to this repository for this model. Report tok/s with engine version, batch size, context length and sampling settings alongside any number.

## Repository paths

| Path | Role |
|------|------|
| [`ternary-kernels/`](../../ternary-kernels/) | Loader, Hub fetch, CUDA kernels, SGLang hooks |
| [`hub/qwen38-flash-next-ternary-latest/README.md`](../../hub/qwen38-flash-next-ternary-latest/README.md) | Hugging Face model card source |
| [`docs/architecture.md`](../architecture.md) | Runtime design |

## License

Apache-2.0 for Goethe code and pack metadata. The base checkpoint license on Hugging Face applies when you download and run Qwen3.8-Flash-Next.
