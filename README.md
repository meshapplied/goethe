# Goethe

Apache-2.0: ternary MoE packs for SGLang (`ternary_runtime`), and a pinned NInfer recipe for Qwen3.8-27B NVFP4 on Blackwell.

## Catalog

| Model | Hugging Face | Run from this repo |
|-------|--------------|-------------------|
| Qwen3.8-Flash-Next | Ternary: [`Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest). NVFP4: [`Qwen3.8-Flash-Next-NVFP4-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest). | Hub cards only (not maintained here). |
| DeepSeek-V4.1-Flash | [`DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest) (same pack as the `…-NVFP4` repo name). | `ternary_runtime` below; full serve flags on the Hub card. |
| Qwen3.8-27B NVFP4 | Weights: [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer). Recipe card: [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer). | [`nvfp4-kernels/`](nvfp4-kernels/) |

Pack on-disk format: [docs/pack.md](docs/pack.md).

## Ternary packs (DeepSeek; Flash-Next uses the same runtime)

```bash
git clone https://github.com/meshapplied/goethe.git
cd goethe
pip install -e ./ternary-kernels
```

```python
import ternary_runtime

# DeepSeek (id matches the published Hub card)
ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest")
ternary_runtime.bootstrap()
# launch SGLang with --model-path "$TERNARY_MODEL_DIR"; see the Hub README for flags
```

Flash-Next ternary: `fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")`. Serve and benchmarks for Flash-Next variants live on Hugging Face, not in this tree.

Requires Linux, NVIDIA GPU, Python ≥ 3.10, PyTorch with CUDA, SGLang for serve. JIT build cache: `~/.cache/ternary-flash/build` unless `TERNARY_BUILD_DIR` is set.

## Qwen3.8-27B NVFP4

```bash
cd nvfp4-kernels
./install.sh && ./fetch_model.sh && ./serve.sh
```

Pins, profiles, benchmark protocol and numbers: [nvfp4-kernels/README.md](nvfp4-kernels/README.md) and [nvfp4-kernels/RESULTS.md](nvfp4-kernels/RESULTS.md).

## License

Apache-2.0 for this repository. Base checkpoints and third-party weights keep their licenses on Hugging Face.
