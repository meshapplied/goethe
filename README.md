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

Pack on-disk format: [docs/pack.md](docs/pack.md).

## Layout

| Path | Role |
|------|------|
| [`ternary-kernels/ternary_runtime/`](ternary-kernels/ternary_runtime/) | Loader, Hub fetch, SGLang patches, CUDA kernels |
| [`examples/`](examples/) | Serve config templates |
| [`docs/pack.md`](docs/pack.md) | Expert pack contract |
| [`nvfp4-kernels/`](nvfp4-kernels/) | Optional Qwen3.8-27B NInfer helpers |

## License

Apache-2.0 for this repository. Base model licenses on Hugging Face still apply. The Flash-Next NVFP4 Hub tree is also covered by the NVIDIA Open Model License (`LICENSE` / `NOTICE` on that repo).
