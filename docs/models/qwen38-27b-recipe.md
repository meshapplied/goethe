# Qwen3.8-27B NVFP4 serve recipe

## Hub

| Field | Value |
|-------|-------|
| Repository | [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer) |
| Kind | Model card and pins (no weights in this Hub repo) |
| Weights | [qwen38-27b-artifact.md](qwen38-27b-artifact.md) |

## Upstream base

Same as the NInfer artifact: Qwen3.8-27B NVFP4 hybrid with MTP and DFlash2 tensors in the downloaded file.

## Engine

[NInfer](https://github.com/Neroued/ninfer), commit pinned in [`nvfp4-kernels/NINFER_PIN`](../../nvfp4-kernels/NINFER_PIN). Goethe builds that commit unmodified and starts `ninfer-serve` with fixed profiles.

## Hardware

NVIDIA Blackwell, compute capability 12.0, 32 GiB VRAM. About 26 GB disk for the artifact and a 1.3 GB build tree. Linux, CUDA toolkit 13.x with `nvcc`, `cmake` 3.28+, `ninja`, C++20.

## Install

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe/nvfp4-kernels
./install.sh
./fetch_model.sh
```

## Serve

```bash
./serve.sh
```

| profile | drafter | draft tokens | lanes | max context | status |
|---------|---------|--------------|-------|-------------|--------|
| `default` | DFlash2 | 7 | 4 | 100000 | measured |
| `long` | DFlash2 | 7 | 1 | 131072 | boots, not benchmarked |
| `batch` | MTP | 4 | 8 | 131072 | measured |

`PROFILE=<name> ./serve.sh`. Overrides: `SPEC`, `K`, `MAX_CONCURRENCY`, `MAX_CONTEXT`. HTTP: OpenAI Chat Completions, Responses, Anthropic Messages on `127.0.0.1:8080` by default.

DFlash2 does not start at 8 lanes and 131072 context on 32 GiB (engine reservation 14.45 GB). MTP is required for 8 lanes.

## Measurement

Aggregate decode tok/s by actual decode batch `n`, replayed mini-swe-agent turns, mean context about 28k, thinking on, fp8 KV cache. One run per config on one RTX 5090; bins with `n >= 4` have fewer than 10 requests.

| config | tok/round | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|-----------|-----|-----|-----|-----|-----|
| MTP, K=3 | 2.86 | 173 | 311 | 428 | 565 | 648 |
| MTP, K=4 | 3.13 | 181 | 339 | 450 | 577 | 697 |
| DFlash2, K=5 | 3.7 | 209 | 393 | 546 | 615 | - |
| DFlash2, K=7 | 4.0-4.15 | 239 | 404 | 569 | 686 | - |
| DFlash2, K=9 | 4.28 | 236 | 396 | - | - | - |
| DFlash2, K=12 | 4.37 | 236 | 405 | 525 | - | - |

Full protocol, hardware, live-agent row and limits: [`nvfp4-kernels/RESULTS.md`](../../nvfp4-kernels/RESULTS.md). Reproduce:

```bash
pip install httpx
bench/run_matrix.sh "mtp:3:8:131072 mtp:4:8:131072 dflash2:7:4:100000" "1 2 4"
```

## Repository paths

| Path | Role |
|------|------|
| [`nvfp4-kernels/`](../../nvfp4-kernels/) | `install.sh`, `fetch_model.sh`, `serve.sh`, benchmark |
| [`hub/qwen38-27b-nvfp4-ninfer/README.md`](../../hub/qwen38-27b-nvfp4-ninfer/README.md) | Hugging Face model card source |
| [`nvfp4-kernels/NOTICE`](../../nvfp4-kernels/NOTICE) | Third-party attribution |

## License

Apache-2.0 for Goethe scripts and documentation. Engine and artifact licenses are listed in NOTICE.
