# Qwen3.8-27B NVFP4 artifact

## Hub

| Field | Value |
|-------|-------|
| Repository | [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Kind | Prebuilt NInfer weight artifact (23.7 GB file `qwen3_8_27b_nvfp4.ninfer`) |
| Publisher | Neroued (third-party; Goethe pins a revision and checksum) |

## Upstream base

Qwen3.8-27B dense hybrid (Gated DeltaNet), NVFP4-quantized weights plus bundled DFlash2 drafter tensors. Provenance on the artifact card: Qwen, Unsloth NVFP4 quant, z-lab DFlash2.

## Engine

[NInfer](https://github.com/Neroued/ninfer). Goethe does not build or patch this artifact; it downloads the pinned file and verifies sha256. Serve and benchmark: [qwen38-27b-recipe.md](qwen38-27b-recipe.md).

## Hardware

NVIDIA Blackwell, compute capability 12.0 (RTX 5090 or RTX PRO 6000), 32 GiB VRAM for the measured profiles. NInfer builds for sm_120a only.

## Install

Use the recipe tree; `fetch_model.sh` downloads this artifact:

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe/nvfp4-kernels
./install.sh && ./fetch_model.sh
```

Pinned revision and sha256: [`nvfp4-kernels/NINFER_PIN`](../../nvfp4-kernels/NINFER_PIN).

## Serve

Not standalone. Start the server through [qwen38-27b-recipe.md](qwen38-27b-recipe.md) after `fetch_model.sh`.

## Measurement

Decode throughput for this artifact under Goethe pins is reported in [qwen38-27b-recipe.md](qwen38-27b-recipe.md) and [`nvfp4-kernels/RESULTS.md`](../../nvfp4-kernels/RESULTS.md). No separate benchmark for the raw file without the pinned engine.

## Repository paths

| Path | Role |
|------|------|
| [`nvfp4-kernels/NINFER_PIN`](../../nvfp4-kernels/NINFER_PIN) | Model repo, revision, file name, sha256 |
| [`nvfp4-kernels/fetch_model.sh`](../../nvfp4-kernels/fetch_model.sh) | Download and verify |
| [`nvfp4-kernels/NOTICE`](../../nvfp4-kernels/NOTICE) | Third-party attribution |

## License

Apache-2.0 on the artifact per its Hugging Face card. Qwen, Unsloth, z-lab and Neroued licenses apply to upstream components.
