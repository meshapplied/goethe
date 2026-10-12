---
license: apache-2.0
tags:
  - nvfp4
  - qwen
  - speculative-decoding
  - blackwell
  - ninfer
base_model: neroued/Qwen3.8-27B-nvfp4-NInfer
---

# Qwen3.8-27B NVFP4 serve recipe

This Hub repository holds no weights. It documents pins and scripts for Qwen3.8-27B NVFP4 with MTP or DFlash2 on one RTX 5090 class GPU, using [NInfer](https://github.com/Neroued/ninfer).

| item | value |
|------|-------|
| weights | [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| engine commit | `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` |
| artifact revision | `a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d` |
| sha256 | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| GPU | compute capability 12.0, 32 GiB |

## Run

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe/nvfp4-kernels
./install.sh && ./fetch_model.sh && ./serve.sh
```

Profiles, measured throughput, benchmark protocol and limits: [docs/models/qwen38-27b-recipe.md](https://github.com/meshapplied/Goethe/blob/main/docs/models/qwen38-27b-recipe.md) and [nvfp4-kernels/RESULTS.md](https://github.com/meshapplied/Goethe/blob/main/nvfp4-kernels/RESULTS.md).

## License

Apache-2.0 for Goethe scripts and this card. NInfer, the artifact and upstream Qwen, Unsloth and z-lab components carry their own Apache-2.0 terms; see NOTICE in the repository.
