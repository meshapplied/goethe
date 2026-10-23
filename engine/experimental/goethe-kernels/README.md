# goethe-kernels (archived, not built, not gated, not packaged)

Marlin NVFP4 routing for compressed-tensors linear layers was promoted to the product (`engine/sglang/goethe-marlin.patch`, default-on via `goethe_marlin.py`). The custom kernel package and old hook below stay archived.

- `kernels/`: `goethe_kernels` torch extension (`fused_nvfp4_gemv` with `.scaled` overload, `custom_moe_dispatch`), `build.sh <sm>`, `setup.py`. All six SM packs compile; sm120 runs.
- `sglang/goethe-hook.patch`, `apply_hook.sh`: loader hook plus the `--enable-goethe-fused-gemv` flag. The current patch implements Marlin NVFP4 W4A16 routing (not the custom gemv): SM<100 uses Marlin for every M, SM120 uses Marlin for M<=128 (env `GOETHE_MARLIN_MAX_ROWS`) and the stock W4A4 path above, `GOETHE_FORCE_PATH=w4a16` emulates SM86 on SM120.
- `tests/`: ops, hook, layer, e2e model and arch-compile gates.

Restore: move back to `engine/kernels`, `engine/sglang` and `engine/tests`, apply the patch to the fork with `apply_hook.sh`.

Measured on RTX PRO 6000 (SM120), 17408x5120 NVFP4 layer, microseconds:

| M | Marlin | stock W4A4 | custom FMA gemv |
|---|---|---|---|
| 1 | 36 | 193 | 36 |
| 8 | 36 | 172 | 148 |
| 32 | 42 | 173 | 2514 |
| 128 | 93 | 173 | - |
| 256 | 168 | 163 | - |
| 1024 | 633 | 202 | - |

Marlin wins up to M=128 on the three model shapes (17408x5120, 5120x17408, 34816x5120); stock wins from M=256.

End to end, Qwen3.8-27B NVFP4 NInfer, tok/s at concurrency 1 / 4: stock 28.5 / 16.8, Marlin routing 31.1 / 32.2, forced Marlin 31.1 / 32.1. Dual layout costs about 8.1 GB of extra weights at SM120.

SM86 findings: compressed-tensors MoE NVFP4 requires Blackwell and has no Marlin wiring (`compressed_tensors_w4a4_nvfp4_moe.py`); FP8 W8A8 attention falls back to W8A16 Marlin below capability 89.

Not kept: tensor-core and native FP4 kernels (attempts failed or were slower than Marlin), W4A8 and dequant ops (parked in `goethe-home/parked/paths-20261021`).
