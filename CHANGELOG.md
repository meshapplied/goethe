# Changelog

## Unreleased

- `nvfp4-kernels/`: pinned NInfer build, checksummed Qwen3.8-27B NVFP4 download, serve profiles
  (DFlash2 K=7, MTP K=4), agent-replay benchmark and measured results for one RTX 5090.

## 0.1.0

- Public Goethe runtime: `ternary_runtime` under Apache-2.0.
- Plug-and-play Hub helper: `ternary_runtime.fetch()` + `ternary_config.json`.
- Supported packs: Qwen3.8-Flash-Next ternary, DeepSeek-V4.1-Flash ternary overlay.
- CUDA build cache under `~/.cache/ternary-flash` (or `TERNARY_BUILD_DIR`).
- Docs: install, quickstart, architecture, pack format.
