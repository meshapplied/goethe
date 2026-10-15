# Changelog

## Unreleased

- Port remaining PR #1 bits: `fetch()` rewrites legacy Hub ids to Ternary-Latest, and `docs/HUB_CARDS.md` records the card contract.

- `goethe serve <model>` starts that model. No name still starts Qwen3.8-27B on NInfer. The other three names run the SGLang command in `examples/`.

- `nvfp4-kernels/image.sh` builds `goethe:pack` from a `pack.sh` tarball. The image does not compile NInfer.

- Flash-Next NVFP4 weight-only (compressed-tensors): SGLang patch in `nvfp4-kernels/patches/` and serve example `examples/nvfp4-weight-only.toml`, with tests.

- `goethe serve` uses `$NVFP4_HOME/bin/ninfer-serve` when that file is executable, and does not build. `nvfp4-kernels/pack.sh` writes that layout with `libcudart.so.13` beside the binary.

- OpenAI and Responses tool names may be 128 characters, the same limit the Anthropic adapter already had. A fit that cannot boot prints one line instead of a traceback.

- `docs/spikes.md` records the 5090 checks: context above capacity, four-lane pressure, cold start, a 4 MiB range download, and a Messages stream.

- Qwen3.8-27B NVFP4 NInfer: agentic table (c1 **255** / c4 **686**) plus cold prefill/long-ctx + VRAM.
- DeepSeek-V4.1 Ternary performance: agentic **194** tok/s (conc=1 TP4) and cold prefill/long-ctx latency tables.
- Flash-Next Ternary performance: agentic **168** tok/s (conc=1 TP1) and cold prefill/long-ctx latency tables.
- Flash-Next NVFP4 performance: agentic **184** tok/s (conc=1) and cold prefill/long-ctx latency tables.
- `goethe` command (`serve`, `list`, `doctor`), standard library only, with an installer. See `docs/cli.md`.
- `goethe connect` runs a coding agent against the server for one command.
- `goethe update --check` compares the installed engine, patch and model file to the pin.
- `goethe doctor` names missing engine libraries and the apt line.
- `goethe bench` replays the bundled agent traces and reports token-weighted decode tok/s, cache hit, time to first token, acceptance, tool-call validity and power.
- `goethe serve` treats a CUDA out-of-memory boot as a failed fit, and lowers context, instead of waiting out the probe.
- The engine-link command was removed. `serve` starts the binary under `NVFP4_HOME`.
- `nvfp4-kernels/`: patched NInfer tile for the down projection and a draft-shortlist patch for the model file, applied by `install.sh` and `fetch_model.sh` (about 5 % faster decode).
- Passive lineup README; serve via [`examples/`](examples/) (ternary + Flash-Next NVFP4).
- SGLang patch: skip ternary for DSpark/NEXTN draft stages and mismatched expert counts.
- Current Hub ids in docs and examples.

## 0.1.0

- `ternary_runtime` and MoE ternary packs on Hugging Face.
- `nvfp4-kernels/`: pinned NInfer, Qwen3.8-27B NVFP4 fetch and benchmark on one RTX 5090.
