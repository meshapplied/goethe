# Changelog

## Unreleased

- No CUDA toolkit install is needed: engines build run-time kernels with the toolkit bundled in the py-env (nvcc, crt, nvvm and cccl pinned to 13.0 to match the 13.0 runtime headers). Before each start Goethe adds the `lib64`, unversioned library and `libcuda.so` links that pip wheels leave out. `CUDA_HOME` or `CUDA_PATH` in goethed's environment or a card still wins.
- The installer offers to install the NVIDIA driver when a GPU has none or one older than 580. It asks first, uses `ubuntu-drivers` on Ubuntu or NVIDIA's CUDA repository on Debian and the RHEL family, then loads the driver or asks for a reboot. `GOETHE_INSTALL_DRIVER=1` installs without asking and `GOETHE_INSTALL_DRIVER=0` never installs. It never installs inside containers. A working driver is left alone.
- When the NVIDIA driver is unusable, `goethe doctor`, `goethe load` and the installer name the cause and the fix: no GPU attached to the VM or container, no driver loaded, driver older than 580, no `/dev/nvidia*`, or missing `libcuda.so.1`/`libnvidia-ml.so.1`. The installer reads the driver version from `/proc/driver/nvidia/version` when `nvidia-smi` is absent.
- Volta (7.0) and Turing (7.5) GPUs are recognised and named: `goethe gpu list` marks them UNSUPPORTED with the reason, `goethe doctor` explains it, and placement never uses them.
- `goethe recommend` shows where each fitting model would load (GPUs, VRAM per GPU, context) and its download size, instead of one size column that looked like the VRAM need.

- GPU support covers Ampere (8.0, 8.6), Hopper (9.0) and data-center Blackwell (10.0, 10.3) next to consumer Blackwell (12.0, 12.1), from one architecture table. Machines with mixed GPU architectures work: placement only uses GPUs a model runs on, a tensor-parallel group never mixes architectures, `--gpus` sets that mix them are refused with the reason, and `goethe doctor` lists the architectures. Engines build run-time kernels for their own target (`TORCH_CUDA_ARCH_LIST`, `sm_90a`/`sm_100a` style) with one JIT cache per architecture. Cards record the compute capabilities of their catalog entry. The Marlin weight copy is budgeted only on GPUs that keep one (12.x); `goethe recommend` used to add it on 10.x too.

- Drop standing comments outside function bodies; keep file-top lines and docstrings only.

- `docs/HUB_CARDS.md` records the Hub card contract for the Ternary-Latest ids.

- `goethe-lab serve <model>` starts that model. No name still starts Qwen3.8-27B on NInfer. The other three names run the SGLang command in `examples/`.

- `nvfp4-kernels/image.sh` builds `goethe:pack` from a `pack.sh` tarball. The image does not compile NInfer.

- Flash-Next NVFP4 weight-only (compressed-tensors): SGLang patch in `nvfp4-kernels/patches/` and serve example `examples/nvfp4-weight-only.toml`, with tests.

- `goethe-lab serve` uses `$NVFP4_HOME/bin/ninfer-serve` when that file is executable, and does not build. `nvfp4-kernels/pack.sh` writes that layout with `libcudart.so.13` beside the binary.

- OpenAI and Responses tool names may be 128 characters, the same limit the Anthropic adapter already had. A fit that cannot boot prints one line instead of a traceback.

- `docs/spikes.md` records the 5090 checks: context above capacity, four-lane pressure, cold start, a 4 MiB range download, and a Messages stream.

- Qwen3.8-27B NVFP4 NInfer: agentic table (c1 **255** / c4 **686**) plus cold prefill/long-ctx + VRAM.
- DeepSeek-V4.1 Ternary performance: agentic **194** tok/s (conc=1 TP4) and cold prefill/long-ctx latency tables.
- Flash-Next Ternary performance: agentic **168** tok/s (conc=1 TP1) and cold prefill/long-ctx latency tables.
- Flash-Next NVFP4 performance: agentic **184** tok/s (conc=1) and cold prefill/long-ctx latency tables.
- `goethe` command (`serve`, `list`, `doctor`), standard library only, with an installer. See `docs/cli.md`.
- `goethe-lab connect` runs a coding agent against the server for one command.
- `goethe-lab update --check` compares the installed engine, patch and model file to the pin.
- `goethe-lab doctor` names missing engine libraries and the apt line.
- `goethe-lab bench` replays the bundled agent traces and reports token-weighted decode tok/s, cache hit, time to first token, acceptance, tool-call validity and power.
- `goethe-lab serve` treats a CUDA out-of-memory boot as a failed fit, and lowers context, instead of waiting out the probe.
- The engine-link command was removed. `serve` starts the binary under `NVFP4_HOME`.
- `nvfp4-kernels/`: patched NInfer tile for the down projection and a draft-shortlist patch for the model file, applied by `install.sh` and `fetch_model.sh` (about 5 % faster decode).
- Passive lineup README; serve via [`examples/`](examples/) (ternary + Flash-Next NVFP4).
- SGLang patch: skip ternary for DSpark/NEXTN draft stages and mismatched expert counts.
- Current Hub ids in docs and examples.

## 0.1.0

- `ternary_runtime` and MoE ternary packs on Hugging Face.
- `nvfp4-kernels/`: pinned NInfer, Qwen3.8-27B NVFP4 fetch and benchmark on one RTX 5090.
