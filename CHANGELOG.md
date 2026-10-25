# Changelog

## Unreleased

- GPUs are named as `goethe gpu` lists them: `goethe load --model <id> --gpu gpu3`, or `--gpu gpu2 gpu3` to split a model across both (the number of GPUs is the tensor-parallel size), also on `install` and `config`. A count the model cannot be split over is refused with the counts that work, and when the named GPUs are too few the error names a `--gpu` list that fits. The older `--gpus 2,3` still works.
- Memory budgets are accurate for long contexts and hybrid models. The KV cache holds one request at the full context, or every parallel request at a quarter of it, instead of four full contexts. Hybrid models (Qwen3.5, Qwen3-Next and other linear-attention or Mamba models) budget their linear-attention state for two slots per parallel request and pass that count to SGLang (`--max-mamba-cache-size`); before, SGLang gave about half of the KV budget to that state on its own. Qwen3.8-27B-NVFP4 at 131,072 tokens and 8 requests now needs 44.9 GB per GPU instead of 58.3 GB, and SGLang used 44.84 GB with a KV cache of 299,167 tokens.

- Third-party notices: `THIRD_PARTY_NOTICES.md` lists the main components (SGLang, shipped in a modified build, PyTorch, FlashInfer, Triton, Transformers, the NVIDIA CUDA libraries, uv, CPython, nginx, SQLite) and every Python package and Rust crate with its version and license, with notes on the NVIDIA, LGPL and MPL components. `THIRD_PARTY_LICENSES.txt` carries the license texts of the Rust crates compiled into goethe and goethed. Both ship in the signed package with Goethe's `LICENSE`, and the installer places them in `$GOETHE_HOME`. `engine/packaging/third_party.py` regenerates them, and CI checks that they match `Cargo.lock` and `requirements.lock`.
- `goethe uninstall` now deletes the models Goethe installed, with the same rules as `goethe remove`: downloads and managed copies are deleted, models registered from a folder elsewhere or the Hugging Face cache lose only Goethe's links, and a folder of your own files inside the models directory is kept and named. The plan lists the models and the space they free, also with `--yes`. `--keep-models` keeps them; the old `--purge-models` still works.

## 0.1.4 (2026-10-25)

- The installer shows what it is doing: numbered steps (checking the machine, verifying the signed package, setting up the Python environment, installing goethe and goethed, setting up the gateway), a result line for each check (driver and GPUs, compiler, signature, version, nginx, TLS certificate), progress bars for downloads, and uv's live download and install progress for the Python environment. The banner now also shows in normal-width terminals, in a compact form, and as one line in narrow ones.

## 0.1.3 (2026-10-25)

- The installer handles far more machines without stopping. It installs missing `curl`, `zstd`, `openssl`, CA certificates, a C++ compiler and nginx with apt, dnf, yum or zypper (asking first unless root; `GOETHE_INSTALL_DEPS=1`/`0`). For the GPU it tells apart: no GPU, no driver, a driver installed but not loaded (it runs `modprobe`, and names Secure Boot when that blocks it), a driver older than 580, NVIDIA's proprietary module on a Blackwell GPU (it switches to the open module), a GPU that did not start (it explains the power cycle), some GPUs not starting, and GPUs older than compute capability 8.0. A driver installed from NVIDIA's `.run` file is removed before the packaged driver replaces it. GPU problems warn and the install continues; `GOETHE_REQUIRE_GPU=1` still stops. It checks disk space before downloading, says what it does when consent was given in advance, and survives an unset `HOME`.
- Right after a release, the package host's cache can serve old and new files side by side for up to 10 minutes, which made installs stop with `checksum mismatch`. The installer now refreshes the signed checksums and retries every 30 s for up to 12 minutes; `goethe update` says to run it again in a few minutes and changes nothing.
- Kernels built on a model's first load use the system C++ compiler, which Goethe now finds and passes to engines (`CXX`, `CC`, `CUDAHOSTCXX`). A gcc newer than the bundled nvcc 13.0 supports (above 15) gets `-allow-unsupported-compiler`. `goethe doctor` has a compiler check, and a model that fails because of the compiler, missing GPU kernels or an old driver says so in its failure report.
- `goethe doctor` and `goethe start` recognise a GPU the driver did not start: the proprietary module on Blackwell, or a card that failed to initialise. `goethe start` explains such problems before the first step instead of failing on them. A GPU that does not answer no longer breaks GPU listing and placement for the others; it is skipped and shown in `goethe doctor`.
- Ada GPUs (compute capability 8.9: RTX 4090, RTX 6000 Ada, L40S, L4) are supported, on the same path as Ampere.

## 0.1.2 (2026-10-25)

- The installer no longer stops with `No: unbound variable` when `nvidia-smi` is installed but finds no usable GPU ("No devices were found", for example after the GPU crashed). Only output shaped like a driver version is used. Otherwise the version comes from the loaded kernel module (`/proc/driver/nvidia/version`), and with neither the machine is treated as having no driver.

## 0.1.1 (2026-10-25)

- The API accepts the OpenAI, Anthropic and LiteLLM schemas at the same time, at the same address and with the same keys. New routes: `/v1/messages` and `/v1/messages/count_tokens` (Anthropic Messages, with Anthropic-shaped errors), `/v1/responses` and `/v1/rerank`; every route also without `/v1`, plus `/model/info`, `/model_group/info`, `/health/liveliness` and `/health/readiness` as a LiteLLM proxy serves them. Keys are accepted from `Authorization: Bearer`, `x-litellm-api-key`, `x-api-key` and `api-key`. Usage metering and `goethe key` counts cover Anthropic and Responses replies. `goethe api` names the accepted schemas. `/metrics` stays private.
- `goethe api` shows the API endpoint, the models running, the API keys in full and one curl command (with `<model_name>` and `<your_bearer_token>`) to query it. New keys are stored in `state.db` (owner-only) so they can be listed again; keys created before show only their first characters. In `goethe start` each screen starts clean, step 1 offers Install a new model, Serve an existing model and Run your own model, and step 3 lets you pick an existing key (listed by name and id) or name a new one (`[dev/my-app-name]`, default `dev`). A fresh install in a wide enough terminal opens with an "Own your inference -> Goethe" banner in the same light blue.
- New `goethe start`: a guided first run that starts Goethe (start at boot included), then takes you through three steps (install a model, serve it through the API, configure API access), with an advanced track for each part on its own; without a terminal it prints the steps as commands. New `goethe api` shows the API address, how to authenticate, the API keys and the loaded models; `goethe status` and `goethe configure api` point to it, and `goethe get api` still works. `goethe install --model <id> --hf-token` asks for a Hugging Face token for gated or private models, checks it and uses it for that download only, never storing it. The installer no longer starts goethed or turns on start at boot on a fresh install (an upgrade over a running goethed still applies the new binaries and keeps engines serving) and ends with `goethe start`.
- Every value you type is now a flag: `goethe install --model <id>`, `load --model`, `unload --model`, `remove --model`, `config --model`, `logs --model`, `scan --dir`, `key --create <name>` / `--revoke <id>`, `allowlist --add` / `--remove`, `alias --set <a> --model <id>` / `--remove`, `gpu --allow gpuN` / `--deny gpuN`, `gateway --bind`, and `registry validate --source`, `entry --repo`, `add --repo`. Plain `goethe key`, `allowlist`, `alias`, `gpu` and `gateway` list or show. New `goethe status` says whether Goethe is running, with the API address and the key names and ids; `goethe up` prints the same summary. `goethe domain` explains how to reach the API by a domain name (a DNS A record to the API's public or private address; Goethe no longer runs certbot) and `goethe domain --check <domain>` verifies it. `goethe key --help` shows the options for creating a key. Output is more spacious, with tables and indented commands marked with an arrow. `goethe recommend` and `goethe models` show what the allowed GPUs can run when empty (their total VRAM, per GPU) and group models by the VRAM they take; models that cannot be downloaded (manual or gated) are no longer listed. GPU tables show the PCIe capability (`PCIe 5 x16`) instead of the current/maximum link generation. In menus such as `goethe configure api` the selected row is marked with a light blue dot and shown in light blue (256-color 117, the same on every theme). Command examples use `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Flags that name things take several at once, for example `goethe gpu --allow gpu0 gpu1`, `goethe key --revoke <id> <id>`, `goethe allowlist --add <cidr> <cidr>`, `goethe alias --remove`, `goethe unload --model` and `goethe remove --model`. A value typed without its flag (`goethe unload <id>`) or in the old verb form (`goethe gpu allow gpu0`, `goethe key revoke <id>`) prints the corrected command instead of a bare usage error.
- No CUDA toolkit install is needed: engines build run-time kernels with the toolkit bundled in the py-env (nvcc, crt, nvvm and cccl pinned to 13.0 to match the 13.0 runtime headers). Before each start Goethe adds the `lib64`, unversioned library and `libcuda.so` links that pip wheels leave out. `CUDA_HOME` or `CUDA_PATH` in goethed's environment or a card still wins.
- The installer offers to install the NVIDIA driver when a GPU has none or one older than 580. It asks first, uses `ubuntu-drivers` on Ubuntu or NVIDIA's CUDA repository on Debian and the RHEL family, then loads the driver or asks for a reboot. `GOETHE_INSTALL_DRIVER=1` installs without asking and `GOETHE_INSTALL_DRIVER=0` never installs. It never installs inside containers. A working driver is left alone.
- When the NVIDIA driver is unusable, `goethe doctor`, `goethe load` and the installer name the cause and the fix: no GPU attached to the VM or container, no driver loaded, driver older than 580, no `/dev/nvidia*`, or missing `libcuda.so.1`/`libnvidia-ml.so.1`. The installer reads the driver version from `/proc/driver/nvidia/version` when `nvidia-smi` is absent.
- Volta (7.0) and Turing (7.5) GPUs are recognised and named: `goethe gpu` marks them UNSUPPORTED with the reason, `goethe doctor` explains it, and placement never uses them.
- `goethe recommend` shows where each fitting model would load (GPUs, VRAM per GPU, context) and its download size, instead of one size column that looked like the VRAM need.

- GPU support covers Ampere (8.0, 8.6), Hopper (9.0) and data-center Blackwell (10.0, 10.3) next to consumer Blackwell (12.0, 12.1), from one architecture table. Machines with mixed GPU architectures work: placement only uses GPUs a model runs on, a tensor-parallel group never mixes architectures, `--gpus` sets that mix them are refused with the reason, and `goethe doctor` lists the architectures. Engines build run-time kernels for their own target (`TORCH_CUDA_ARCH_LIST`, `sm_90a`/`sm_100a` style) with one JIT cache per architecture. Cards record the compute capabilities of their catalog entry. The Marlin weight copy is budgeted only on GPUs that keep one (12.x); `goethe recommend` used to add it on 10.x too.

## Earlier, unversioned

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
