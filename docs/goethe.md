# Goethe

Goethe serves large language models on your own NVIDIA GPUs behind one OpenAI-compatible HTTPS endpoint. You install a model, load it onto one or more GPUs, and clients call it with an API key. Goethe chooses the GPUs, sizes memory, supervises the engines, restarts them when they crash, and keeps working next to other software on the same machine without ever stopping it.

This page describes what is implemented today. The source lives in `engine/`.

## Contents

- [Supported hardware and platforms](#supported-hardware-and-platforms)
- [Architecture](#architecture)
- [Features](#features)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Status and known limitations](#status-and-known-limitations)

## Supported hardware and platforms

| Area | Supported |
|---|---|
| GPUs | NVIDIA compute capability 8.6 (RTX 3090, RTX A6000) and 12.0/12.1 (RTX 5090, RTX PRO 6000 Blackwell, GB10) |
| GPU memory | Discrete VRAM, and unified memory on GB10 |
| Multi-GPU | Tensor parallelism across GPUs (PCIe or NVLink); expert parallelism for MoE models |
| CPU architecture | x86_64 and aarch64 |
| Operating system | Linux with glibc 2.30 or newer |
| NVIDIA driver | 580 or newer (CUDA 13) |
| Install location | Rootless by default; root installs are supported too |
| Environments | Bare metal and VMs with systemd; containers without systemd (Docker, Podman, Kubernetes, LXC) |
| Cloud detection | RunPod, Vast.ai, Lambda, AWS, GCP, Azure, OCI, CoreWeave, Paperspace |

Models: anything the bundled SGLang can serve from a safetensors checkpoint.
- **Architectures:** dense, mixture of experts, hybrid linear attention (for example Qwen3.5 and Qwen3-Next style models), and MLA.
- **Weight formats:** BF16, FP16, FP8, and NVFP4. NVFP4 covers both ModelOpt and compressed-tensors checkpoints, plus the other quantizations SGLang supports.
- **NVFP4 kernels:**
  - On SM86 NVFP4 layers run on Marlin kernels at every batch size.
  - On SM120 they use Marlin up to 128 rows, then the stock FP4 path.

## Architecture

```
clients --HTTPS--> nginx gateway --> mux (inside goethed) --> SGLang engines (loopback only)
                   TLS, allowlist,     API keys, scopes,        one process group per instance,
                   per-IP rate limit   rpm, routing, drain      own API key, own port
                                       usage metering

goethe CLI --Unix socket RPC--> goethed: supervisor, placement, VRAM ledger, watchdog,
                                         restore, hardware profiler, SQLite state
```

### Components

- **`goethe`:** the command line tool. Everything an operator does goes through it.
- **`goethed`:** the daemon. It owns the engines, the router (the "mux") on a loopback port, the nginx gateway, and the state database. It accepts commands from the CLI over a Unix socket.
- **nginx:** the public entry point. It handles TLS, the IP allowlist, per-address rate limits and JSON error pages. It forwards to the mux.
- **SGLang:** the inference engine. Goethe runs a patched SGLang build inside its own Python 3.11 environment. Engines listen on 127.0.0.1 only, each with a random API key that only the mux knows.

### State

Everything lives under `GOETHE_HOME`:

```
GOETHE_HOME/
  bin/goethe, bin/goethed     binaries
  py-env/                     Python environment with SGLang
  models/<org>__<name>/       model files and goethe_card.toml
  nginx/                      gateway config, certificates, ports
  logs/                       goethed, gateway and per-engine logs
  run/                        identities of running engines
  state.db                    SQLite (WAL) state: models, instances, keys, allowlist
  hardware_state.json         last hardware profile
  goethe.env                  package URLs written by the installer
```

## Features

### Installation and upgrades

- One command install from a signed release:
  - It checks curl, zstd, openssl, glibc, the NVIDIA driver and the CPU architecture before doing anything.
  - Every downloaded file is verified against a signed checksum list.
  - The Python environment is built from hash-locked requirements.
- No root needed. The CLI is linked into `/usr/local/bin` or `~/.local/bin`.
- Where state goes (`GOETHE_HOME`), in this order:
  1. the `GOETHE_HOME` variable;
  2. the home the running binary belongs to;
  3. an existing `/workspace/.goethe`;
  4. a writable `/workspace` (RunPod and Vast persistent volumes);
  5. `/var/lib/goethe`;
  6. `~/.goethe`.
- Ports: when a default port is taken by other software, Goethe picks a free one and remembers it.
- Upgrades happen in place:
  - Running the installer again, or `goethe update`, replaces the binaries when the build differs, including two builds of the same version.
  - Loaded models keep serving through the upgrade (see zero-downtime restart below).

### Hardware and environment profile

- `goethe hw profile` reports:
  - GPUs: model, compute capability, VRAM, PCIe generation (current and max), running processes.
  - The link type between every GPU pair.
  - CPU model, cores, threads, NUMA layout and instruction sets.
  - RAM and swap.
  - Disk space for `GOETHE_HOME` and the models folder, including a network filesystem warning.
  - OS, glibc and driver versions.
  - Container type, whether systemd runs, and the cloud provider.
- Cloud environments:
  - On RunPod and Vast.ai Goethe stores state on the persistent volume and prints the public URL for the gateway port.
  - In containers without systemd, autostart uses nohup and cron instead of a unit.

### Models

- Public Hugging Face repos:
  - `goethe install org/model` downloads the repo, checks every weight file against its Hugging Face sha256, and registers it.
  - Interrupted downloads resume per file, and re-installs re-check only files that changed.
  - Duplicate weights and `original/` checkpoints are skipped, and free disk is checked first.
- Gated or private repos:
  - Goethe never logs in to Hugging Face. Download the model yourself and move the folder into `GOETHE_HOME/models`.
  - `goethe scan` finds it and checks that it can be served. It needs a known architecture, complete safetensors, a supported quantization that the GPUs can run, and a size that fits.
  - It then asks `Model found: <name> ... Add to goethe? [y/N]`, or explains why a folder is not supported.
- Local folders: `goethe install ./dir` registers a folder in place, or copies it with `--copy`.
- Model names: commands that take an installed model match its id exactly, ignoring case, or by its folder name (`Qwen__Qwen3-0.6B` means `Qwen/Qwen3-0.6B`). A typo gets a suggestion, for example `did you mean Qwen/Qwen3-0.6B?`.
- Model cards: each model has a `goethe_card.toml` with its default context, concurrency, tensor parallelism, GPUs and extra SGLang flags.
  - Goethe infers the engine flags for you: quantization is detected by SGLang, Qwen3-family models get their reasoning and tool-call parsers, and repos that ship their own Python code need `--trust-remote-code`.
  - Flags that Goethe controls itself (host, port, tensor parallel size, context length, max running requests, memory fraction, API key, served name) are refused in extra flags.
- Removing:
  - `goethe remove` deletes what Goethe downloaded, copied or scanned.
  - For a folder registered in place it only unregisters the model and prints the `rm -rf` command, because Goethe never deletes files it did not create.

### Loading and GPU placement

- Choosing GPUs:
  - `--gpus 2,3` uses exactly those GPUs; `--tp N` picks N GPUs.
  - With neither flag Goethe picks the smallest tensor parallel size that fits. It prefers GPUs without other processes, then the most free VRAM, then the best link between GPUs.
  - If nothing fits it shrinks the context, then the concurrency, then fails with the exact shortfall.
- Memory budget:
  - The budget per GPU is weights plus KV cache plus overhead, with a 1.5 GB safety margin. The weights include the extra Marlin copy of NVFP4 layers on SM120.
  - After a model is RUNNING, Goethe measures its real VRAM use and learns any extra overhead, so the next load of that model is sized correctly. This keeps co-located models and other software from running out of memory.
- Several models can share one GPU. Goethe prints a notice with what already runs there.
- Replicas: loading an already loaded model says so. Use `--replica` to start another copy; requests go to the copy with the fewest in-flight requests.
- Safe starts:
  - Engines on the same GPU start one at a time.
  - A load is ready only when the engine port is owned by this engine's own process and an authenticated probe returns the expected model name. A stranger listening on the port is detected and the engine retries on another port.

### Coexisting with other software

- Goethe never stops, signals or kills a process it did not start.
- If a target GPU runs other processes, the load stops and lists each one with the command to stop it yourself:

  ```
  GPU 2 has unmanaged process PID 3127 (python, user admin) using 10.3 GB VRAM
  Goethe never stops processes it did not start. To stop it yourself: kill 3127
  ```

  - On a terminal it asks whether to load alongside them. In scripts it exits with code 3; add `--yes` to proceed.
  - If they leave too little memory, the error says how much is missing and which processes hold it.
- Every engine carries an identity: boot id, process start time and an instance tag. Goethe checks it before any signal, so a reused process ID can never be hit.

### Lifecycle and reliability

- **Unload:** `goethe unload` drains first. New requests get 503 with Retry-After, and in-flight requests finish (up to `GOETHE_DRAIN_SECS`, default 120 s). `--force` stops at once, and cut streams end with an OpenAI-style error event instead of a dropped connection. GPU memory is released within a few seconds and no zombie processes are left.
- **Crash recovery:** a crashed engine shows as RESTARTING and is restarted after 5, 15, then 45 seconds, up to 3 times in 10 minutes. After that it is stopped and marked FAILED.
- **Startup watchdog:** an engine that writes no log for 180 s while starting, or takes more than 30 minutes in total, is stopped and marked FAILED.
- **Restore:** after a daemon restart or a reboot, loaded models come back in the order they were first loaded, unless you unloaded them.
- **Zero-downtime restart:** `goethe stop --keep-engines`, the installer and `goethe update` let engines and the gateway keep running while goethed is replaced.
  - In-flight requests get up to 30 s to finish.
  - The new goethed adopts the running engines after verifying them.
  - Requests that arrive in the short gap get a JSON 502 from nginx.
- **Daemon crash:** if goethed itself crashes, its engines keep running. The next goethed adopts the ones that still verify and answer instead of reloading them, and stops the rest. The gateway answers 502 until goethed is back.

### Gateway and API

- **OpenAI-compatible routes:**
  - `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`;
  - `GET /v1/models`, `GET /v1/models/{id}`;
  - streaming (SSE) and non-streaming responses;
  - OpenAI-style JSON errors everywhere;
  - `X-Request-Id` passed through or created.
- **Health:**
  - `/health` (no key needed);
  - `/health/ready?model=<id>` (200 only when that model's engine answers);
  - `/health/admin` (admin keys only).
- **Metrics:** Prometheus metrics on the loopback mux port only, including tokens per second and usage counted from streams.
- **Default deny:** nothing reaches the gateway, not even `/health`, until you allow an address range with `goethe allowlist add`. Ranges are validated, and a broken config is rolled back automatically. Every other path answers a JSON 404.
- **API keys:**
  - stored hashed;
  - optional requests-per-minute limit, expiry, and model scope (a scoped key only sees its models; anything else gets 404 `Model Not Found`);
  - admin flag;
  - usage counted per key;
  - revocation takes effect immediately.
- **Rate limits:** nginx limits every client address (20 requests per second, burst 40, 32 open API connections); per-key limits are enforced in the mux. Request bodies are capped at `GOETHE_MAX_BODY_MB` (default 10).
- **Aliases:** `goethe alias set chat org/model` lets clients use a short name. A real model id always wins over an alias spelled the same.
- **TLS and binding:**
  - TLS uses a self-signed certificate by default.
  - The gateway listens on 127.0.0.1. Use `goethe gateway bind 0.0.0.0` (plus an allowlist entry) to serve other machines.

### Security

- Engines listen on loopback only and each one requires its own random API key, so local processes cannot bypass the gateway. The one exception is an engine started with `--tokenizer-worker-num` above 1, which SGLang cannot combine with a key.
- Client keys are never forwarded to engines.
- State files are private (mode 0600, directories 0700).
- Release files are signed, and updates verify the signature before anything is replaced.
- Goethe never sends Hugging Face credentials.

### Operations

- **Logs:** `goethe logs` shows engine, goethed and gateway logs. All logs rotate by size (engines 256 MB, goethed and nginx 100 MB), and `--prune` removes logs of unloaded instances.
- **Doctor:** `goethe doctor` runs 17 checks in parallel, each with a fix hint. They include the driver, hardware, glibc, GOETHE_HOME, state database location, Python environment, RAM, disk, ports, goethed, nginx, gateway, TLS expiry, Hugging Face reachability, container and cloud.
- **Autostart:** `goethe service enable` sets up a systemd unit (system or user) or a cron entry. Each install home has its own; uninstalling one home never touches another.
- **Updates and removal:**
  - `goethe update` verifies and applies signed updates for goethe and the engine.
  - `goethe uninstall` refuses to delete anything that does not look like a Goethe home.

## CLI reference

Tables are aligned, with statuses in color (green OK, RUNNING and ALLOWED; yellow WARN and STARTING; red FAIL and DENIED) when the output is a terminal. Set `NO_COLOR=1` to turn color off; piped output never has color. Most listing commands also take `--json`.

Exit codes: `0` success, `1` error, `2` usage error, `3` confirmation needed but no terminal (use `--yes`).

### Hardware and GPUs

| Command | Description | Arguments |
|---|---|---|
| `goethe hw profile` | Profile GPUs, links, CPU, RAM, disk, OS, container and cloud | `--json` |
| `goethe hw pick-gpus` | Show where a load would be placed right now | `--tp N`, `--model ID` |
| `goethe gpu list` | List GPUs with VRAM, processes and policy | `--json` |
| `goethe gpu deny` | Exclude a GPU from placement | `ID` |
| `goethe gpu allow` | Allow a GPU again | `ID` |

### Models

| Command | Description | Arguments |
|---|---|---|
| `goethe recommend` | List known models that fit this machine | `--json` |
| `goethe install` | Download or register a model; optionally load it | `MODEL_ID` (repo id or local path), `--ctx N`, `--max-conc N`, `--tp N`, `--gpus LIST`, `--load`, `--no-load`, `--no-wait`, `--yes`, `--force`, `--reset`, `--trust-remote-code`, `--no-verify`, `--name ID`, `--copy`, `--sglang-arg FLAG` |
| `goethe scan` | Register model folders you moved into `GOETHE_HOME/models` | `[DIR]`, `--yes`, `--json` |
| `goethe models` (alias `list`) | List installed and available models with fit and load state | `--json`, `--installed`, `--available` |
| `goethe config` | Change a model's default load settings | `MODEL_ID`, `--ctx N`, `--max-conc N`, `--tp N`, `--gpus LIST`, `--sglang-arg FLAG`, `--clear-sglang-args` |
| `goethe remove` | Remove a model (unload, delete Goethe's files or unregister) | `MODEL_ID`, `--yes`, `--force` |

### Serving

| Command | Description | Arguments |
|---|---|---|
| `goethe load` | Load a model and wait until it is RUNNING | `MODEL_ID`, `--gpus LIST`, `--tp N`, `--ep N`, `--ctx N`, `--max-conc N`, `--yes`, `--force`, `--replica`, `--no-wait`, `--json`, `-- EXTRA_SGLANG_FLAGS` |
| `goethe unload` | Drain and stop a model's instances | `[MODEL_ID]`, `--all`, `--force` |
| `goethe ps` | Show instances, GPUs, ports, status and the gateway URL | `--json` |
| `goethe alias set` | Map a short name to an installed model | `ALIAS MODEL` |
| `goethe alias list` | List aliases | none |
| `goethe alias remove` | Remove an alias | `ALIAS` |

### Gateway and access

| Command | Description | Arguments |
|---|---|---|
| `goethe gateway status` | Show bind address, port, URL, state and allowlist size | `--json` |
| `goethe gateway bind` | Set where the gateway listens | `ADDR` (`127.0.0.1`, `0.0.0.0` or an IP of this host) |
| `goethe allowlist add` | Allow an address range | `CIDR`, `--description TEXT` |
| `goethe allowlist remove` | Remove an address range | `CIDR` |
| `goethe allowlist list` | List allowed ranges | none |
| `goethe key create` | Create an API key (printed once) | `--name NAME`, `--raw`, `--rpm N`, `--expires 30d\|DATE`, `--models LIST`, `--admin` |
| `goethe key list` | List keys with limits and usage | `--json` |
| `goethe key revoke` | Revoke a key | `KEY_ID` |
| `goethe domain add` | Get a TLS certificate for a domain and bind publicly (experimental) | `DOMAIN` |

### Daemon and maintenance

| Command | Description | Arguments |
|---|---|---|
| `goethe stop` | Stop goethed (loaded models are restored on next start) | `--keep-engines` |
| `goethe service enable` | Start goethed at boot (systemd, systemd user, or cron) | none |
| `goethe service disable` | Remove this home's autostart | none |
| `goethe service status` | Show autostart state | none |
| `goethe service restore` | Turn restore of loaded models at start on or off | `--on`, `--off` |
| `goethe logs` | Show engine, goethed or gateway logs | `[MODEL_ID]`, `-f`, `-n N`, `--daemon`, `--gateway`, `--prune` |
| `goethe doctor` | Run environment checks with fix hints | `--json`, `--probe-public` |
| `goethe update` | Apply signed updates | `--check`, `--engine`, `--ternary`, `--yes`, `--allow-downgrade` |
| `goethe uninstall` | Remove Goethe from this home | `--purge-models`, `--yes` |
| `goethe --version` | Print version, commit and build date | none |

### Quick start

```bash
curl -fsSL https://meshapplied.com/install.sh | bash
goethe hw profile
goethe install Qwen/Qwen3-8B --load
goethe allowlist add 127.0.0.1/32
KEY=$(goethe key create --name me --raw)
curl -k https://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}]}'
```

`goethe ps` prints the real gateway URL when 8443 is taken.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GOETHE_HOME` | resolved as above | State directory |
| `GOETHE_MUX_PORT` | 30000 | Loopback router port (falls back to 30100-30199 when busy) |
| `GOETHE_PUBLIC_PORT` | 8443 | Gateway port (falls back to 8080, then 18443-18499) |
| `GOETHE_ENGINE_PORTS` | 30001-30099 | Port range for engines |
| `GOETHE_DRAIN_SECS` | 120 | Longest drain on unload |
| `GOETHE_MAX_BODY_MB` | 10 | Largest request body |
| `GOETHE_MARLIN` | on | `0` turns off the Marlin path for NVFP4 |
| `GOETHE_PKG_URL`, `GOETHE_REGISTRY_URL` | official release site | Where installs and updates come from (kept in `goethe.env`) |
| `NGINX_BIN` | system nginx, else one the installer extracts under `GOETHE_HOME` | Use a specific nginx binary |
| `GOETHE_REQUIRE_GPU` | off | Installer refuses hosts without an NVIDIA driver |

Model card keys you may edit by hand in `goethe_card.toml`:
- `context_length`, `max_running_requests`, `tp_size`, `preferred_gpus`;
- `kv_cache_dtype` (only passed to SGLang when not `auto`);
- `extra_sglang_args`, `env`, `pythonpath`;
- `custom_all_reduce` (set to `true` to keep SGLang's custom all-reduce with tensor parallelism).

## Status and known limitations

Verified on real hardware (RTX PRO 6000 Blackwell Max-Q, SM120, x86_64, Ubuntu 24.04, driver 595), next to other teams' GPU processes that were never touched:
- install, the Ubuntu 22.04 container install, and in-place upgrades;
- model install, scan and remove;
- the 27B NVFP4 model loading next to foreign processes, co-residency and replicas;
- drain, force unload, crash recovery, and restore after restart;
- every gateway and key check;
- VRAM calibration;
- the zero-downtime update:
  - `goethe update` ran while the 27B served a live stream, and the stream completed normally;
  - the engine kept its process and was not reloaded;
  - clients saw 0.27 s of 502s.
- An upgrade from a build older than zero-downtime restart reloads the models instead (about 70 s for the 27B).

The automated gate (49 test files plus 194 unit tests) passes.

Not yet verified on hardware:
- SM86 GPUs, GB10, and aarch64 hosts;
- tensor parallelism across two GPUs, and the Flash-Next model on a single GPU. These tests are waiting for a free second GPU on the test machine.

Known limitations:
- Gated and private Hugging Face models must be downloaded manually and added with `goethe scan`.
- Downloads resume per file, not per byte.
- `goethe domain add` is implemented but not yet tested.
- Ternary model packs are supported in the model cards but not yet tested end to end.
- Multimodal models print a burst of harmless torchcodec warnings at engine start when the host has no matching FFmpeg libraries; video input then needs FFmpeg.
