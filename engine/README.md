# Goethe

Serves LLMs on NVIDIA GPUs (SM86 and SM120) behind an OpenAI compatible API. Rust control plane (`goethe` CLI and `goethed` daemon), SGLang engines, nginx gateway. Linux only (glibc 2.30 or newer, x86_64 and aarch64).

## Install

    curl -fsSL https://meshapplied.com/install.sh | bash

The installer verifies a signed manifest, creates a Python environment with SGLang, installs `goethe` and `goethed`, starts the daemon and links `goethe` into `/usr/local/bin` (or `~/.local/bin`). No root needed. State lives in `$GOETHE_HOME`, else in the home whose `bin/` holds the running binary (links are followed, so the linked `goethe` finds a custom home), else `/workspace/.goethe`, else `/var/lib/goethe`, else `~/.goethe`.

To install from a local release tree (testing before publishing):

    cd dist/cdn && python3 -m http.server 8080 --bind 127.0.0.1 &
    GOETHE_PKG_URL=http://127.0.0.1:8080 bash dist/cdn/install.sh

## Quickstart

    goethe hw profile                     # GPUs and links, CPU, RAM, disk, OS, container and cloud
    goethe recommend                      # registry models that fit this machine
    goethe install <model_id> --load      # download, verify, register, wait until RUNNING (or --no-wait)
    goethe ps [--json]                    # instances (instance_id, slot_key, status), gateway URL
    goethe --version                      # version, git commit, build date
    goethe allowlist add 127.0.0.1/32     # the public gateway denies everything by default
    goethe key create --name me --raw     # prints the API key once

    curl -k https://127.0.0.1:8443/v1/chat/completions \
      -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
      -d '{"model":"<model_id>","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

`goethe ps` prints the real gateway port (8443 by default, another free port when it is busy).

## Choosing GPUs

- `--gpus 2,3` uses exactly those GPUs; tensor parallelism is the number listed.
- `--tp N` alone picks N GPUs.
- Neither: Goethe chooses the smallest valid TP that fits, preferring GPUs without foreign processes, then most free VRAM, then GPUs on the same NUMA node. If nothing fits it shrinks the context, then the concurrency, then fails with the exact shortfall.
- The decision is printed, for example `auto: tp=2 on GPUs 2,3 (1 GPU needs 61.2 GB, best free 52.0 GB)`.
- Goethe never stops a process it did not start. When a target GPU runs other software, the load lists each process (PID, name, user, VRAM) with the command to stop it yourself, asks for confirmation on a terminal, and exits 3 without one. `--yes` loads alongside them; if they leave too little VRAM, the error says how much is missing and which processes hold it.

## Models

Any public Hugging Face repo can be installed. Goethe does not log in to Hugging Face, so gated and private repos are downloaded by you (see below). Only registry models appear in `goethe recommend`.

    goethe install org/model              # public repo from Hugging Face
    goethe install ./my-model-dir         # local directory, registered in place (--copy to copy)
    goethe install org/model --sglang-arg --reasoning-parser --sglang-arg qwen3
    goethe models                         # installed and available, with fit on this machine
    goethe remove org/model               # deletes what Goethe downloaded, copied or scanned

Downloads resume per file, every weight is checked against its Hugging Face sha256 (a changed file is hashed again on re-install), and free disk is checked first. Engine flags are inferred from `config.json` and can be set with `--sglang-arg` on `install` and `config`, or `goethe load <model> -- <args>` for one run. Flags Goethe sets itself (`--host`, `--port`, `--tp-size`, `--context-length`, `--max-running-requests`, `--mem-fraction-static`, `--api-key`, `--served-model-name`) are refused there; use `--gpus`, `--tp`, `--ctx` and `--max-conc` instead. Repos that ship their own Python code need `goethe install --trust-remote-code`.

`goethe remove` on a folder installed in place (`goethe install ./dir`) only unregisters it and prints the `rm -rf` command, because Goethe does not delete files it did not create. A loaded model is drained and unloaded first (asks on a terminal, `--yes` in scripts).

## Models you download yourself

For gated or private repos, accept the license on Hugging Face, download with your own account, move the folder into `$GOETHE_HOME/models`, and let Goethe register it:

    hf download org/model --local-dir $GOETHE_HOME/models/org__model
    goethe scan                           # asks per model; --yes adds all, --json for scripts
    goethe load org/model

`goethe scan` registers folders it can serve: a `config.json` with an architecture the installed SGLang knows, safetensors weights (all shards present), a quantization SGLang supports, and a size that fits the GPUs. It names the reason for anything it skips. `org__model` becomes `org/model`, any other name `local/<name>`. Scanned models live in Goethe's models folder, so `goethe remove` deletes them.

## Daily use

    goethe load <model_id> [--gpus csv] [--tp N] [--ep N] [--ctx N] [--max-conc N] [--no-wait]
    goethe unload <model_id> [--force] | --all  # drains in-flight requests (new ones get 503), then unloads; --force cuts them
    goethe list                               # alias of goethe models
    goethe alias set <alias> <model_id> | list | remove <alias>
    goethe gpu list [--json]                  # index, compute capability, managed and foreign VRAM, ALLOWED or DENIED
    goethe gpu deny <id> | allow <id>         # denied GPUs are skipped by automatic placement and refused with --gpus
    goethe config <model_id> --ctx 32768 --max-conc 4
    goethe logs <model_id> [-f] [-n 200]  # also --daemon, --gateway, --prune
    goethe doctor [--json] [--probe-public]  # checks with a fix hint each; probe-public needs GOETHE_CANARY_URL for a real answer
    goethe service enable                 # start goethed at boot (systemd, systemd --user, or cron)
    goethe stop                           # stop daemon and engines (loaded models return on next start)
    goethe update [--check] [--engine]    # signed self-update
    goethe uninstall [--purge-models]

Boot entries belong to one home: `service disable` and `uninstall` remove only the unit or cron line that names this `GOETHE_HOME`, and `service enable` refuses when the systemd unit already starts another home.

Loaded models are restored after a daemon restart or reboot unless you unloaded them, in the order they were first loaded, one engine per GPU at a time. A blocking `goethe load` waits until the instance from that command is RUNNING (or FAILED). Crashed engines are restarted with backoff. If goethed itself crashes or is restarted without `goethe stop`, healthy engines keep serving and are adopted when it starts again. Every engine carries an identity (boot id, start time, instance tag) that is checked before Goethe signals it, so a reused PID is never signaled.

## Cloud and containers

`goethe hw profile` and `goethe doctor` report the container (Docker, Podman, Kubernetes, LXC), whether systemd runs, and the cloud: RunPod, Vast.ai, Lambda, AWS, GCP, Azure, OCI, CoreWeave or Paperspace, from environment variables and DMI only (no network calls).

- RunPod and Vast.ai: state goes to `/workspace/.goethe` on the persistent volume, and Goethe prints the public endpoint for the gateway port (`https://<pod>-<port>.proxy.runpod.net`, or the Vast port mapping).
- No systemd as PID 1: `goethe service enable` uses `nohup` and cron instead of a unit.
- `doctor` warns when `state.db` sits on a network filesystem, when free RAM is low for models that offload to host memory, and when disk is short for the installed models.

## Security

- The public gateway (nginx, TLS, self-signed by default) allows only addresses added with `goethe allowlist add|remove|list`.
- API keys: `goethe key create --name N [--rpm N] [--expires 30d] [--models a,b] [--admin]`, `goethe key list`, `goethe key revoke <id>`. Keys are stored hashed; usage per key is counted.
- A key with `--models` sees only those models in `/v1/models`. Requests for any other model, or for a model that is not loaded, get 404 `Model Not Found`.
- Health: `/health` (public), `/health/ready?model=<id>` (200 only when that model's engine answers), `/health/admin` (admin key only; other callers get 404) returns GPU VRAM, instances with in-flight counts, and inactive models.
- Engines listen on loopback only and each one gets its own random API key, so a local process cannot use an engine without going through the gateway. SGLang refuses keys with `--tokenizer-worker-num` above 1; such engines run keyless (still loopback only).
- nginx rate-limits every client address (20 requests/s, burst 40) on all routes; per-key `--rpm` limits are enforced in the mux. Request bodies are capped at `GOETHE_MAX_BODY_MB` (default 10). Errors are OpenAI-style JSON.
- Installer requirements: glibc 2.30+, curl 7.50+, zstd 1.3+, openssl 1.1.1+, and an NVIDIA driver 580+ (CUDA 13) when `nvidia-smi` is present (set `GOETHE_REQUIRE_GPU=1` to require it).
- Prometheus metrics at `http://127.0.0.1:<mux port>/metrics` on loopback only.

## Registry

Source file: `cdn/v1/registry.json`, served at `https://pkg.meshapplied.com/v1/registry.json` (override with `GOETHE_REGISTRY_URL`, `file://` works). It describes models only; GPU count, TP, context and concurrency are computed per machine.

Required per entry: `id` (Hugging Face repo, owned by an organization), `runtime` (`sglang` or `ternary`), `architecture` (`dense` or `moe`), `supported_sm`, `static_vram_gb` (whole model), `kv_elems_per_token`, `max_ctx`, `num_attention_heads`, `num_kv_heads`.
Optional: `attention` (`gqa`, `mla`, `hybrid`), `extra_sglang_args`, `revision`, `gated`, `download_gb`, `description`, `family`, `license`, `released`, `modalities`. Ternary entries also take `pack_id`, `base_model_id`, `env`, `pythonpath`.

Add models without writing numbers by hand:

    goethe registry add org/model-a org/model-b --file cdn/v1/registry.json [--overlay hand-written.json] [--dry-run]
    goethe registry validate cdn/v1/registry.json --online --check-sglang

`add` measures size, KV elements, heads and context from Hugging Face, merges hand-written fields from the overlay, validates, and writes atomically. Gated repos cannot be measured without logging in, so their entries are written by hand with `"gated": true`; `goethe recommend` and `goethe models` mark them as manual downloads.

## Defaults worth knowing

- NVFP4 layers use Marlin: SM86 for every batch size, SM120 up to 128 rows, stock W4A4 above. `GOETHE_MARLIN=0` restores stock behavior.
- `--tp` above 1 adds `--disable-custom-all-reduce`; set `custom_all_reduce = true` on the card to keep it.
- `--kv-cache-dtype` is passed only when the card sets one other than `auto`; the VRAM budget uses that dtype.
- Ports: mux on loopback 30000 (`GOETHE_MUX_PORT`), public 8443 (`GOETHE_PUBLIC_PORT`), engines 30001 to 30099 (`GOETHE_ENGINE_PORTS=lo-hi` to move them away from other software). Busy defaults fall back to a free port that is remembered.
- `--mem-fraction-static` is the budget divided by the free VRAM at launch, so several engines can share a GPU.
- `goethe unload` drains for up to `GOETHE_DRAIN_SECS` (default 120) seconds.

## Environment

`GOETHE_HOME`, `GOETHE_SOCK`, `GOETHE_MUX_PORT`, `GOETHE_PUBLIC_PORT`, `GOETHE_ENGINE_PORTS`, `GOETHE_DRAIN_SECS`, `GOETHE_MAX_BODY_MB`, `GOETHE_MARLIN`, `GOETHE_PKG_URL`, `GOETHE_PKG_TOKEN` (private packages), `GOETHE_REGISTRY_URL`, `HF_ENDPOINT`, `HF_HUB_OFFLINE`, `NGINX_BIN`, `GOETHE_REQUIRE_GPU`, `GOETHE_NO_LINK` (installer), `GOETHE_INSECURE` (installer, skips signature checks, development only).

`$GOETHE_HOME/goethe.env` (written by the installer, `KEY=VALUE` lines) supplies `GOETHE_PKG_URL`, `GOETHE_REGISTRY_URL` and `GOETHE_PKG_TOKEN` when the environment does not set them (goethe reads no other keys from it); the systemd unit loads the same file.

## Building a release

    bash packaging/keygen.sh <dir>                                   # once: release.key and release_pub.pem
    GOETHE_SIGN_KEY=<dir>/release.key bash packaging/build_cdn.sh      # builds dist/cdn (needs docker, uv, zstd, rsync)

`build_cdn.sh` builds portable binaries in an Ubuntu 20.04 container (x86_64 and aarch64, or one with `--arch x86_64`) with the Rust toolchain pinned in `rust-toolchain.toml` (`rustup toolchain install 1.98.1` once), the patched SGLang wheel, hash-locked Python requirements, `VERSION`, `BUILDINFO` (SGLang fork commit and whether its tree was dirty), `SHA256SUMS` and its signature. Release builds set `GOETHE_RELEASE_PUB_FPR` to the expected public key fingerprint. Add `--with-ternary` to include the ternary pack (keep it off the public origin). Upload the contents of `dist/cdn/` to `https://pkg.meshapplied.com/` and serve `install.sh` at `https://meshapplied.com/install.sh`. Replace `packaging/release_pub.pem` with your production public key before the first public release.

## Tests

    GOETHE_TEST_GPUS=2,3 bash tests/gate.sh <0..6|all> [--only <file>]

The gate first builds, runs `cargo test` and clippy, and uses those fresh binaries. Each test runs in its own session with a timeout and may only clean up processes it started. A pending test, a timeout, or zero collected tests fail the run; skipped opt-in tests are listed. Opt-in: `GOETHE_TEST_REAL=1` (real engines), `GOETHE_TEST_E2E=1` with `GOETHE_TEST_CDN=dist/cdn` (installer), `GOETHE_TEST_DOCKER=1` (container).

## Troubleshooting

Run `goethe doctor` first. Gated or private model: download it yourself and run `goethe scan` (see Models you download yourself). A load exits 3 or reports missing VRAM because of other software on the GPU: the message lists the processes and the command to stop each one yourself, or pick other GPUs with `--gpus`. `nginx binary not found`: install nginx or set `NGINX_BIN`. Logs: `goethe logs <model>`, `goethe logs --daemon`, `goethe logs --gateway`.
