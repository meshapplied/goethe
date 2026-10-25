# Goethe

Serves LLMs on NVIDIA GPUs (Ampere, Ada, Hopper and Blackwell, SM 8.0 to 12.1) behind one API that accepts the OpenAI, Anthropic (`/v1/messages`) and LiteLLM schemas at once. Rust control plane (`goethe` CLI and `goethed` daemon), SGLang engines, nginx gateway. Linux only (glibc 2.30 or newer, x86_64 and aarch64).

This README is for building and releasing Goethe. To install and use it, see the [user guide](../docs/user-guide.md).

## Install

    curl -fsSL https://meshapplied.com/install.sh | bash

The installer verifies a signed manifest, creates a Python environment with SGLang, installs `goethe` and `goethed` and links `goethe` into `/usr/local/bin` (or `~/.local/bin`), then ends with `goethe start`. It does not start goethed or turn on start at boot; `goethe start` or `goethe up` does. Re-running it over a running goethed applies the new binaries to it and keeps engines serving. No root needed. State lives in `$GOETHE_HOME`, else in the home whose `bin/` holds the running binary (links are followed, so the linked `goethe` finds a custom home), else `/workspace/.goethe`, else `/var/lib/goethe`, else `~/.goethe`.

To install from a local release tree (testing before publishing):

    cd dist/cdn && python3 -m http.server 8080 --bind 127.0.0.1 &
    GOETHE_PKG_URL=http://127.0.0.1:8080 bash dist/cdn/install.sh

## Quickstart

    goethe start                          # guided first run: install a model, serve it, configure API access

Or step by step:

    goethe up                             # start goethed and the gateway, and start them at boot
    goethe hw profile                     # GPUs and links, CPU, RAM, disk, OS, container and cloud
    goethe recommend                      # registry models that fit this machine
    goethe install --model <id> --load    # download, verify, register, wait until RUNNING (or --no-wait)
    goethe ps [--json]                    # instances (instance_id, slot_key, status), gateway URL
    goethe --version                      # version, git commit, build date
    goethe allowlist --add 127.0.0.1/32   # the public gateway denies everything by default
    goethe key --create me --raw          # prints the API key once
    goethe status                         # running or not, API URL, key names and ids

    curl -k https://127.0.0.1:8443/v1/chat/completions \
      -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
      -d '{"model":"<model_id>","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

`goethe ps` prints the real gateway port (8443 by default, another free port when it is busy).

## Choosing GPUs

- `--gpu gpu2 gpu3` uses exactly those GPUs; tensor parallelism is the number listed. A count the model's heads cannot be split over is refused with the counts that work, and a set too small says which larger `--gpu` list fits. The older `--gpus 2,3` still works.
- `--tp N` alone picks N GPUs.
- Neither: Goethe chooses the smallest valid TP that fits, preferring GPUs without foreign processes, then most free VRAM, then GPUs on the same NUMA node. If nothing fits it shrinks the context, then the concurrency, then fails with the exact shortfall.
- The decision is printed with the budget in parts, for example `auto: tp=2 on GPUs 2,3: 33.4 GB per GPU (weights 23.6 + KV 8.0 + overhead 1.8)`. The KV part holds max(ctx, max_conc x ctx / 4) tokens: one request at the full context, or every parallel request at a quarter of it. Every model's pools come from its memory spec (`memory_spec.rs`, from `config.json` or the catalog entry): sliding-window models add `sliding-window KV` sized from what max_conc requests need (passed as `--swa-full-tokens-ratio`, with `--chunked-prefill-size 8192`), hybrid models add `per-request state` for 5 slots per request on SGLang's extra-buffer architectures (3 otherwise) plus one padding slot (passed as `--max-mamba-cache-size`), and DeepSeek-V4 adds its per-request compression state. After a load Goethe learns the real weight size and the engine's CUDA context per GPU type, so later loads hit the planned token counts. `goethe hw pick-gpus --model <id> [--tp N]` shows where a load would go right now without loading anything.
- Goethe never stops a process it did not start. When a target GPU runs other software, the load lists each process (PID, name, user, VRAM) with the command to stop it yourself, asks for confirmation on a terminal, and exits 3 without one. `--yes` loads alongside them; if they leave too little VRAM, the error says how much is missing and which processes hold it.

## Models

Any public Hugging Face repo can be installed. Gated and private repos need `--hf-token`, which asks for a Hugging Face token, checks it and uses it for that download only (it is never stored); `goethe start` offers the same. Only registry models appear in `goethe recommend`.

    goethe install --model org/model      # public repo from Hugging Face
    goethe install --model org/model --hf-token   # gated or private repo: asks for your token
    goethe install --model ./my-model-dir # local directory, registered in place (--copy to copy)
    goethe install --model org/model --sglang-arg --reasoning-parser --sglang-arg qwen3
    goethe models                         # installed and available: loaded instances, or fits: N.N GB per GPU / does not fit
    goethe remove --model org/model       # deletes what Goethe downloaded, copied or scanned

Downloads resume per file, every weight is checked against its Hugging Face sha256 (a changed file is hashed again on re-install), and free disk is checked first. Engine flags are inferred from `config.json` (Qwen3-family models get `--reasoning-parser qwen3`, and the chat template picks `--tool-call-parser qwen3_coder` or `qwen25`; registry and your own flags win) and can be set with `--sglang-arg` on `install` and `config`, or `goethe load --model <id> -- <args>` for one run. Flags Goethe sets itself (`--host`, `--port`, `--tp-size`, `--context-length`, `--max-running-requests`, `--mem-fraction-static`, `--max-mamba-cache-size`, `--api-key`, `--served-model-name`) are refused there; use `--gpu`, `--tp`, `--ctx` and `--max-conc` instead. Repos that ship their own Python code need `goethe install --trust-remote-code`.

`goethe remove` on a folder installed in place (`goethe install --model ./dir`) only unregisters it and prints the `rm -rf` command, because Goethe does not delete files it did not create. A loaded model is drained and unloaded first (asks on a terminal, `--yes` in scripts).

## Models you download yourself

A model folder you placed yourself goes into `$GOETHE_HOME/models`, and Goethe registers it:

    mv /data/org-model $GOETHE_HOME/models/org__model
    goethe scan                           # asks per model; --yes adds all, --json for scripts
    goethe load --model org/model

`goethe scan` registers folders it can serve: a `config.json` with an architecture the installed SGLang knows, safetensors weights (all shards present), a quantization SGLang supports, and a size that fits the GPUs. It names the reason for anything it skips. `org__model` becomes `org/model`, any other name `local/<name>`. Scanned models live in Goethe's models folder, so `goethe remove` deletes them.

## Daily use

    goethe load --model <id> [--gpu gpuN ...] [--tp N] [--ep N] [--ctx N] [--max-conc N] [--replica] [--no-wait] [--json] [-- <sglang flags>]
    goethe unload --model <id> [--force] | --all  # drains in-flight requests (new ones get 503), then unloads; --force cuts them
    goethe list                               # alias of goethe models
    goethe alias [--set <alias> --model <id> | --remove <alias>]   # no flags lists them; an alias is not empty and not an installed id; installing that id later removes it
    goethe gpu [--json]                       # index, compute capability, managed and foreign VRAM, ALLOWED or DENIED
    goethe gpu --deny gpuN | --allow gpuN     # denied GPUs are skipped by automatic placement and refused with --gpu
    goethe config --model <id> --ctx 32768 --max-conc 4
    goethe logs --model <id> [-f] [-n 200]  # also --daemon, --gateway, --prune
    goethe doctor [--json] [--probe-public]  # checks with a fix hint each; probe-public needs GOETHE_CANARY_URL for a real answer
    goethe gateway [--bind 0.0.0.0]       # where the public gateway listens (default 127.0.0.1), or move it
    goethe service enable                 # start goethed at boot (systemd, systemd --user, or cron)
    goethe up                             # start daemon and gateway; prints the goethe status summary
    goethe status                         # goethe: running or not running, the API URL, key names and ids
    goethe down [--keep-engines]          # stop daemon and engines (loaded models return on next up); --keep-engines leaves engines serving
    goethe api [--json]                   # API base URL, auth hint, endpoint, models running, API keys in full, one curl example
    goethe update [--check] [--engine] [--ternary]  # signed self-update; compares the installed binaries, engine wheel and ternary pack
    goethe uninstall [--keep-models]       # deletes the installed models unless --keep-models; your own registered folders stay

`goethe update` notices another build at the same version (the installed `bin/goethe` and `bin/goethed` against the signed checksums, shown as `0.1.0 (commit A) -> 0.1.0 (commit B)`), swaps both binaries, then stops the old goethed with `--keep-engines` and starts the new one (through systemd when this home has a unit, whose `KillMode=process` lets engines outlive goethed), so loaded models keep serving.

Boot entries belong to one home: `service disable` and `uninstall` remove only the unit or cron line that names this `GOETHE_HOME`, and `service enable` refuses when the systemd unit already starts another home.

Loaded models are restored after a daemon restart or reboot unless you unloaded them, in the order they were first loaded, one engine per GPU at a time. A blocking `goethe load` waits until the instance from that command is RUNNING (or FAILED) and ends with `RUNNING: <model> instance <id> on GPU <g> port <p> (<t> s)`; `--json` prints the instance row instead. Loading a model that is already loaded reports it and exits 0; `--replica` starts another instance. SGLang flags for one run go after `--`; anything else unknown is a usage error (exit 2). A value typed without its flag (`goethe unload <id>`, `goethe gpu allow gpu0`) prints the corrected command (`goethe unload --model <id>`) and runs nothing. Crashed engines are restarted with backoff. If goethed itself crashes or is restarted without `goethe down`, healthy engines keep serving and are adopted when it starts again. Every engine carries an identity (boot id, start time, instance tag) that is checked before Goethe signals it, so a reused PID is never signaled.

## Cloud and containers

`goethe hw profile` and `goethe doctor` report the container (Docker, Podman, Kubernetes, LXC), whether systemd runs, and the cloud: RunPod, Vast.ai, Lambda, AWS, GCP, Azure, OCI, CoreWeave or Paperspace, from environment variables and DMI only (no network calls).

- RunPod and Vast.ai: state goes to `/workspace/.goethe` on the persistent volume, and Goethe prints the public endpoint for the gateway port (`https://<pod>-<port>.proxy.runpod.net`, or the Vast port mapping). Remote clients need `goethe gateway --bind 0.0.0.0` and an allowlist entry for the address they connect from.
- No systemd as PID 1: `goethe service enable` uses `nohup` and cron instead of a unit.
- `doctor` warns when `state.db` sits on a network filesystem, when free RAM is low for models that offload to host memory, and when disk is short for the installed models.

## Security

- The public gateway (nginx, TLS, self-signed by default) allows only addresses added with `goethe allowlist --add <cidr>` (`--remove <cidr>` removes one, plain `goethe allowlist` lists them).
- It listens on 127.0.0.1 unless you run `goethe gateway --bind 0.0.0.0` (or one IP address of this host); `goethe gateway --bind 127.0.0.1` closes it again and `goethe gateway` shows bind, port, URL and allowlist size. Moving between loopback and 0.0.0.0 restarts nginx (a few seconds without the gateway); `goethe domain --add <domain>` also binds 0.0.0.0; plain `goethe domain` explains the setup and `goethe domain --check <domain>` verifies it.
- API keys: `goethe key --create <name> [--rpm N] [--expires 30d] [--models a,b] [--admin] [--raw]`, `goethe key`, `goethe key --revoke <id>`. Keys are stored hashed; usage per key is counted. `--rpm` defaults to 120; `--rpm 0` means unlimited and `goethe key` shows it as `unlimited`. An empty name or `--models` is refused.
- A key with `--models` sees only those models in `/v1/models`. Requests for any other model, or for a model that is not loaded, get 404 `Model Not Found`.
- Health: `/health` (public), `/health/ready?model=<id>` (200 only when that model's engine answers), `/health/admin` (admin key only; other callers get 404) returns GPU VRAM, instances with in-flight counts, and inactive models.
- Engines listen on loopback only and each one gets its own random API key, so a local process cannot use an engine without going through the gateway. SGLang refuses keys with `--tokenizer-worker-num` above 1; such engines run keyless (still loopback only).
- nginx rate-limits every client address (20 requests/s, burst 40) on all routes; per-key `--rpm` limits are enforced in the mux. Request bodies are capped at `GOETHE_MAX_BODY_MB` (default 10). Errors are OpenAI-style JSON, Anthropic-style on `/v1/messages`. Public routes are the OpenAI ones (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/models`), Anthropic `/v1/messages` and LiteLLM's spellings (the same routes without `/v1`, `/model/info`, `/model_group/info`, `/health/liveliness`, `/health/readiness`); keys are read from `Authorization: Bearer`, `x-litellm-api-key`, `x-api-key` or `api-key`. `/metrics` stays on loopback.
- Installer requirements: glibc 2.30+, curl 7.50+, zstd 1.3+, openssl 1.1.1+, and an NVIDIA driver 580+ (CUDA 13) when `nvidia-smi` is present (set `GOETHE_REQUIRE_GPU=1` to require it).
- Prometheus metrics at `http://127.0.0.1:<mux port>/metrics` on loopback only.

## Registry

Source file: `cdn/v1/registry.json`, served at `https://pkg.meshapplied.com/v1/registry.json` (override with `GOETHE_REGISTRY_URL`, `file://` works). It describes models only; GPU count, TP, context and concurrency are computed per machine.

Required per entry: `id` (Hugging Face repo, owned by an organization), `runtime` (`sglang` or `ternary`), `architecture` (`dense` or `moe`), `supported_sm`, `static_vram_gb` (whole model), `kv_elems_per_token`, `max_ctx`, `num_attention_heads`, `num_kv_heads`.
Optional: `memory` (the memory spec `registry add` measures from `config.json`; when present, `static_vram_gb` is the weights alone), `attention` (`gqa`, `mla`, `hybrid`), `extra_sglang_args`, `revision`, `gated`, `manual_download` (weights are published somewhere other than Hugging Face), `download_gb`, `description`, `family`, `license`, `released`, `modalities`. Ternary entries also take `pack_id`, `base_model_id`, `env`, `pythonpath`.

Add models without writing numbers by hand:

    goethe registry add --repo org/model-a --repo org/model-b --file cdn/v1/registry.json [--overlay hand-written.json] [--dry-run]
    goethe registry validate --source cdn/v1/registry.json --online --check-sglang

`add` measures size, KV elements, heads, context and the memory spec from Hugging Face, merges hand-written fields from the overlay, validates, and writes atomically. Gated repos cannot be measured without logging in, so their entries are written by hand with `"gated": true`. Entries with `"gated": true` or `"manual_download": true` are never recommended: `goethe recommend` lists them as manual downloads with their fit, `goethe models` labels them, `goethe install` explains where to put the files for `goethe scan`, and `validate --online` does not require weights on Hugging Face for `manual_download` entries.

`goethe recommend` and `goethe models` judge fit with the plan `goethe load` would make on the live GPUs (shrinking context, then concurrency) plus the overhead learned from earlier loads, and say how much is missing when nothing fits.

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

`build_cdn.sh` builds portable binaries in an Ubuntu 20.04 container (x86_64 and aarch64, or one with `--arch x86_64`) with the Rust toolchain pinned in `rust-toolchain.toml` (`rustup toolchain install 1.98.1` once), the patched SGLang wheel, hash-locked Python requirements, `VERSION`, `BUILDINFO` (SGLang fork commit and whether its tree was dirty), `SHA256SUMS` and its signature. Release builds set `GOETHE_RELEASE_PUB_FPR` to the expected public key fingerprint. Add `--with-ternary` to include the ternary pack (keep it off the public origin). `packaging/release_pub.pem` is the production public key (fingerprint `24afa2b3bddcaa4ec385f7640ec3c42b824446de0c0cac6d863ffa84ec4f473f`); local builds signed with a dev key set `GOETHE_RELEASE_PUB=<dev public key>` so the installer embeds that key instead (`--release` refuses the override). The SGLang source is upstream at the commit in `sglang/PIN` plus `sglang/goethe-marlin.patch`.

Releases come from GitHub, never from a laptop:
- Every pull request to `main` and every merge runs `.github/workflows/ci.yml`: Rust build, tests and clippy, the style lint, the CPU gates (`gate.sh 0`, `stage6_integrity`) and an unsigned x86_64 package build.
- To release, set the version in `crates/*/Cargo.toml` through a pull request, then push a tag `vX.Y.Z` on that `main` commit (or run the release workflow by hand on `main`).
- `.github/workflows/release.yml` then checks the commit is on `main` and the tag matches the version, reruns CI, and builds both architectures with `--release` using the `GOETHE_SIGN_KEY` secret. It verifies the checksums, signature, embedded key fingerprint and BUILDINFO, and installs the package in a clean Ubuntu 22.04 container.
- After an approval on the `release` environment, it publishes the GitHub release and deploys the package tree to GitHub Pages, which serves it at `https://pkg.meshapplied.com/` (DNS: `CNAME pkg -> meshapplied.github.io`). `https://meshapplied.com/install.sh` redirects there from the website.
- Existing installs then pick the release up with `goethe update`.

## Tests

    GOETHE_TEST_GPUS=2,3 bash tests/gate.sh <0..6|all> [--only <file>]

The gate first builds, runs `cargo test` and clippy, and uses those fresh binaries. Each test runs in its own session with a timeout and may only clean up processes it started. A pending test, a timeout, or zero collected tests fail the run; skipped opt-in tests are listed. Opt-in: `GOETHE_TEST_REAL=1` (real engines), `GOETHE_TEST_E2E=1` with `GOETHE_TEST_CDN=dist/cdn` (installer), `GOETHE_TEST_DOCKER=1` (container).

## Troubleshooting

Run `goethe doctor` first. Gated or private model: accept its license on Hugging Face, then `goethe install --model <id> --hf-token`. A load exits 3 or reports missing VRAM because of other software on the GPU: the message lists the processes and the command to stop each one yourself, or pick other GPUs with `--gpu`. `nginx binary not found`: install nginx or set `NGINX_BIN`. Logs: `goethe logs --model <id>`, `goethe logs --daemon`, `goethe logs --gateway`.
