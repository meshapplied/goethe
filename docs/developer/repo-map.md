# Repository map

Where things are. "Product" means it ships in the Goethe package or decides what ships.

## Top level

| Path | Product? | What |
|---|---|---|
| `goethe/` | yes | Everything that ships: Rust crates, packaging, pins, gateway templates, registry, tests. |
| `docs/` | yes | User documentation (`user-guide.md`, `cli.md`, `learn.md`, `traces.md`, `pack.md`, `registry.md`, `research-models.md`) and this developer folder. `docs/dev/` is git-ignored old notes; do not rely on it. |
| `.github/workflows/` | yes | CI, release, installer VM, macOS, Pages redeploy, engine pin bump ([workflows.md](workflows.md#ci)). |
| `install.sh` (root) | no (dev) | Builds Goethe from source with cargo (`GOETHE_REF`, `CARGO_INSTALL_ROOT`). The shipped installer is `goethe/cdn/install.sh`. |
| `AGENTS.md` | yes (rules) | Hard rules for anyone changing the repo. |
| `CHANGELOG.md` | yes | `## Unreleased` is dated by `release.sh prepare`. |
| `LICENSE`, `NOTICE` | yes | Apache 2.0 and third-party notices; both are copied into the package. |
| `nvfp4-kernels/` | partly | NInfer for the Qwen3.8-27B NVFP4 model: `NINFER_PIN`, patches, `install.sh` (builds `ninfer-serve` into `~/.cache/goethe/nvfp4/ninfer`, where `engines/ninfer.rs` finds it), `fetch_model.sh`, benches. Not packaged; CI checks the patches still apply. |
| `ternary-kernels/` | optional | The `ternary_runtime` Python library for ternary models. `build_cdn.sh --with-ternary` packs it as `goethe/ternary-kernels.tar.zst`; keep it off the public origin. |
| `lab/`, `tests/` (root), `examples/`, `ruff.toml` | no | `goethe-lab`, the Python research launcher, with its unit tests and example cards. CI lints and tests them, but nothing here ships. |
| `modal/` | no | Local, git-ignored proof of concept. |
| `WINDOWS-COMPATIBILITY.md` | no | Untracked plan. |

## goethe/

| Path | What |
|---|---|
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` | Workspace of three crates; the toolchain is pinned (1.98.1, with clippy and rustfmt). |
| `crates/goethe/` | The CLI binary. `main.rs` holds the clap `Cli`, `HELP_SECTIONS` and the dispatch; `complete.rs` supplies live values for shell completion (clap_complete `CompleteEnv`). |
| `crates/goethed/` | The daemon binary: `main.rs` (socket RPC, main loop, shutdown), `mux.rs` (the HTTP API), `heartbeat.rs` (heartbeat task), `curator.rs` (learn-note model calls), `instance_lock.rs`. |
| `crates/core/` | `goethe-core`, the library both binaries use; nearly all logic lives here (modules below). `build.rs` embeds the git commit and build date. |
| `nginx/` | `goethe_gateway.conf.tmpl` (the gateway) and `goethe_public.conf.tmpl` (the public share listener). Rendered by `nginx.rs` and `public_share.rs`. |
| `cdn/install.sh` | The `curl \| bash` installer, with the release public key substituted in at build. |
| `cdn/v1/registry.json` | The model catalog, also compiled into the binary as a fallback. `cdn/v1/sglang_support.json` lists the architectures the pinned SGLang serves. |
| `pyenv/` | `bootstrap.sh` (builds the py-env), `layers.sh` (regenerates layer locks), `requirements*.in` and `requirements*.lock` (hash-locked, with x86_64 and aarch64 variants). |
| `sglang/` | `PIN` (upstream commit), `goethe-marlin.patch`, `apply.sh`. |
| `goethe-engine/PIN` | goethe-engine wheel version, URL and sha256, compiled into the binary. |
| `llama-cpp/` | `PIN` (release tag, commit, archive URLs and sha256 for Linux CUDA, cudart and macOS), `ARCHS` (architectures the pin serves), `update_archs.sh`. |
| `packaging/` | Build, sign and release scripts ([packages.md](packages.md), [workflows.md](workflows.md#release)). `release_pub.pem` is the production public key. `nvfp4-patch/` duplicates `nvfp4-kernels/NINFER_PIN` and patches; nothing references it. |
| `tests/` | Integration tests ([workflows.md](workflows.md#tests)). |
| `experimental/goethe-kernels/` | Archived kernel work; not built, not packaged. |
| `dist/`, `target/`, `target-portable/` | Build output; not tracked. |

## goethe-core modules

Every file starts with a one-line header that says what it is; read that before opening the file.

### Daemon, supervision, processes

| Module | What |
|---|---|
| `supervisor.rs` | Load, unload, ps, adoption at start, watchdog, restart scheduling. The largest module. |
| `coordinator.rs` | One planning lock, the VRAM ledger of STARTING engines, the per-GPU startup gate, cancel tickets. |
| `load_slot.rs` | Per-slot cancel flags, the in-flight count of startup threads, notices per instance. |
| `slots.rs` | Slot keys: model plus sorted GPUs plus ordinal. |
| `load_wait.rs` | CLI side: wait for a load to reach RUNNING. |
| `autostart.rs` | `desired_loads`: what to restore after a restart. |
| `shutdown.rs` | Shutdown flags, `goethe down`, keep-engines stops for upgrades. |
| `watchdog_liveness.rs`, `watchdog_policy.rs` | Pure watchdog decisions (probes, restart window, backoff). |
| `startup_stall.rs` | Startup readiness timeout by stall detection. |
| `proc_ident.rs` | Engine identity and the only path that signals processes. |
| `run_registry.rs` | `run/<instance>.json` identities for reaping strays. |
| `foreign.rs` | Live NVML view and the notice about processes Goethe did not start. |
| `listener.rs` | Who listens on a port (`/proc/net/tcp`, or lsof on macOS). |
| `ports.rs` | Mux, public and engine port selection and persistence. |
| `inflight.rs` | In-flight counters per model and instance. |
| `rpc.rs` | Client for the goethed socket. |
| `log_rotation.rs`, `logs.rs` | Log caps and retention; `goethe logs`. |
| `service.rs` | Start at boot: systemd system or user unit, cron with nohup, LaunchAgent. |
| `sys.rs` | Process and host facts for Linux (`/proc`) and macOS (libproc, sysctl). |

### Engines

| Module | What |
|---|---|
| `engines/mod.rs` | The `Engine` descriptor (id, py-env layer, supported request params, command builder), `ENGINES`, shared env and key helpers. |
| `engines/sglang.rs` | SGLang launch: Goethe-owned flags, card extras, ternary env. |
| `engines/goethe_engine.rs` | goethe-engine: a signed kernel pack plus weights. |
| `engines/llama_cpp.rs` | llama.cpp `llama-server` for GGUF, CUDA or Metal build from `llama-cpp/PIN`. |
| `engines/ninfer.rs` | NInfer: the pinned `.ninfer` artifact and `ninfer-serve`. |
| `sglang_infer.rs`, `sglang_validate.rs` | SGLang flag groups, reserved flags, config-derived notes; validation of registry flags against the installed SGLang. |
| `engine_overhead.rs` | Learned extra VRAM per model, GPU and TP. |
| `cuda_compat.rs`, `cuda_dev.rs`, `cuda_toolkit.rs`, `host_compiler.rs` | CUDA forward compatibility, device facts for pack grids, the bundled toolkit, the host C++ compiler. |
| `runtime.rs` | Optional ternary runtime under the home. |

### Fit and placement

| Module | What |
|---|---|
| `model_facts.rs` | Shape facts from `config.json` and weights on disk. |
| `memory_spec.rs` | What SGLang allocates per GPU generation. |
| `safetensors.rs` | Header scans (Marlin extra, offloaded bytes). |
| `calc.rs` | The VRAM budget and the one fit rule. |
| `chooser.rs` | TP and GPU set choice, shrinking context then concurrency. |
| `placement.rs` | GPU set ranking, managed pids, live instance ids, hardware profile. |
| `recommend.rs` | Registry recommendations judged with the same plan. |
| `gpu_arch.rs`, `gpu_kind.rs`, `gpu_policy.rs` | Compute-capability table, GPU kind (CUDA, Metal, ROCm), allow and deny. |
| `hw.rs`, `host.rs`, `apple.rs`, `nvidia_driver.rs` | NVML probe and `hardware_state.json`, host profile, Mac GPU and memory, why the NVIDIA driver is unusable. |

### Models

| Module | What |
|---|---|
| `install.rs` | `goethe install`: Hugging Face or local trees, engine choice, card, registration. |
| `hf.rs`, `net.rs`, `progress.rs` | Hugging Face download and verify, the shared IPv4-first HTTP agent, download progress. |
| `registry.rs` | Registry fetch, signature check, validation, `goethe registry add`. |
| `models.rs`, `scan.rs`, `alias.rs`, `config.rs` | List and remove, scan of user-placed folders, aliases, card and default edits. |
| `packs.rs`, `pack_update.rs` | Kernel-pack pull from the console (trust decided by goethe-engine) and block-level pack updates. |
| `pyenv.rs` | The layered py-env. |

### Gateway and API

| Module | What |
|---|---|
| `gateway_cache.rs` | In-memory view of state.db for the mux, plus the batched key-usage writer. |
| `security.rs`, `key_usage.rs`, `keys_menu.rs` | Keys and allowlist; usage per key per day; the full-screen key manager. |
| `nginx.rs` | Gateway config, certificates, nginx process control, limits. |
| `public_share.rs` | `goethe api share`: the time-boxed public listener. |
| `configure.rs`, `netif.rs`, `domain.rs` | `goethe api expose`, interface and route discovery, domain help and check. |
| `metrics.rs` | Prometheus exposition on the loopback mux. |
| `api_schema.rs` | Example calls per API schema for the menus. |
| `trace.rs`, `trace_assemble.rs` | Trace recorder; stream-to-body assembly (also used by the mux). |
| `learn.rs` | In-session learning notes. |

### Console link and telemetry

| Module | What |
|---|---|
| `platform.rs`, `platform_api.rs`, `platform_cli.rs` | URL policy, credentials and state files, the device API client, `goethe login` and `goethe logout`. |
| `heartbeat.rs`, `snapshot.rs` | One heartbeat step; the `goethe.snapshot.v1` payload. |
| `usage_log.rs`, `metrics_view.rs` | Minute and hour usage rows; the Metrics page and `goethe metrics`. |

### Install, update, lifecycle

| Module | What |
|---|---|
| `setup.rs` | `goethe setup`: everything after `install.sh` verified the binary. |
| `update.rs` | `goethe update`: signed package, per-component deltas, rollback-safe swap. |
| `release_key.rs` | The compiled-in release key and `verify`. |
| `uninstall.rs` | Removes a proven Goethe home. |
| `fix.rs`, `fix_driver.sh`, `doctor.rs` | `goethe doctor` and `--fix`. |
| `completion.rs` | Per-shell completion files and the rc block. |
| `paths.rs`, `db.rs` | Home resolution, model id validation, settings; the SQLite schema. |
| `build_info.rs` | `--version` and `--build-info`. |

### Terminal UI

| Module | What |
|---|---|
| `term.rs`, `theme.rs`, `step.rs` | Tables, colors (`NO_COLOR`, `TERM=dumb`), raw mode, select and live views; themes; step output with a log of every subprocess. |
| `menu.rs`, `onboard.rs`, `status_view.rs`, `chat.rs`, `run.rs` | The bare `goethe` menu and its pages, the guided first run, Status page, the chat screen, `goethe run`. |
| `messages.rs` | Sentences used in more than one place; a test fails on a second copy elsewhere. |
| `doctor.rs` | Health checks with a fix hint each. |

## goethe/tests

| Path | What |
|---|---|
| `gate.sh` | Builds, runs cargo test, clippy and the style lint, then the stage tests `stage<N>_*.py` for stages 0 to 6 in isolated sessions. |
| `stage0_*` | Harness self-test. |
| `stage1_*` | The py-env and SGLang fork, the NVFP4 Marlin routing (layer and MoE); the real-model test is opt-in (`GOETHE_TEST_REAL=1`). These need the dev py-env and a GPU. |
| `stage2_*` | Hardware profile, GPU policy, foreign pids, pick and recommend. |
| `stage3_*` | CLI flows (install, models, keys, logs, update, uninstall, service, registry, and others). |
| `stage4_*` | Supervisor with stub engines (`*_stub.py`) and with real GPUs (`stage4_hw.py`, `stage4_recovery_hw.py`). |
| `stage5_*` | Gateway, bind, hardening, `api expose`. |
| `stage6_integrity.py` | Package signing and verification. |
| `lib/` | `goethe_test.py` (shared harness) and `stub_engine.py` (fake `sglang.launch_server`). |
| `e2e_cli.py`, `e2e/` | The real binaries with no GPU. `world.py` fakes the package host, console, Hub mirror, packs and engines; `fake/` holds fake `nvidia-smi`, `hf`, `uv`, `llama-server` and `goethe_engine`; `cuda_shim.c` and `nvml_shim.c` are fake driver libraries. |
| `mutants/` | Patches that bring back bugs that shipped once; `run.sh` checks the e2e cells go red for each. |
| `vm/` | Installer in fresh KVM VMs (`run.sh`, `scenarios.sh`, `inside.sh`). |
| `macos_e2e.sh` | Package install and a small GGUF on an Apple Silicon Mac. |
| `accept.sh` | GPU acceptance on a real machine; not in CI. |
| `load/mixed_load.py` | Mixed concurrent load scenarios for registry vetting. |
| `net/` | VPN interface test in a container with its own network. |
| `trace/`, `learn/` | Trace and learn end-to-end checks. `icl/` and the `.work/` folders are local and git-ignored. |
| `lint_style.py` | The style linter ([best-practices.md](best-practices.md#style-rules)). |
| `ux_run.py`, `ux_gaps.py` | Time a command's output (exit code, seconds, longest silence) and list the silences of a trace; used by the acceptance UX cells. |
