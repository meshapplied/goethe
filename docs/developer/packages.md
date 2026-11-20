# Packages and pins

Everything Goethe depends on, why it is there, where it is pinned, and how to move the pin. A pin bump is a normal pull request; CI and the package build prove it.

## Rust

The workspace is `goethe/Cargo.toml` with three members. The toolchain is pinned in `goethe/rust-toolchain.toml` (1.98.1, with clippy and rustfmt); `build_bin.sh` uses the same pin. The release profile uses thin LTO and strips symbols. `Cargo.lock` is committed, and CI builds with `--locked`.

| Crate | Dependencies | Why |
|---|---|---|
| `goethe-core` | `rusqlite` (bundled SQLite) | `state.db`, and SQLite needs no system library. |
| | `nvml-wrapper` | GPU facts and processes through NVML, loaded at run time. |
| | `ureq` (TLS) | Blocking HTTPS for the registry, Hugging Face, the package host and the console. |
| | `serde`, `serde_json`, `toml` | JSON everywhere; `goethe_card.toml` cards. |
| | `sha2`, `flate2` | Weight and package hashes; gzip trace files. |
| | `chrono` (clock only) | Dates for key expiry and display. |
| | `libc` | pidfd, signals, termios, flock, umask, sysctl. |
| `goethe` | `clap` (derive), `clap_complete` (dynamic) | Argument parsing and live shell completion. |
| `goethed` | `tokio`, `hyper` 1, `hyper-util`, `http-body-util` | The async mux server and the pooled upstream client. |

- **Avoid new crates.** The terminal UI is hand-written over libc on purpose, with no TUI crate. A new dependency needs a reason a small module cannot cover.
- **`test-release-key` feature.** Present on `goethe-core` and `goethe`, it lets a test build trust `GOETHE_RELEASE_PUB`. It must never be a default feature or appear in a release script or the release workflow; `lint_style.py` enforces both. See [security.md](security.md#release-trust).

## Python py-env

Goethe owns a hermetic Python 3.11 environment at `$GOETHE_HOME/py-env`, built by `goethe/pyenv/bootstrap.sh` with the pinned `uv`. It is installed lazily in layers (`pyenv.rs`, `Layer::{Base, Engine, Sglang}`), only when a model needs that layer:

| Layer | Lock | Contents |
|---|---|---|
| base | `requirements.base.lock` | `huggingface-hub` (the `hf` CLI downloads models). Every install has it. |
| engine | `requirements.engine.lock` | goethe-engine's runtime: `torch`, `transformers`, `gguf`, plus the goethe-engine wheel from `goethe-engine/PIN`. |
| sglang | `requirements.lock` | The full lock, plus Goethe's patched SGLang wheel (`wheels/LATEST` in the package). |

- **Locks.** Every lock is hash-locked (`--generate-hashes`) and installed with hashes required. aarch64 has its own `.aarch64.lock` files.
- **The main locks** (`requirements.lock`, `requirements.aarch64.lock`) are compiled from `requirements.in` and `requirements.aarch64.in` with the `uv pip compile` command recorded in each lock's header (Python 3.11, `manylinux_2_28` for the arch).
- **The layer locks are derived, never edited.** After changing a main lock, run `bash goethe/pyenv/layers.sh`. It resolves each layer's closure from `requirements.<layer>.in` and copies the matching entries from the main lock, so every layer pins the same versions.
- **Concurrency.** `pyenv::ensure` holds `run/pyenv.lock` while it installs, so concurrent commands do not race.

**To bump a Python dependency:**

1. Edit `requirements.in` (and the aarch64 `.in`).
2. Recompile both main locks with the command in their header.
3. Run `layers.sh`.
4. Build the package (`build_cdn.sh --arch x86_64`) and run the GPU gates for anything that touches torch or SGLang.

## Engines

| Engine | Pin | Where it comes from | How it is trusted |
|---|---|---|---|
| SGLang | `goethe/sglang/PIN` (`SGLANG_REPO`, `SGLANG_REV`) | Upstream at that commit, plus `goethe/sglang/goethe-marlin.patch` applied by `apply.sh`; `build_cdn.sh` builds the wheel with `uv build` and ships it in `wheels/`. | Listed in the signed `SHA256SUMS`. |
| goethe-engine | `goethe/goethe-engine/PIN` (version, wheel URL, sha256) | A GitHub release wheel of the separate goethe-engine repository; not in the package. The pin is compiled into the binary (`pyenv.rs` includes the file). | Installed with `--require-hashes` against the compiled sha256. Kernel packs are verified by goethe-engine itself. |
| llama.cpp | `goethe/llama-cpp/PIN` (tag, commit, archive URLs and sha256 for Linux CUDA, cudart and macOS) | Upstream release archives, downloaded on first use into `engines/llama.cpp`. | sha256 from the pin. |
| NInfer | `nvfp4-kernels/NINFER_PIN` | Built from source by `nvfp4-kernels/install.sh`; `engines/ninfer.rs` looks for `ninfer-serve` (or `NINFER_SERVE`). The model artifact is fetched from Hugging Face at a pinned revision. | The artifact's size and sha256 (before and after the head patch) are constants in `engines/ninfer.rs` that repeat the pin. |

### Bumping each engine pin

- **SGLang.**
  1. Move `SGLANG_REV` in `sglang/PIN`.
  2. Make `goethe-marlin.patch` apply to the new commit (`apply.sh` refuses a patch that does not apply cleanly).
  3. Regenerate the pinned Python locks if SGLang's requirements moved.
  4. Regenerate `cdn/v1/sglang_support.json`. `build_cdn.sh` prints the exact command, which unpacks the wheel and runs `GOETHE_SGLANG_SITE=<unpacked wheel> GOETHE_SGLANG_SNAPSHOT_WRITE=1 cargo test -p goethe-core --lib -- --ignored --exact scan::tests::sglang_snapshot_matches`. A `--release` build fails when the snapshot does not match the wheel.
  5. Run the GPU gates, then re-vet registry models against `docs/registry.md`.
- **goethe-engine.**
  1. The `engine-pin` workflow runs daily: it opens a pull request to `dev` that moves `goethe-engine/PIN` to the newest goethe-engine release, with the checked sha256.
  2. Review the pull request and merge it.
  3. Installed nodes pick the new pin up through `goethe update`, which runs `goethe update engine` when the pin moved.
- **llama.cpp.**
  1. Update every URL and sha256 in `llama-cpp/PIN`.
  2. Rerun `llama-cpp/update_archs.sh` so `llama-cpp/ARCHS` lists the architectures the new build serves.
  3. The macOS workflow runs when the pin changes.
- **NInfer.**
  1. Move `NINFER_REV` in `nvfp4-kernels/NINFER_PIN` and in `goethe/packaging/nvfp4-patch/NINFER_PIN`, the copy compiled into the binary (`engines/ninfer.rs`). CI's `ninfer-patch` job checks that the patches still apply to it.
  2. If the model artifact changes, update the constants in `engines/ninfer.rs` too.

## nginx

Goethe ships its own nginx so the gateway behaves the same everywhere. It is built with static OpenSSL and PCRE2.

- **Linux:** `packaging/build_nginx_linux.sh`, on Ubuntu 20.04 for a glibc 2.31 baseline, for the host's architecture (aarch64 builds on an arm runner).
- **macOS:** `packaging/build_macos.sh`.
- **Pins:** source versions and sha256 are in `packaging/nginx_versions.sh` (nginx, OpenSSL, PCRE2). To bump, change a version and its sha256 there.
- **Installed path:** `$GOETHE_HOME/nginx-root/usr/sbin/nginx`. `NGINX_BIN` overrides it, which is useful for dev homes.

## uv

`uv` builds the SGLang wheel and installs the py-env.

- **Pin:** version and per-arch sha256 in `packaging/build_cdn.sh` (`UV_VERSION`, `UV_SHA256`). The aarch64 job in `.github/workflows/release.yml` repeats the version and sha256, and the `build` job there and CI's `package-smoke` job name the version too.
- **Bumping:** change all four places together.
- **Installed path:** `$GOETHE_HOME/bin/uv`.

## CUDA forward compatibility

On drivers 535 to 579, CUDA 13 runs through NVIDIA's forward-compatibility libraries. `goethe doctor --fix` installs them from an NVIDIA `.deb` whose sha256 is pinned in `crates/core/src/fix_driver.sh`, into `$GOETHE_HOME/cuda-compat` (`cuda_compat.rs`).

## The package

`build_cdn.sh` assembles `goethe/dist/cdn`, which release CI publishes at `https://pkg.goethe.sh`:

```
SHA256SUMS, SHA256SUMS.sig, VERSION, BUILDINFO, install.sh, install.ps1, LICENSE, NOTICE
bin/linux-x86_64/{goethe,goethed,nginx,uv}  bin/linux-aarch64/...  bin/darwin-aarch64/{goethe,goethed,nginx}
pyenv/bootstrap.sh, pyenv/requirements*.lock
wheels/LATEST, wheels/sglang-*.whl
v1/registry.json, v1/registry.json.sig
goethe/ternary-kernels.tar.zst   (only with --with-ternary)
```

- **Build steps:** `build_bin.sh` builds the portable Linux binaries in an Ubuntu 20.04 container. `sign_cdn.sh` writes `VERSION`, `SHA256SUMS` and the signature and verifies them with the key embedded in `install.sh`. `pages_site.sh` lays out the Pages site: the release being deployed at the root (what older installers read) and the newest 4 under `v/<version>/`.
- **`BUILDINFO`** records the SGLang fork commit and whether its tree was dirty.
