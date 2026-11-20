# Workflows

How to build, test, lint and release, and the recipes for the common kinds of change. Commands run from `goethe/` unless they say otherwise.

## Build

```bash
cargo build                                        # target/debug/goethe, goethed
cargo build --release                              # target/release
cargo build --locked --features goethe/test-release-key   # what the e2e and mutant tests use
```

| Build | Command | Output |
|---|---|---|
| Portable Linux binaries (glibc 2.31 baseline, pinned toolchain, Ubuntu 20.04 container) | `GOETHE_ARCH=x86_64 packaging/build_bin.sh` | `target-portable/<triple>/release`. `GOETHE_CARGO_FEATURES=goethe/test-release-key` builds test binaries for the VM harness. |
| Goethe's nginx for the host architecture | `packaging/build_nginx_linux.sh` | |
| macOS (on an Apple Silicon Mac) | `packaging/build_macos.sh` | `dist/cdn/bin/darwin-aarch64` |
| Whole package | `packaging/build_cdn.sh [--arch x86_64] [--release] [--with-ternary]` | `dist/cdn` |

`build_cdn.sh` needs docker, uv, zstd, rsync and `SGLANG_FORK_SRC`, a checkout of the commit in `sglang/PIN`. It also needs one of three signing setups:

- **Unsigned:** `GOETHE_ALLOW_UNSIGNED=1`.
- **Signed with a dev key:** `GOETHE_SIGN_KEY=<dev key>` plus `GOETHE_RELEASE_PUB=<dev public key>`, so `install.sh` embeds the dev key.
- **Release (`--release`):** `GOETHE_RELEASE_PUB_FPR`, plus `GOETHE_SIGN_KEY` or `GOETHE_ALLOW_UNSIGNED=1` (release CI builds unsigned and signs in `merge`). It refuses `GOETHE_RELEASE_PUB` and always embeds `packaging/release_pub.pem`.

To install from a local package tree:

```bash
cd dist/cdn && python3 -m http.server 8080 --bind 127.0.0.1 &
GOETHE_PKG_URL=http://127.0.0.1:8080 bash dist/cdn/install.sh
```

Binaries that are not `test-release-key` builds trust only the production key, so a package signed with a dev key installs only with test binaries.

## Running a staged dev home

Never point a development goethed at a real install: one home has one daemon, and an extra daemon on the same ports breaks both. Give the dev build its own home, ports and GPUs:

```bash
export GOETHE_HOME=$HOME/goethe-staged/home                 # its own home; a home in the repo goes under .work/ (git-ignored)
export GOETHE_MUX_PORT=30180 GOETHE_PUBLIC_PORT=38480 GOETHE_ENGINE_PORTS=31070-31079
export GOETHE_NO_AUTOSTART=1                                # never install a boot unit for it
export NGINX_BIN=/path/to/goethe-nginx                      # when the home has no nginx-root yet
unset GOETHE_PLATFORM_URL                                    # or point it at a dev console
export PATH=$GOETHE_HOME/bin:$PATH
mkdir -p $GOETHE_HOME/bin
cp target/release/goethe target/release/goethed $GOETHE_HOME/bin/
goethe up
goethe gpu deny gpu0                                        # keep it off GPUs that other work uses
```

Keep these in an `env.sh` you `source`.

- **One goethed per staged home.** Each home has its own `goethed.lock` and socket.
- **Stopping it.** Use `goethe down`, never kill by port or process name. Other software on a shared host may run as the same user.

## Tests

| What | Command | Needs |
|---|---|---|
| Unit tests | `cargo test --locked` (or `-p goethe-core -p goethe -p goethed`) | nothing; they use temp homes, fake servers and stub processes |
| Clippy | `cargo clippy --locked --all-targets -- -D warnings` | |
| Style lint | `python3 tests/lint_style.py` | `ruff` or `uv` on PATH |
| Gate stages | `bash tests/gate.sh <0..6\|all> [--strict] [--only <file>]` | see below |
| CLI end to end, no GPU | `cargo build --locked --features goethe/test-release-key && python3 tests/e2e_cli.py FreshUser` (or `Matrix`) | nginx on PATH |
| Mutants | `bash tests/mutants/run.sh [NAME]` | as e2e; takes up to an hour |
| Installer in VMs | `bash tests/vm/run.sh` | docker, `/dev/kvm`, test-key portable binaries and Goethe's nginx (the script prints the build commands); `DISTROS`, `KEEP=1`, `UV_BIN` |
| macOS | `bash tests/macos_e2e.sh` | an Apple Silicon Mac and `build_macos.sh` output |
| VPN interface | `bash tests/net/run.sh` | docker with `NET_ADMIN` |
| GPU acceptance | `bash tests/accept.sh [--json] [--fresh] [--ux] [MODEL[:QUANT]...]` | a GPU machine; never in CI |
| Mixed load | `python3 tests/load/mixed_load.py` | a running gateway with a model |
| Python suite (lab only) | `python3 -m unittest discover -s tests -t .` from the repo root | Python 3.11+ |

### Gate stages

`gate.sh` builds the binaries (release by default, `GOETHE_GATE_PROFILE=debug` for the dev build) and runs cargo test, clippy and the style lint (`GOETHE_GATE_CARGO_CHECKS=0` skips those three). It then runs each `stage<N>_*.py`:

- **Isolation.** Each test gets its own session and a timeout (`GOETHE_TEST_TIMEOUT`, default 2400 s), and may reap only the processes it started.
- **Failing results.** A pending test, a timeout or zero collected tests fail the run. `--strict` also fails on skips.
- **Environment.** It unsets the port and package variables, sets `GOETHE_NO_AUTOSTART=1`, and gives each test a temp home.
- **GPUs.** `GOETHE_TEST_GPUS` picks the GPUs that GPU tests may use, and it **defaults to `2,3`**. Always set it explicitly, to an empty string on a machine without spare GPUs, as CI does.
- **Real models.** `GOETHE_TEST_REAL=1` enables the tests that run real models (`stage1_e2e_model`, `stage4_hw`, `stage4_recovery_hw`). `GOETHE_DEV_HOME` and `GOETHE_DEV_PY` name the dev py-env that stage 1 uses.

CI runs `gate.sh 0` and `gate.sh 6 --only stage6_integrity.py` with no GPU.

### Writing tests that need no GPU

- **Rust.** Put the decision in a pure function and test that function. Examples are `watchdog_policy`, `calc::missing_gb` and `rebuilt_reply` in the mux. For the rest:
  - Use `tempfile` homes with `GOETHE_HOME`.
  - Use a local TCP listener for an engine or the console; see `fake_engine` in `mux.rs` and the `mock` server in `heartbeat.rs`.
  - Use `sleep` children for process tests.
  - Environment variables are process-wide, so give each test a unique variable name or avoid them.
- **Python.** Use `tests/lib/goethe_test.py` for homes, ports and cleanup, `tests/lib/stub_engine.py` as a fake SGLang, and `tests/e2e/world.py` for a fake package host, console, Hub mirror, packs and driver (`nvml_shim.c`, `cuda_shim.c`).
- **Every fix gets a test** that fails without it. The mutants in `tests/mutants/` keep the end-to-end cells honest: each mutant brings back a bug that shipped once.

## Style rules

`tests/lint_style.py` (run from `goethe/`) is the CI style gate.

**Comments and headers:**

- Every source file (`.rs`, `.py`, `.sh`, `.toml`, `.conf`, `.tmpl`, `.service`, `.sql`, `.cu`, `.cuh`, `.cpp`, `.h`, Dockerfile) starts with a **one-line header** comment. A shebang may come before it, and the header must not continue onto a second comment line.
- **Rust, C and shell:** a comment is allowed only as the header, or as the **first line inside a function body**. There are no inline trailing comments and no `/* */` blocks. A `//` after a `:` (as in a URL) is not a comment.
- **Python:** a one-line module docstring is required, and docstrings must be one line. There are no `#` comments at all (except a shebang) and no stray string statements used as comments.

**Characters and lines:**

- No emoji and no en or em dash (or similar dash characters) in any source, Markdown, patch, JSON, lock or text file.
- Python lines are at most 120 columns.

**Ruff:** Python is checked with `ruff 0.17.0` (rules E, F, W, B, UP, I, line length 120). Run it through `lint_style.py`; a bare `ruff --fix` without these settings sorts imports wrongly.

**Leak guard and test key:**

- No tracked file may contain internal host names, local machine paths (the patterns are in `LEAKS` in `lint_style.py`; `/home/<name>` is allowed only for placeholders such as `/home/user`), tailnet addresses, or an AI co-author line.
- `PROJECT.md`, `.local/` and the agent skill folders must never be tracked.
- `test-release-key` must not be a default feature, and no packaging script, `cdn/install.sh` or `release.yml` may pass it.

CI also runs `ruff` over `lab` and `tests`, `shellcheck -S warning` over several scripts, and `bash -n` over the installer and packaging scripts.

## CI

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | push and PR to `main` and `dev`, nightly, manual | The `changes` job decides from the changed paths which jobs run. Jobs: `python` (lab suite on 3.11 and 3.13); `lint`; `attribution` (fails on an AI co-author or "Generated with" line in new commits); `ninfer-patch`; `engine-test` (cargo test, clippy, test-key build, gate 0 and stage 6); `e2e-user` (`e2e_cli.py FreshUser` and `Matrix`). `gate` is the one required check: every job passed or was skipped for a false flag, and every job except `package-smoke` and `mutants` is listed in its `needs`. Main, nightly and manual runs add `package-smoke`: an unsigned x86_64 package, the portable binary runs, and it refuses a foreign key set in its environment. Nightly and manual runs add `mutants`. |
| `installer.yml` | PRs that touch the installer, setup, fix, update, service, nginx or release-key code, nginx packaging, or `tests/vm` | The `curl \| bash` install in a Rocky 9 VM: root and user installs, reinstall, upgrade, tampering, pinning, uninstall. |
| `macos.yml` | PRs that touch the crates, packaging, installer, llama.cpp pin; `mac-support` and `mac/**` branches | Unit tests and clippy on macOS, package build and `macos_e2e.sh`. |
| `wsl-install.yml` | PRs to `main` and `dev` that touch `cdn/install.ps1` or `cdn/install.sh`, manual | The Windows install path: `install.ps1` on a Windows runner, then `install.sh` inside WSL2 Ubuntu. |
| `engine-pin.yml` | daily, manual | Opens a PR to `dev` that moves `goethe-engine/PIN` to the newest release. |
| `release.yml` | tag `vX.Y.Z` on `main`, manual | The release (below). |
| `pages.yml` | manual, with a tag | Serves an already published release on Pages again. Use it when `publish` succeeded but `pages` failed. |

When you add a job to `ci.yml`, add it to `gate.needs`, or `gate` fails.

## Release

Releases come from GitHub Actions on `main`, never from a laptop. The maintainer runs the merge and tag commands.

1. `goethe/packaging/release.sh prepare X.Y.Z [title]`, from the repo root. On a fresh `release-X.Y.Z` branch from `origin/main`, it sets the version in the three crates and `Cargo.lock`, dates the `## Unreleased` changelog section, checks the build, commits, pushes, and opens the PR.
2. Merge that PR.
3. `goethe/packaging/release.sh tag X.Y.Z` tags `origin/main` as `vX.Y.Z` and pushes the tag. The crate version must match.
4. `release.yml` runs:
   1. `guard`: the commit is on `main` and the tag equals the version in `crates/goethe/Cargo.toml`.
   2. `ci`: runs CI unless a push run already passed on that commit.
   3. In parallel: `build` (the x86_64 package with `build_cdn.sh --release`), `build-aarch64`, `nginx-aarch64` and `build-macos` (build plus e2e).
   4. `merge`: waits for approval on the `release` environment, the only job that sees the signing key. It makes one package; `SHA256SUMS` and the registry signature recomputed and re-signed, then verified with the key embedded in `install.sh`, with the fingerprint and `BUILDINFO` checked.
   5. `install-smoke`: an install in a clean Ubuntu 22.04 container.
   6. `publish`: creates the GitHub Release and uploads the signed package, `install.sh` and `install.ps1`.
   7. `pages`: deploys `pkg.goethe.sh`.
5. Existing installs update with `goethe update`.

### Keys

- **Generating:** `packaging/keygen.sh <dir>` creates a P-256 `release.key` (mode 600) and `release_pub.pem`, and refuses to overwrite either.
- **Repository settings:** the `release` environment's secret `GOETHE_SIGN_KEY` holds the private key (not a repository secret). The variable `GOETHE_RELEASE_PUB_FPR` holds the sha256 of the DER public key, the one `packaging/release_pub.pem` must match.
- **Losing the key** means existing installs cannot verify new releases. A key rollover is a new binary that carries the new key, published in a package list signed by the old key.

## Recipes

### Adding a CLI command

1. Add the variant to `Commands` (or a subcommand enum) in `crates/goethe/src/main.rs`, with `about` and `help` text, and dispatch it to a function in `goethe-core`. Keep `main.rs` thin.
2. Add a row to `HELP_SECTIONS`. A test fails when a visible command is missing from `goethe help`, and another fails when a help example does not parse.
3. If the command takes model ids, keys or other live values, wire completion in `crates/goethe/src/complete.rs`.
4. Exit codes: 0 for success, 1 for an error, 2 for usage (clap), 3 when foreign GPU processes need confirmation.
5. Document the command in `docs/cli.md`, and in `docs/user-guide.md` if users need it.

### Adding an RPC method

1. Add a match arm in `dispatch` in `crates/goethed/src/main.rs`. It returns `(ok, code, message, error, result)`. Work runs on a blocking thread; long work must not hold the plan lock longer than needed.
2. Call it from the CLI with `rpc::call("<method>", params)`. In `rpc::call`, decide whether the method may be re-sent after an I/O error. `load` and `unload` may not be; anything with side effects should not be.
3. The socket is owner-only (0600), so the caller is the same user. Still validate `params` as untrusted JSON.
4. Add a test for the method in the supervisor, plus a stage 4 stub test if it changes process state.

### Adding an engine

1. Create `crates/core/src/engines/<name>.rs` with `pub const ENGINE: Engine { id, overhead_gb, layer, params, command }`, and add it to `ENGINES` in `engines/mod.rs`. A card selects it with `runtime = "<id>"`.
2. The command builder must:
   - start from `env_clear()` plus `kept_env()`;
   - apply card env before `placement_env`, so a card cannot move the engine to other GPUs or retag it;
   - pass the engine key with `engine_key_env`, never on argv, because argv is readable by every local user;
   - bind 127.0.0.1 only.
3. List the request fields the engine honours in `params`; anything else is refused with `unsupported_parameter` before it reaches the engine.
4. Teach `install.rs` when to choose the engine, and pin its binary or wheel with a sha256 ([packages.md](packages.md)).
5. Add the runtime id to the console's `RUNTIMES` list in the goethe-web snapshot validator. Unknown runtimes are shown as null there.

### Adding a model to the registry

```bash
goethe registry add --repo org/model --file cdn/v1/registry.json [--overlay hand-written.json] [--dry-run]
goethe registry validate --source cdn/v1/registry.json --online --check-sglang
```

- **What `add` does:** it measures size, KV elements, heads, context and the memory spec from Hugging Face, merges the overlay, validates, and writes atomically.
- **Gated repos** are written by hand with `"gated": true`.
- **Vetting:** then vet the model on real GPUs with the checklist in `docs/registry.md`. That covers default loads, the plan against the engine log, unload and reload, the first request, and `tests/load/mixed_load.py`.
- **Shipping:** the file is signed at package build (`v1/registry.json.sig`) and compiled into the binary as a fallback.

### Adding a heartbeat field

The heartbeat payload is a contract with the console, which lives in the separate goethe-web repository (`lib/platform/snapshot.js`).

1. Add the field to the structs in `crates/core/src/snapshot.rs`.
   - Build it from local state only.
   - Bound every string (`clean_text` with a length) and every list (the `MAX_*` caps).
   - Never include a token, key, prompt or path that names a user.
2. Make sure the encoded snapshot still fits in 256 KB with full lists. `encode_sent` sheds data in a fixed order; add the new field to that order if it can be large.
3. Change the console validator in the same release window.
   - The console drops unknown fields, so a new field is harmless until the console reads it.
   - The console validates enums strictly. A new value of an existing enum, such as a new instance status, makes the console reject the **whole** heartbeat with 400 until the console knows it.
4. Add a test in `snapshot.rs`, and in `heartbeat.rs` with the mock console if sending behavior changes.

### Updating pins

See [packages.md](packages.md#bumping-each-engine-pin) for SGLang, goethe-engine, llama.cpp, NInfer, nginx, uv and the Python locks.
