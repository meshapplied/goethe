# AGENTS

What is in this repository and how to check a change. Read `../AGENTS.md` first when it exists.

## Layout
- `goethe/crates/goethe`: the `goethe` CLI (commands, menu, `goethe ask`).
- `goethe/crates/goethed`: the daemon (engine supervisor, gateway mux, heartbeat).
- `goethe/crates/core`: the shared library (install, models, engines, packs, console link, doctor, usage and error logs).
- `goethe/crates/proto`: the wire contract between a node, the platform and the console.
- `goethe/cdn`: `install.sh` (Linux, macOS, WSL) and `install.ps1` (Windows: enables WSL2, then runs `install.sh`).
- `goethe/packaging`: release build, signing, container image, ask model build.
- `goethe/nginx`, `goethe/sglang`, `goethe/llama-cpp`, `goethe/goethe-engine`, `goethe/pyenv`: gateway templates, engine pins and patches, Python environment layers.
- `goethe/tests`: style lint, gates, end-to-end harness.
- `docs`: user guide, CLI reference (`cli.md`; `cli.json` is generated), learn, traces, registry.

## Check a change
- `cd goethe && cargo test --workspace --locked`
- `cd goethe && cargo clippy --workspace --locked --all-targets -- -D warnings`
- `cd goethe && python3 tests/lint_style.py`
- `bash goethe/tests/gate.sh <n>` for a stage (no GPU needed).
- Regenerate `docs/cli.json` after a CLI change: `GOETHE_WRITE_CLI_JSON=1 cargo test -p goethe cli_json`.
