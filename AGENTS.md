# AGENTS

How to work on Goethe with a coding agent: what is where, the rules a change must follow, and how to check it.

## Layout
- `goethe/crates/goethe`: the `goethe` CLI (commands, menu, `goethe ask`).
- `goethe/crates/goethed`: the daemon (engine supervisor, gateway mux, heartbeat).
- `goethe/crates/core`: the shared library (install, models, engines, packs, console link, doctor, usage and error logs).
- `goethe/crates/proto`: the wire contract between a node, goethe-server and the console (state events, signed commands, sealed secrets).
- `goethe/cdn`: `install.sh` (Linux, macOS, WSL) and `install.ps1` (Windows: enables WSL2, then runs `install.sh`; never duplicate install logic in PowerShell).
- `goethe/packaging`: release build, signing, container image, ask model build.
- `goethe/nginx`, `goethe/sglang`, `goethe/llama-cpp`, `goethe/goethe-engine`, `goethe/pyenv`: gateway templates, engine pins and patches, Python environment layers.
- `goethe/tests`: style lint, gates, end-to-end harness.
- `docs`: user guide, CLI reference (`cli.md`; `cli.json` is generated), learn, traces, registry.

## Rules
- Fail closed: allowlists, packs, the release key, public sharing and signed commands refuse when unsure. Silent success is a bug.
- Smallest change that fixes the real problem; no speculative frameworks or abstractions.
- Pure logic (fit, placement, budgets, key and share policy, pack and command checks) stays free of I/O and has unit tests; process spawning, nginx, Hugging Face, the database and menus are the shell around it.
- Pack and release signatures are trust anchors: the release key is compiled in and never read from the environment in a release build.
- Never store, log or print an API key secret; keys are kept as hashes and a secret is shown once.
- Prompts and replies are never written to disk unless the user turned tracing on for that key.
- Keep modules around 400 to 500 lines; split before a file grows further.
- Follow `goethe/tests/lint_style.py`: a header comment in every file, a one-line comment as the first line of each non-obvious function, no long dashes or emojis, no AI co-author trailers. CI runs it.
- Commit only when asked; never force-push `main` or `dev`.

## Check a change
- `cd goethe && cargo test --workspace --locked`
- `cd goethe && cargo clippy --workspace --locked --all-targets -- -D warnings`
- `cd goethe && python3 tests/lint_style.py`
- A single area: `cd goethe && cargo test -p goethe-core --lib -- <filter>`
- `bash goethe/tests/gate.sh <n>` for a stage (no GPU needed).
- Regenerate `docs/cli.json` after a CLI change: `GOETHE_WRITE_CLI_JSON=1 cargo test -p goethe cli_json`.
- Add a test for each new rule and break the rule once to watch it fail.
