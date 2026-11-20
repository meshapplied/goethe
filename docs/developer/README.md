# Goethe developer documentation

For people who change Goethe itself: the `goethe` CLI, the `goethed` daemon, the gateway, packaging and releases. To install and use Goethe, read the [user guide](../user-guide.md) and the [CLI reference](../cli.md) instead.

| File | Read it when |
|---|---|
| [architecture.md](architecture.md) | You need to know which process does what, where state lives, and how a request travels. |
| [repo-map.md](repo-map.md) | You are looking for the code behind a feature, or wondering whether a folder is product at all. |
| [packages.md](packages.md) | You touch a dependency, a pin (SGLang, llama.cpp, goethe-engine, NInfer, nginx, uv, Python locks) or the package layout. |
| [workflows.md](workflows.md) | You build, test, lint, release, or add a command, RPC, engine, registry model or heartbeat field. |
| [security.md](security.md) | You touch keys, signatures, the gateway, the public share, file permissions or anything that leaves the machine. |
| [best-practices.md](best-practices.md) | Before your first change, and as the checklist before every pull request. |
| [audit1.md](audit1.md) | You want the findings of the November 2026 audit: what was fixed, the open risks and what to do about them. |

The rules agents and humans follow in this repo are in [AGENTS.md](../../AGENTS.md). Where this documentation and the code disagree, the code wins; fix the documentation in the same pull request.

## Five-minute orientation

**What Goethe is.** A machine-local control plane that turns GPUs (or an Apple Silicon Mac) into one HTTPS API that speaks the OpenAI, Anthropic and LiteLLM schemas. It installs models, starts inference engines on the right GPUs, and puts an authenticated, rate-limited gateway in front of them. Everything product lives under `goethe/`. It is written in Rust (three crates), plus shell for packaging and Python for the integration tests.

**The processes.**

- `goethe`: the CLI and the interactive menu. It does local work itself (install, scan, doctor, update, keys) and asks the daemon for anything that needs a running process (load, unload, ps, shutdown), over a JSON-lines Unix socket.
- `goethed`: the daemon, one per Goethe home. It supervises engines (spawn, readiness, watchdog, drain, adoption after a restart). It also runs the mux, the HTTP API server on 127.0.0.1 that authenticates keys and proxies to engines, and it sends the optional heartbeat to the console.
- Engines: one process per loaded model instance, started by goethed on a loopback port with its own random API key. These are SGLang, goethe-engine (signed kernel packs), llama.cpp (GGUF) and NInfer.
- nginx: Goethe's own build, run by goethed. It terminates TLS on the public port (8443 by default), enforces the IP allowlist and the per-IP limits, and forwards to the mux. While a public share window is open it also serves a second listener.

**How a request flows.**

```
client --HTTPS--> nginx :8443 (allowlist, per-IP rate) --> mux 127.0.0.1:30000 (key auth, scope, rpm, model routing)
       --> engine 127.0.0.1:3000x (engine key) --> response metered (tokens, TTFT) --> usage log, metrics, optional trace
```

The mux streams every generation request from the engine, even when the client did not ask for a stream. A non-streamed request is rebuilt into one JSON body, so time to first token and decode speed are always measured the same way.

**Where state lives.** Everything is under one home directory, `GOETHE_HOME` (see [architecture.md](architecture.md#goethe_home)). The SQLite database `state.db` is the source of truth for installed models, instances, keys and settings.

**First commands.**

```bash
cd goethe
cargo build                                    # target/debug/goethe, goethed
cargo test -p goethe-core -p goethe -p goethed
cargo clippy --locked --all-targets -- -D warnings
python3 tests/lint_style.py                    # style, leak guard, ruff
```

Do not start a dev goethed against your real home. Use a staged home ([workflows.md](workflows.md#running-a-staged-dev-home)).
