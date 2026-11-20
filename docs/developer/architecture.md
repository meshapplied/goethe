# Architecture

How the parts of Goethe fit together at runtime. Module names refer to `goethe/crates/core/src/` unless another crate is named. [repo-map.md](repo-map.md) lists every module.

## Processes

| Process | Binary or command | Started by | Listens on |
|---|---|---|---|
| CLI and menu | `goethe` (`crates/goethe`) | the user | nothing |
| Daemon | `goethed` (`crates/goethed`) | `goethe up`, the boot unit, or setup and update after a swap | `GOETHE_SOCK`, the mux port (127.0.0.1, default 30000), and the public-share mux address while a share window is open |
| Engines | SGLang, `python -m goethe_engine serve`, `llama-server`, `ninfer-serve` | goethed (`supervisor.rs` with `engines/*`) | 127.0.0.1, one port each from `GOETHE_ENGINE_PORTS` (default 30001-30099) |
| Gateway | Goethe's own nginx (`nginx.rs`) | goethed at start; `goethe api expose` and `api allow` reload it | the public port (default 8443) on the chosen bind address; the share port while a window is open |

- **One daemon per home.** `goethed` takes an exclusive `flock` on `goethed.lock` (`goethed/instance_lock.rs`). Only the holder may remove a stale socket.
- **Socket location.** The socket is `$GOETHE_HOME/goethe.sock` (mode 0600). For root it is `/var/run/goethe.sock`, unless `GOETHE_SOCK` is set.
- **Startup.** On start, goethed does the following in order:
  1. resolves ports (`ports.rs`);
  2. writes the hardware profile when it is missing or older than this boot;
  3. adopts or stops leftover engines (`supervisor::reconcile_on_start`);
  4. writes the nginx key map;
  5. binds the socket and the mux;
  6. starts nginx, the autostart restore, the heartbeat task and the usage-log sampler.
- **Main loop.** It runs the watchdog every 5 s, follows network interface changes and public-share expiry every 2 s, and rotates logs hourly.

### The RPC

The `goethe` to `goethed` protocol is JSON lines over the Unix socket. Each line holds one `{"method", "params"}` request and gets one `{"ok", "code", "message", "error", "result"}` reply.

- **Methods:** `load`, `unload`, `ps`, `platform_changed`, `shutdown` (with `keep_engines`). The server side is in `goethed/src/main.rs` and the client in `rpc.rs`.
- **Retries:** `load` and `unload` are never re-sent after an I/O error; the client tells the user to check `goethe ps`.
- **Blocking:** each connection is served on a blocking thread, and a `load` can block for minutes behind the per-GPU startup gate.

## GOETHE_HOME

**Resolution** (`paths::goethe_home`):

1. `GOETHE_HOME`;
2. else the home whose `bin/` holds the running binary (links followed, the folder must contain the `.goethe-home` marker);
3. else `/workspace/.goethe` when it exists or `/workspace` is writable (RunPod, Vast.ai);
4. else `/var/lib/goethe` when it exists and is writable;
5. else `~/.goethe`.

On macOS the home is `~/.goethe`.

**Settings.** `paths::setting` reads a few settings from the environment first, then from `goethe.env`: `GOETHE_PKG_URL`, `GOETHE_REGISTRY_URL` and `GOETHE_PKG_TOKEN`. Every other setting, `GOETHE_PLATFORM_URL` included, comes from the environment only.

| Path under the home | What |
|---|---|
| `.goethe-home` | Marker that proves the folder is a Goethe home. `uninstall` refuses to delete a folder without it. |
| `state.db` (+ `-wal`, `-shm`) | SQLite in WAL mode, mode 0600. See below. |
| `bin/` | `goethe`, `goethed`, `uv`; `.prev` copies kept by `update` and `setup` for rollback. |
| `nginx-root/usr/sbin/nginx` | Goethe's own nginx binary. |
| `nginx/` | nginx prefix: `nginx.conf`, `goethe_gateway.conf`, `allowlist.conf`, `api_keys.conf`, `certs/`, `logs/`, `nginx.pid`, `listen_ip`, `public_share.json`. It also holds one-value files: `mux_port`, `public_port`, `max_body_mb`, `rate_per_ip`, `conn_per_ip`, `share_rate_per_ip`, `share_conn_per_ip`. |
| `models/<org>__<name>/` | Installed weights plus `goethe_card.toml`, the model's card (runtime, defaults, engine flags). |
| `py-env/` | Hermetic Python 3.11 for the hf CLI, goethe-engine and SGLang ([packages.md](packages.md#python-py-env)). |
| `engines/llama.cpp/` | The pinned llama.cpp release, downloaded on first use. |
| `packs/` | Kernel-pack cache and the last accepted index `serial` (rollback guard). |
| `cuda-compat/` | NVIDIA forward-compatibility libraries, used only on drivers 535 to 579. |
| `logs/` | `goethed.log`, `gateway.log`, `<instance_id>.log` per engine, `index.jsonl` (which log belongs to which model). |
| `run/` | `<instance_id>.json` identity of every spawned engine (`run_registry.rs`), so a restarted goethed can find and reap strays; `pyenv.lock`. |
| `platform/` | `credentials` (node token, 0600) and `platform_state.json` once `goethe login` linked the node to the console. |
| `traces/` | Trace recorder output (`docs/traces.md`), when tracing is on. |
| `learn/sessions/` | In-session learning notes (`docs/learn.md`), when learning is on. |
| `hardware_state.json` | Hardware profile written by `hw.rs` (GPUs, links, host). |
| `registry.cache.json` | Last verified registry. |
| `goethe.env` | `KEY=VALUE` written by the installer, only for non-default settings. |
| `VERSION` | Installed release. |
| `goethe.sock`, `goethed.lock` | Daemon socket and instance lock (non-root). |

### state.db

One schema created by `db.rs`, version 1. New columns and tables are added in `db::ensure_columns` or `usage_log::ensure_tables` (called from `db.rs`) without a version bump, so an older Goethe still opens a newer database.

| Table | Holds |
|---|---|
| `installed_models` | Every registered model: path, architecture, sizes, defaults, `managed` (Goethe owns the files) or not. |
| `loaded_instances` | Every engine instance: pid, port, GPUs, status, slot key, identity (`boot_id`, `start_ticks`), engine key. |
| `desired_loads` | What should be running after a restart, per slot; written when an instance reaches RUNNING, deleted only by unload. |
| `api_keys` | Keys: id, name, sha256 of the token, prefix, rpm, expiry, model scope, admin. The `token` column keeps the raw token so the public share can reprint it. |
| `key_usage` | Requests and tokens per key and UTC day. |
| `ip_allowlist` | CIDRs the gateway admits. |
| `gpu_policy` | Denied GPU indices. |
| `model_aliases` | Alias to model id. |
| `goethe_settings` | Key/value settings: trace and learn config, `restore_desired`, the usage-log send cursor `usage_sent_t`, and others. |
| `watchdog_restarts`, `watchdog_probe_state` | Restart budget per slot and liveness probe state per instance. |
| `engine_overhead` | Learned extra VRAM per model, GPU name and TP. |
| `usage_minute`, `usage_hour`, `keyuse_*`, `gpu_*`, `host_*` | The usage log (below). |

## Loading a model

The supervisor and coordinator handle this. The flow, from a `load` RPC to a RUNNING row:

1. **Validate.** `supervisor::handle_load` checks that the model is installed and has a readable card, then checks the extra engine arguments, the pinned GPUs and the denied set.
2. **Queue.** It takes a cancel ticket and loops. Under the global plan lock (`coordinator.rs`), `plan_and_start` does the following:
   - checks for a live copy of the model; "already loaded" is reported, `--replica` starts another;
   - snapshots NVML;
   - plans the TP size and GPU set (`chooser.rs` with `calc.rs`, `memory_spec.rs`, `model_facts.rs`, `placement.rs`), shrinking the context, then the concurrency;
   - lists foreign processes (exit 3 without `--yes`);
   - checks the per-GPU startup gate: one engine starts on a GPU at a time.
3. **Start.** `start_engine` picks a port and slot, spawns the engine in its own session, records its identity (`proc_ident::capture`, `run_registry`), inserts a STARTING row and reserves the VRAM in the ledger.
4. **Wait for ready.** A startup thread polls the port, the leader pid, `/health` and `/v1/models`, sends a warm-up request, then marks the instance RUNNING and writes `desired_loads` in one transaction.
   - A VRAM shortfall at startup refits and respawns up to 2 times.
   - A port that is taken moves to another port up to 3 times.
   - A stall is judged by progress (`startup_stall.rs`), not by a fixed clock.

**Status values in `loaded_instances`:**

- STARTING moves to RUNNING, or to FAILED or CANCELLED (kept as tombstones with pid 0 so `ps` shows them).
- RUNNING moves to STOPPING (watchdog claim), then to RESTARTING (with backoff) or FAILED.
- Unload goes DRAINING, then STOPPING, then the row is deleted. New requests to a DRAINING instance get 503. The drain lasts up to `GOETHE_DRAIN_SECS`, default 120 s.

**Watchdog** (`watchdog_liveness.rs`, `watchdog_policy.rs`): a cheap probe and a periodic deep probe, at most 3 restarts per 600 s per slot, backoff 5, 15, then 45 s.

**Adoption.** `goethe down --keep-engines` and upgrades leave engines running. The new goethed adopts every RUNNING engine whose identity verifies and that answers `/v1/models`.

## The request path

1. **nginx** (`goethe/nginx/goethe_gateway.conf.tmpl`): TLS, `include allowlist.conf; deny all;`, a per-IP request rate and connection cap, and a body size cap. It answers `/metrics` with 404 and proxies everything else (the health routes included) to the mux, setting `X-Real-IP`.
2. **Mux** (`goethed/src/mux.rs`):
   1. Route and method check; the same routes answer with and without `/v1`.
   2. Key from `Authorization: Bearer`, `x-litellm-api-key`, `x-api-key` or `api-key`, hashed and looked up in `GatewayCache` (an in-memory copy of state.db reloaded when `PRAGMA data_version` changes); expired keys are refused.
   3. On the public-share listener only, `public_share::check`.
   4. Per-key GCRA rate limit (`--rpm`, default 120).
3. **Proxy** (`route_post`):
   1. Read the body, capped.
   2. Resolve the model: a real id wins over an alias of the same name.
   3. Scope check, then take an in-flight guard (`inflight.rs`). The guard is what unload drains on.
   4. Pick the least busy non-draining instance.
   5. Refuse parameters the engine does not support (`engines::check_params`).
   6. Optionally splice in a learn note.
   7. Forward with the engine's own key. The client's key never reaches an engine.
4. **Metering** (`MeteredBody`):
   - Counts prompt and completion tokens from reported usage, or counts output chunks.
   - Measures time to first token.
   - Records exactly once, at end of stream or when the client goes away (cancelled), into Prometheus metrics, the usage log, key usage and the trace recorder.
   - A stream the engine cuts ends with an error event in the API's shape.
5. **Rebuilt replies.** For a client that did not stream, the engine stream is read whole (capped) and assembled into one body (`trace_assemble.rs`). An error event inside the stream becomes an error reply with a matching status, not a 200.

**Endpoints without a key:**

- `/health`, `/health/ready?model=` and the LiteLLM health spellings.
- `/metrics` (Prometheus) is served on the loopback mux only.
- `/health/admin` needs an admin key; everyone else gets 404.

## Console link, heartbeat, usage log

- **Link.** `goethe login` runs a device flow against the console (`platform_api.rs`, default `https://goethe.sh`, `GOETHE_PLATFORM_URL` to change it) and stores a node token in `platform/credentials`. Everything here is optional; an unlinked node sends nothing.
- **Heartbeat** (`heartbeat.rs`, `goethed/src/heartbeat.rs`):
  - Every 10 s, sooner after a change (debounced 1 s, at least 2 s apart), and backed off on failure.
  - Each beat builds a `StateSnapshot` (`snapshot.rs`, `goethe.snapshot.v1`) from local state only: node, network addresses, GPUs, installed and loaded models, the key inventory without tokens, and new usage rows.
  - The encoded snapshot must fit in 256 KB. Oversize snapshots drop the installed list first, then halve the usage rows, then drop everything but the node.
  - The console's validator is the contract ([workflows.md](workflows.md#adding-a-heartbeat-field)).
- **Usage log** (`usage_log.rs`). It runs for linked and unlinked nodes alike.
  - Per-model and per-key request counts, token counts, latency and TTFT histograms, plus 5 s GPU and host samples (10 s on a Mac).
  - Kept as clock-minute rows for 48 h and hourly rows for 90 days. The data is counts and timings only, never text.
  - A linked node sends the minutes after the `usage_sent_t` cursor with its heartbeat, about 400 rows per beat. The cursor moves only after the console accepted the beat. Per-key rows are never sent.
  - `goethe metrics` and the Metrics menu page read the same tables.

## Invariants

- **Signals.** Every signal to an engine goes through `proc_ident::signal`: the pid is pinned with a pidfd, the identity (boot id and start ticks, or the instance env tag) is verified, then the signal is sent. Goethe never signals a process it did not start.
- **Keys.** Engines listen on loopback with a per-instance random key, and the gateway is the only keyed path in. `goethe run` talks to the engine directly with the engine key from state.db.
- **Desired loads.** `desired_loads` changes only at RUNNING (insert) and unload (delete). Restore after a reboot replays it oldest first, one engine per GPU at a time.
- **Unload races.** `mark_running` only moves a row from STARTING, so an unload always wins a race with a finishing start.
- **Public share fails closed.** No open window, an admin key, an unlisted key, an expired grant or a client outside the grant's CIDRs is refused, and the nginx block always ends in `deny all`.
- **Release signatures.** Every check against the release key goes through `release_key::verify` with the key compiled into the binary ([security.md](security.md)).

## Gotchas

- **Hidden streaming.** Every generation request is streamed from the engine even when the client sent `stream: false`. Debugging engine behavior on non-stream requests means looking at the rebuild in the mux.
- **Unknown model names.** `Snapshot::resolve_model` returns client text unchanged when it is unknown. Never use it as a map key or a path without `is_known` first; metrics and the usage log use the label `_unknown`.
- **Blocking work.** A `load` RPC blocks its socket thread. The watchdog tick takes a non-blocking lock and skips if the previous tick is still running.
- **Tombstones.** FAILED and CANCELLED rows keep their port until the slot is reloaded or unloaded.
- **Restart without `goethe down`.** Healthy engines keep serving and are adopted. Unverifiable rows are deleted without any signal, and the run registry reaps strays later.
- **macOS.** Only llama.cpp engines run there (one Metal GPU, unified memory limited by `iogpu.wired_limit_mb`); there is no NVML, no pidfd and no systemd. `sys.rs` hides the Linux and macOS differences.
