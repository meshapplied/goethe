# Security

The trust model, the rules that keep it, and the risks that are known and still open. Read it before you change keys, signatures, the gateway, the public share, file handling, or anything that leaves the machine.

## Release trust

- **The trust anchor.** `goethe/packaging/release_pub.pem` (P-256) is compiled into every binary by `release_key.rs`. Every release-key signature check goes through `release_key::verify`, and a test enforces that.
  - A key in the home or the environment is never trusted. A release binary fails every release-key signature check while `GOETHE_RELEASE_PUB` is set.
  - Only a build with the `test-release-key` feature reads `GOETHE_RELEASE_PUB`, and `lint_style.py` keeps that feature out of every shipped build path.
- **What is signed.** `SHA256SUMS` lists every file of the package and `SHA256SUMS.sig` signs it. `v1/registry.json` has its own `.sig`.
  - `install.sh` carries the same public key (substituted at build) and verifies the list with `openssl` before it runs any downloaded binary.
  - `goethe setup` and `goethe update` verify the list again and hash every file they stage.
- **Downgrades.** `update` refuses a lower `VERSION` for the binaries unless `--allow-downgrade`.
- **Swaps.** Binaries are swapped with `.prev` copies so a failed update rolls back.
- **Pinned downloads outside the package**, each with a pinned sha256 ([packages.md](packages.md)):
  - the goethe-engine wheel (installed with `--require-hashes`);
  - llama.cpp archives;
  - the NInfer artifact;
  - the CUDA forward-compatibility `.deb`;
  - every Python lock entry.
- **Kernel packs** come from the console with the node token. Trust is decided by goethe-engine, not by goethe: the index signature, rollback by serial (goethe stores the last serial in `packs/serial`), the pack signature and identity, and the architecture.
  - A pack goethe-engine does not accept is not installed.
  - Unsigned packs are refused unless the engine itself allows them, and goethe removes `GOETHE_ENGINE_ALLOW_UNSIGNED` from the engine environment.
- **Model weights** from Hugging Face are checked against the Hub's sha256 for every file.

## Fail closed

A check that cannot complete refuses. Silent success is a bug. Concretely:

- **Unreadable signature.** A signature that is missing, unreadable or made with another key stops the install or update with a message that says what to do.
- **Gateway allowlist.** The gateway denies every address by default: the config is `include allowlist.conf; deny all;`, and an empty allowlist admits nobody.
- **Public share.** It is refused, with a reason code, when no window is open, the key is not in the window, the key is an admin key, the grant has ended, or the client address is not in the grant's CIDRs. Its nginx block always ends in `deny all`, and allow entries are canonicalized before they are written.
- **Pack lookup.** When a kernel pack cannot run here, the plain runtime takes over and goethe says why; it is never silent. `GOETHE_ALLOW_FALLBACK=0` makes it an error, and `--fallback` overrides that. A `pack_only` model with no pack for this GPU is refused before any download.
- **Model scope.** An empty or unparseable model scope admits no model.
- **Foreign GPU processes.** Goethe never stops a process it did not start. A load onto a GPU with foreign processes asks first and exits 3 without `--yes`.

## Gateway, keys and engines

- **Tokens.**
  - Format: `goethe_sk_` plus 128 random bits.
  - Storage: `state.db` stores the sha256 (`token_hash`) and a short prefix for display. The `token` column also keeps the raw token so the public share can reprint it to its owner.
  - Lookup: `GatewayCache` hashes the presented token and looks it up in memory. A miss re-reads the database, so a new key works at once.
  - Revocation: every proxied request checks the exact snapshot, which refuses a key revoked a moment ago.
- **Where the key comes from.** The mux reads `Authorization: Bearer`, `x-litellm-api-key`, `x-api-key` and `api-key`. Admin keys can read `/health/admin`; for anyone else it does not exist (404).
- **Limits.**
  - Per IP in nginx: request rate and connection cap, changed with `GOETHE_GATEWAY_RATE_PER_IP` and `GOETHE_GATEWAY_CONN_PER_IP`.
  - Per key in the mux: GCRA with `--rpm`, default 120, 0 for unlimited.
  - Request bodies are capped by `GOETHE_MAX_BODY_MB` (default 10) in both nginx and the mux.
  - A non-streamed reply is capped at 64 MB while it is rebuilt.
- **Engines.**
  - Every engine listens on 127.0.0.1 with its own random key, and the mux sends that key; the client's key never reaches an engine.
  - New engines must take the key from the environment, never from argv ([workflows.md](workflows.md#adding-an-engine)).
  - Engines get a cleared environment (`env_clear` plus an allowlist), and the placement variables are set last.
- **What the loopback-only endpoints show.** `/metrics` is served on the loopback mux only and is never exposed through nginx. It still shows model names and key ids to local users. `/health/ready` is reachable through the gateway without a key and tells the caller which models are serving.

## Processes

- **Signals.** Every signal to an engine goes through `proc_ident::signal`: pidfd first, then identity verification (boot id and start ticks, or the `GOETHE_INSTANCE_ID` env tag), then `pidfd_send_signal`. A reused pid is never signaled.
- **Daemon socket.** `goethe.sock` is created owner-only (umask during bind, then chmod 0600; failing that, goethed refuses to serve). The socket has no peer-credential check, so any process running as the same user can drive goethed. That is the intended boundary.

## Files

- **The home** is chmod 0700 by setup. Private files are written through `nginx::write_private` or `platform::write_private`, two near-identical helpers: a temp file with `create_new` at mode 0600, fsync, then rename. A symlink at the destination is replaced, not followed.
  - Files written this way include certificates and keys, `public_share.json`, the allowlist, the port and limit files, the credentials and `goethe.env`.
  - `state.db` is mode 0600.
- **Nginx config from user input.** Only canonicalized CIDRs, integers and quoted paths are interpolated into it.
- **Paths from user input** go through validation:
  - Model ids go through `paths::validate_model_id`, and model folders through `contained_in_models`.
  - Trace folders are named after known models only.
  - Learn session files are named by a hash of key and session id.
  - `goethe logs` refuses targets that leave `logs/`.
- **Uninstall** deletes only a folder that carries the `.goethe-home` marker and a real `bin/goethe`.
  - It also removes Goethe's links into the home.
  - It keeps traces and any folder holding the user's own files.

## Secrets

- **Never log, print or send:** API tokens, the node token, Hugging Face tokens, engine keys, prompt or reply text.
- **Exceptions:**
  - A new key's token is printed once at creation.
  - Tracing and learning store text locally, and only when the user turned them on for that key or model.
- **Hugging Face tokens** (`--hf-token`) are asked for, checked and used for one download. They are never stored.
- **The heartbeat snapshot** is built from local state only. It carries the key inventory (id, name, prefix, scope, expiry, admin) without tokens, and usage counts without text.
- **Console errors** (`platform_api.rs`) never include the token.
- **`goethe.env`** holds only non-default settings, at mode 0600; `GOETHE_PKG_TOKEN` is the only secret it may hold.

## Known open risks

These are known and not fixed yet. Fix them through normal pull requests, and remove the line when you do.

- **Engine key on argv (high on shared hosts).** goethe-engine and NInfer receive `--api-key` on the command line, so any local user can read it from `/proc/<pid>/cmdline` and call the engine port directly, past auth, scope and limits. SGLang and llama.cpp already read the key from the environment. The fix needs goethe-engine and NInfer to read `GOETHE_ENGINE_API_KEY`.
- **Raw tokens at rest.** `api_keys.token` keeps raw tokens at rest, readable by anyone who can read `state.db` (same user).
- **Public share and X-Real-IP.** The public-share mux listener is on loopback (`127.0.0.2`, or `127.0.0.1:<mux+200>` on macOS). A local user can connect to it directly and set `X-Real-IP`, so the share's address check holds only behind nginx. Local users can use any key on the private listener anyway, so this grants nothing new.
- **Blocking cache refresh.** `GatewayCache` refreshes under a std mutex on the async request path. Another process holding a write lock can stall a runtime worker for up to the 200 ms busy timeout.
