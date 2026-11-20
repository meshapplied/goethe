# Audit 1 (November 2026)

A full audit of goethe (the `logging` branch) and the goethe-web console (the `analytics` branch): bugs, exploits, unnecessary comments and dead code. Four reviewers each took one area with no file shared: the daemon and request path; install, update and supply chain; the CLI, models, fit and terminal UI; and goethe-web. Each area had two rounds: first the audit with its clear fixes, then the medium findings that were safe to change. Every fix has a test where one was practical. The results were then checked end to end on a GPU host, in an installer VM and against a live console.

## How it was verified

- **goethe unit tests and checks:** Rust tests (goethe-core, goethed, goethe), clippy with warnings as errors, `goethe/tests/lint_style.py`, ruff, `bash -n` on every shell script, and the Python suite from the repo root.
- **CI end-to-end suites:** `tests/e2e_cli.py FreshUser` and `Matrix`.
- **Installer:** the Rocky 9 VM scenario (`goethe/tests/vm/run.sh`), 40 of 40 checks.
- **One GPU, an SGLang model and a llama.cpp model:** every API format (OpenAI chat and completions, Anthropic messages, Responses) streamed and not streamed, `n=2`, usage on and off, unknown model, bad JSON, rate limit and wrong key.
- **Usage log:** checked against the traffic request for request.
- **Console:** a signed-in end-to-end run covering analytics, the docs, the passkey flow and the Docs dropdown, plus the snapshot and install-redirect checks.

## Changes made

### goethe: security

- **Upload routing:** the model field of an audio upload is read per form part. A crafted upload could route one model past the scope check while the engine read another.
- **Memory caps:**
  - rebuilt (non-streamed) replies are capped at 64 MB;
  - live learn sessions are capped at 2000.
- **goethed's socket** is owner-only from the moment it exists, through the umask around `bind`.
- **`goethe doctor --fix`** no longer runs a script file that another process of the same user could swap before `sudo` runs it. The script goes in through `bash -c`.
- **openssl** is run from a system path, never from PATH.
- **Plain http** is refused for the package URL, the registry and the token, except for loopback addresses or with `GOETHE_ALLOW_HTTP=1`. This is checked in both `install.sh` and goethe.
- **The installer** passes the auth key to an older release's installer through the environment, never on the command line, and cleans up its temp folder.
- **Hostile model files:**
  - shards in a model's index cannot point outside its folder;
  - hostile safetensors headers and absurd `config.json` shapes cannot overflow the fit math, so they can no longer wrap round to "fits";
  - `--ctx` and `--max-conc` have upper bounds.
- **Model output** printed on a terminal cannot send escape sequences.
- **Model ids or revisions** starting with `-` never reach `hf`.
- **The public share:** an edited share file cannot inject nginx directives.
- **systemd units:** `%` in the home path is escaped.

### goethe: correctness

- **Usage log:**
  - an oversize usage report is sent in older halves, so the cursor always moves;
  - a minute never reopens once a later one started;
  - the hourly rollup is atomic.
- **A non-streamed reply** whose stream fails midway is an error reply with the engine's status, not a 200.
- **Watchdog:**
  - an unreadable `/proc` entry means "unknown", not dead;
  - engines that survive a stop are retried every tick;
  - a failed RUNNING or desired-load write stops the engine instead of being ignored;
  - the per-slot in-flight count is a drop guard.
- **Loads:**
  - a load during restart backoff cannot start a second engine;
  - a load whose settings differ from the running copy says so;
  - the shutdown flag is re-checked after the plan lock.
- **Updates:**
  - a failed binary swap rolls back correctly;
  - a successful upgrade no longer ends in an error;
  - each update or setup run stages in its own private folder.
- **Setup** reports kept engines correctly.
- **The driver installer** checks that its repository is reachable before it removes the old driver.
- **CLI behaviour:**
  - `load --dry-run` honours `--gpu`, `--ctx` and `--max-conc`;
  - a GPU pair NVML cannot describe no longer breaks the hardware profile.
- **Terminal and shell files:**
  - Ctrl-C at a secret prompt restores the terminal;
  - a broken completion block cannot wipe a shell rc file, and rc writes are atomic.
- **Model names and logs:**
  - `remove --yes` needs an exact id or alias;
  - aliases are one word and match case exactly in both the CLI and the gateway;
  - `goethe logs` cannot leave its folder;
  - port files are written atomically, and port 0 is refused.
- **Cleanup:**
  - dead code removed: the unreachable old home screen and about 15 unused functions;
  - duplicated helpers merged;
  - the unused unit template removed;
  - `goethe/README.md` corrected in four places;
  - CI jobs have read-only token permissions;
  - an intermittent test fixed.
- **New:** `goethe update --check` names nginx by version, marks rebuilds as "another build", and ends with a summary of what would change.

### goethe-web

- **Bugs:**
  - email invites crashed on every send;
  - handing over a personal workspace could lock its old owner out;
  - workspace create and rename crashed on empty names;
  - some docs URLs returned 500.
- **Cross-repo bug:** heartbeats from nodes with an instance RESTARTING or DRAINING were rejected whole, which also stopped their usage. Both statuses are accepted now, and an unknown future status shows as UNKNOWN.
- **Install links:** link previews and HEAD requests no longer use up single-use install links or show their key.
- **Hardening:**
  - a cross-site request guard on mutating console calls;
  - every console call acts on the workspace the page shows (`X-Workspace-Id`);
  - JSON bodies are capped at 16 KB, and malformed ones refused;
  - database errors return JSON;
  - unused database roles lost their table privileges (migration `20261120100000_client_grants.sql`).
- **Heartbeats with an unknown token** are rejected before their body is read.
- **Packs and nodes:**
  - the pack index and its signature are cached together;
  - node PATCH checks the node before creating a cluster;
  - a cluster named "Ungrouped" stays distinct.
- **Cleanup:** dead components, fonts, CSS and the unused `@supabase/ssr` dependency removed.

## Open risks and what to do about them

These are ordered by what to do first. Each has a recommended action and its cost.

1. **The release signing key is exposed during the build.** It is present in the `build` job of `release.yml` while third-party build code runs (SGLang's build, unpinned build dependencies, actions pinned only by tag). One compromised dependency could sign anything every goethe trusts.
   - **Do:**
     - build without the key;
     - sign only in the `merge` job, gated by the `release` environment;
     - make the key an environment secret;
     - pin every action to a commit SHA.
   - **Cost:** half a day, plus one release to verify.
   - **Priority:** do this before the next release.
   - **Status:** done. The `build` job runs with `GOETHE_ALLOW_UNSIGNED=1`, `GOETHE_SIGN_KEY` is a `release`-environment secret used only by `merge`, and every action in `release.yml` is pinned by SHA.
2. **goethe-engine and NInfer receive the engine key on their command line**, where any local user can read it (`/proc/<pid>/cmdline`) and call the engine directly, skipping gateway auth.
   - **Do:** have goethe-engine (and the NInfer server) read `GOETHE_ENGINE_API_KEY` from the environment, as SGLang's boot shim and llama.cpp already do; then drop the argument in `engines/goethe_engine.rs` and `engines/ninfer.rs`.
   - **Cost:** a small change in the goethe-engine repo plus a pin bump.
   - **Priority:** next engine release.
3. **A node can make its own login never expire,** overriding the expiry set in the console (`POST /api/device/token-lifetime`). A stolen node token could make itself permanent.
   - **Do:** store the console's limit in its own column and enforce the shorter of the two in `node_by_token`; a node may only shorten its lifetime.
   - **Cost:** one migration plus a small route change; the goethe CLI keeps working.
4. **Silent downgrades.** A compromised package host can serve an older signed release on reinstall. `goethe update` guards the binaries only, so nginx, the SGLang wheel and the ternary pack follow whatever the package list names.
   - **Do:**
     - refuse a downgrade in `setup` unless a version was pinned;
     - skip nginx and wheel downgrades unless `--allow-downgrade`;
     - add a serial to the registry signature.
5. **Root installs run everything as root,** including model code in engines.
   - **Do:** a dedicated `goethe` system user with GPU device access for root installs, or at least `NoNewPrivileges`, `ProtectSystem` and `PrivateTmp` in the unit.
   - **Verify** on GPU hosts first; device access is the hard part.
6. **The test gate defaults to GPUs 2 and 3** (`GOETHE_TEST_GPUS`), which is unsafe on a shared host.
   - **Do:** have real-engine stages require an explicit `GOETHE_TEST_GPUS`.
7. **Smaller items:**
   - **Console access:**
     - sharing a node as operator also lets the member reveal its API keys; decide whether reveal needs whole-workspace scope;
     - prune `pack_downloads`, expired `device_codes` and `install_links` with a scheduled job.
   - **Registry:** use the cached registry offline, and send the package token for registry requests.
   - **CLI and repo housekeeping:**
     - `restore_desired` has no setter left; add `goethe config restore on|off` or drop the knob;
     - pin uv in one place (it is in four);
     - narrow about 90 `pub` items that are only used inside their own file.

## Suggestions

- **Console:**
  - a page that lists auth keys and passkeys, with revoke;
  - node rename (`nodes.display_name` exists but is never written);
  - an audit log (invites, shares, key reveals, node moves);
  - a workspace-wide token lifetime policy;
  - Analytics totals across all nodes, and CSV export;
  - the workspace name on destructive confirms and on the Connect and Invite dialogs.
- **CLI:**
  - `goethe doctor` warns about a plain-http package URL and a home folder that is not mode 0700;
  - refreshes that never freeze the menu;
  - bracketed paste in chat;
  - restarts in the current window shown in `goethe ps`;
  - drop the "-" memory-bandwidth column on macOS.
- **Code health:**
  - split `main.rs`'s long match into `do_*` functions;
  - merge the disk-space (`statvfs`) helpers into one `sys` function;
  - add a gateway test through a real HTTP server (headers and length as the client sees them);
  - add a VM scenario for `goethe update` with an engine installed, and one for an interrupted swap.
