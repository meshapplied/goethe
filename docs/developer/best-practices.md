# Best practices

The hard rules from [AGENTS.md](../../AGENTS.md), the conventions the code follows, and the checklist for every pull request.

## Hard rules

- **Fail closed.** The allowlist, packs, release key and public share refuse when a check cannot complete. Silent success is a bug: an error path that returns success, a swallowed `Result`, or a skipped step that reports "done".
- **Smallest change that fixes the real bug.** Do not add speculative frameworks, abstractions for one caller, or drive-by refactors in a fix. Recommend larger refactors in the PR description instead.
- **Commits and pushes.** Commit only when asked. Never force-push `main`. No AI co-author or "Generated with" lines; CI's `attribution` job fails them.
- **Local-only files.** Never track `PROJECT.md`, `.local/` or the agent skill folders. `lint_style.py` refuses them.
- **No internal jargon** in commits, PRs or user-facing text: no host names, agent traces, or local paths.
- **Match `lint_style.py`** and the CI lint.
- **Out of scope unless asked:** GPU acceptance loads, a rewrite of the learn or OOM pipeline, more folder renames.

## Code

- **Rust skills.** When writing or reviewing Rust, read the matching skill in `.local/skills/beautiful-rust/` (linked from `.agents/skills`, `.claude/skills` and `.cursor/skills`) when it is present.
- **Where logic lives.** Logic lives in `goethe-core`. The binaries parse arguments, dispatch and print.
- **Pure decisions.** Put decisions in pure functions (inputs in, a decision out) and test those; keep I/O at the edges. `watchdog_policy`, `calc`, `rebuilt_reply` and `public_share::decide` are the pattern.
- **One source of truth per rule.**
  - The fit rule: `calc::missing_gb`.
  - Release-key checks: `release_key::verify`.
  - Signals to processes: `proc_ident::signal`.
  - Sentences said in more than one place: `messages.rs`, where a test fails on a duplicate.
  - Before adding a helper, search for an existing one.
- **Locks and async.**
  - Recover poisoned std mutexes with `lock().unwrap_or_else(|e| e.into_inner())`.
  - Never hold a std mutex across `.await`.
  - Blocking work (SQLite, NVML, subprocesses) belongs on `spawn_blocking` or a thread, not on the mux's async workers.
- **Untrusted input.** Treat anything from the network, the socket, files in the home and `/proc` as untrusted:
  - Bound every buffer and every map keyed by client data.
  - Use checked or saturating arithmetic.
  - Never index or slice without a check.
  - Avoid `unwrap` and `expect` outside tests and startup.
- **SQL.** Use parameters. `format!` may only interpolate constants (table names, status lists).
- **Files.** Write files other people must not read with `write_private` (temp, 0600, rename). Never follow a symlink you did not create.
- **Schema changes.** Add columns and tables in `db::ensure_columns` or `ensure_tables`, without a schema version bump, so an older goethe still opens the database.
- **Dependencies.** Add a crate only when a small module cannot do the job ([packages.md](packages.md#rust)).

## Comments and naming

- **Comments.** Every file opens with a one-line header that says what it is. A function may have exactly one comment, as its first line, saying what it does or why; do not restate the code. Nowhere else ([workflows.md](workflows.md#style-rules)).
- **Names.** Say what a thing is in plain words: `copies_of`, `leader_unreadable`, `rebuild_reply`. Avoid abbreviations a newcomer cannot guess.
- **Test names** are sentences about behavior, for example `a_minute_never_reopens_once_a_later_one_started` or `an_engine_error_inside_a_rebuilt_stream_is_an_error_reply_not_a_200`. An assertion message says what must hold.

## Error messages and user-facing text

- **Content.** Say what happened, why, and what to do, in one line when possible, naming the exact command or flag. Example: `GOETHE_MUX_PORT 30000 is in use; set GOETHE_MUX_PORT to a free port`.
- **Style.** Messages start lowercase and have no trailing period, like `rustc` and `git` messages.
- **Secrets.** Never put a token, a key or prompt text in an error or log line.
- **Plain language.** No internal names, no marketing, and no dashes or emoji (the lint refuses them). Use "Goethe" for the product and `goethe` for the command.
- **Exit codes.** 0 for success, 1 for an error, 2 for usage, 3 when foreign GPU processes need confirmation.
- **Long steps.** A long step prints progress. A silence over 10 seconds, a traceback, or a Rust panic message reaching the user fails the e2e cells.
- **Docs.** When behavior changes, update `docs/cli.md` and `docs/user-guide.md` in the same pull request.

## Tests

- **Every fix gets a test** that fails without the fix, when practical. If it is not practical (it needs a real GPU or an unkillable process), say so in the PR.
- **Prefer tests with no GPU and no network:**
  - temp homes;
  - local fake servers (`fake_engine` in `mux.rs`, the `mock` console in `heartbeat.rs`);
  - stub engines (`tests/lib/stub_engine.py`);
  - the faked world of `tests/e2e/world.py`.
- **Isolation.** Tests may only stop processes they started, must not use the real home, and must not depend on a GPU being free. Gate tests that need GPUs read `GOETHE_TEST_GPUS`.
- **Shipped bugs.** A bug that reached users gets a mutant in `goethe/tests/mutants/` when the e2e suite should have caught it.

## Pull request checklist

- [ ] `cargo test --locked` and `cargo clippy --locked --all-targets -- -D warnings` pass (from `goethe/`).
- [ ] `python3 tests/lint_style.py` prints `lint clean`.
- [ ] New behavior and every fix have a test, or the PR says why not.
- [ ] No new failure path returns success; errors say what to do.
- [ ] No secret reaches a log, an error, the heartbeat or the console.
- [ ] Input from the network, the socket or files is bounded and validated; no new panic is reachable from it.
- [ ] Pins changed with their sha256, and the dependent files regenerated (layer locks, `sglang_support.json`, `llama-cpp/ARCHS`).
- [ ] A changed heartbeat field or enum value is coordinated with the console validator.
- [ ] User docs updated for user-visible changes; `CHANGELOG.md` `## Unreleased` has a line.
- [ ] The PR description is a few lines on what changed and why. No test logs, no AI attribution.
- [ ] Installer, update or service changes: the `installer` workflow ran. macOS-relevant changes: the `macos` workflow ran.
