# AGENTS

Index for coding agents in this repo. Read what you need; do not invent architecture.

## Map
- Product path: `goethe/` (Rust `goethe` / `goethed`, CDN install, gateway, packs).
- Lab path: `lab/` (Python). Do not treat it as the shipped CLI unless asked.
- Local doctrine: `PROJECT.md` (gitignored, never commit). Read it when present; never paste it into commits/PRs/issues.
- Parent tree rules: `../AGENTS.md` when working under `meshapplied/`.

## Rust skills
- Source of truth: `.local/skills/beautiful-rust/` (MIT; `PIN.txt` + `SCAN.txt`).
- Discovered via symlinks (same content):
  - `.agents/skills/{rust-best-practices,rust-review,rust-port}`
  - `.cursor/skills/...`
  - `.claude/skills/...`
- Before any refresh from upstream: re-scan for instruction-override / exfil / secrets; keep evals out.
- When writing or reviewing Rust, read the matching `SKILL.md` first.

## Hard rules
- Fail closed (allowlist, packs, release key, share). Silent success is a bug.
- Smallest change that fixes the real bug. No speculative frameworks.
- Never force-push `main`/`master`. Commit only when asked. No AI co-author trailers.
- No internal jargon ("ponytail", agent traces) in commits/PR/user-facing text.
- Never track `PROJECT.md` or `.local/` / agent skill dirs. Lint refuses if they appear in `git ls-files`.
- Match `goethe/tests/lint_style.py` (and CI lint).

## Verify
- Domain/fit: `cd goethe && cargo test -p goethe-core --lib -- <filter>`
- Clippy: `cd goethe && cargo clippy --workspace --locked --all-targets -- -D warnings`
- Stage/stub: `bash goethe/tests/gate.sh <n>` (no GPU unless told)
- Full Linux PR bar: CI `gate`. Mac package e2e only when packaging paths change.

## Out of scope unless asked
- GPU acceptance loads
- Rewriting learn/OOM DB pipeline
- Further folder renames (see `PROJECT.md` when present)
