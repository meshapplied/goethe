# Goethe PR180 greengoblin E2E traces

Real-user session on greengoblin (isolated `GOETHE_HOME`), 2026-11-18. API keys redacted.

| File | What |
|------|------|
| `TRACE-pr180-greengoblin.md` | Full command TRACE + scoreboard + findings |
| `FOLLOWUP-pr180-greengoblin.md` | Load `--yes`, cert SAN, embed run |
| `TRACE-v1-aborted.md` | First run aborted (`set -e` in helper) |
| `console-v2.txt` | Raw SSH console for v2 |
| `../e2e-greengoblin-pr180-session.sh` | Reusable driver |

## Finding status (landed with this PR)

| Sev | Finding | Status |
|-----|---------|--------|
| P0 | `expose public --yes` binds `0.0.0.0` without `--allow` | **Fixed** (refuse; need `--allow` + `--yes`) |
| P0 | TLS SAN missing tailnet/bind IPs (curl 60) | **Fixed** (`tls_names` includes iface IPs; cert refreshes when SAN changes) |
| P1 | `models scan <foldername>` from CWD | **Fixed** (folder names resolve under `models/`) |
| P1 | Symlink under `models/` refused by scan | **Docs/msg** (use `goethe install <path> --name`; intentional no follow) |
| P1 | Non-TTY load blocked by Xorg unless `--yes` | **Keep** (intentional; E2E/CI use `--yes`) |
| P2 | Second `GOETHE_HOME` → port 18443 | **Keep** (warned; test harness must read status URL) |
| P2 | `goethe run` no `-p` | **Keep** (positional INPUT; help already says so) |

## Security incident note (0.0.0.0 bind during E2E)

**Closed.** Prod is tailnet+localhost only. No evidence of external attack in nginx access logs. NAT + allowlist + API key limited blast radius. Fix: `expose public --yes` without `--allow` refuses; raw `expose 0.0.0.0` refuses.
