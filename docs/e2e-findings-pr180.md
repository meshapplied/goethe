# E2E findings (PR180)

Isolated-home session against the PR binary. Session logs with host paths and keys stay off-repo.

## Status

| Sev | Finding | Status |
|-----|---------|--------|
| P0 | `expose public --yes` binds every interface without a client `--allow` | Fixed |
| P0 | Gateway cert SAN missing interface IPs (`curl --cacert` to the bind address fails) | Fixed |
| P1 | `models scan <foldername>` resolved from the current directory | Fixed |
| P1 | Symlink under `models/` refused by scan | Intentional; use `goethe install <path> --name` |
| P1 | Non-TTY load next to a display manager needs `--yes` | Intentional |
| P2 | A second `GOETHE_HOME` may pick another gateway port | Warned; read status URL |
| P2 | `goethe run` has no `-p`; input is positional | Documented |

## CI coverage

- Unit: public `--yes` without `--allow` refuses; scan folder names under `models/`; cert SAN includes interface IPs
- `engine/tests/stage5_configure_api.py`: bare public `--yes`, raw `0.0.0.0`, and `0.0.0.0/0` refuse; named client allow succeeds
