#!/usr/bin/env python3
"""Gate: goethe service enable/disable/status with redirected unit/crontab paths."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

from lib.goethe_test import home_goethe_env, main_exit, require_cli, run_goethe, temp_home


def _write_fake_crontab(bin_dir: Path, cron_file: Path) -> None:
    """Install a fake crontab CLI that stores data in cron_file."""
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        CF="{cron_file}"
        case "${{1:-}}" in
          -l)
            if [[ ! -s "$CF" ]]; then
              echo "no crontab for user" >&2
              exit 1
            fi
            cat "$CF"
            ;;
          -)
            cat >"$CF"
            ;;
          *)
            echo "unsupported crontab $*" >&2
            exit 2
            ;;
        esac
        """
    )
    path = bin_dir / "crontab"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_systemctl(bin_dir: Path) -> None:
    """Force service.rs to skip systemd --user."""
    script = "#!/usr/bin/env bash\nif [[ \"$1\" == \"--user\" && \"$2\" == \"status\" ]]; then exit 1; fi\nexit 0\n"
    path = bin_dir / "systemctl"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_crontab_scenario(goethe: Path, goethed: Path) -> None:
    """Enable/disable via fake crontab -l/- with unrelated lines preserved."""
    with temp_home(stub_python=False) as home:
        (home / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(goethed, home / "bin" / "goethed")
        fake_bin = home / "fake-bin"
        fake_bin.mkdir()
        cron_file = home / "user.cron"
        cron_file.write_text(
            "0 1 * * * /usr/bin/backup\n@reboot /home/u/bin/goethed-other --keep\n",
            encoding="utf-8",
        )
        _write_fake_crontab(fake_bin, cron_file)
        _write_fake_systemctl(fake_bin)
        env = {
            **home_goethe_env(home),
            "HOME": str(home),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
        rr = subprocess.run(
            [str(goethe), "service", "enable"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if rr.returncode != 0:
            raise RuntimeError(f"crontab enable failed: {rr.stderr}")
        body = cron_file.read_text(encoding="utf-8")
        if "# goethe-goethed-reboot" not in body or "@reboot GOETHE_HOME=" not in body:
            raise RuntimeError("goethe crontab entry missing")
        if "/home/u/bin/goethed-other --keep" not in body:
            raise RuntimeError("unrelated crontab line removed on enable")
        st = subprocess.run(
            [str(goethe), "service", "status"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if "crontab" not in st.stdout:
            raise RuntimeError(f"expected crontab status: {st.stdout!r}")
        subprocess.run([str(goethe), "service", "disable"], env=env, check=True)
        after = cron_file.read_text(encoding="utf-8")
        if "# goethe-goethed-reboot" in after or f"GOETHE_HOME={home}" in after:
            raise RuntimeError("goethe crontab not removed on disable")
        if "/home/u/bin/goethed-other --keep" not in after:
            raise RuntimeError("unrelated crontab line removed on disable")


def main() -> None:
    """Exercise boot helpers without touching the real crontab or systemd dirs."""
    goethe = require_cli("goethe")
    goethed = require_cli("goethed")
    _run_crontab_scenario(goethe, goethed)
    with temp_home(stub_python=False) as home:
        (home / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(goethed, home / "bin" / "goethed")
        unit_dir = home / "systemd-user"
        unit_dir.mkdir()
        cron_file = home / "crontab.txt"
        cron_file.write_text("", encoding="utf-8")
        env = {
            **home_goethe_env(home),
            "HOME": str(home),
            "GOETHE_SYSTEMD_USER_UNIT_DIR": str(unit_dir),
            "GOETHE_CRONTAB_FILE": str(cron_file),
        }
        rr = run_goethe([str(goethe), "service", "enable"], env=env)
        if rr.returncode != 0:
            raise RuntimeError(f"service enable failed: {rr.stderr}")
        unit = unit_dir / "goethed.service"
        has_unit = unit.is_file()
        has_cron = "# goethe-goethed-reboot" in cron_file.read_text(encoding="utf-8")
        if not has_unit and not has_cron:
            raise RuntimeError("enable did not install user unit or crontab entry")
        st = run_goethe([str(goethe), "service", "status"], env=env)
        if st.returncode != 0:
            raise RuntimeError(f"service status failed: {st.stderr}")
        out = st.stdout
        if "boot method:" not in out or "goethed:" not in out:
            raise RuntimeError(f"unexpected status output: {out!r}")
        if has_unit and "systemd" not in out:
            raise RuntimeError("expected systemd method in status")
        if has_cron and not has_unit and "crontab" not in out:
            raise RuntimeError("expected crontab method in status")
        rr2 = run_goethe([str(goethe), "service", "enable"], env=env)
        if rr2.returncode != 0:
            raise RuntimeError("second enable should be idempotent")
        run_goethe([str(goethe), "service", "disable"], env=env)
        if unit.is_file():
            raise RuntimeError("user unit not removed on disable")
        if "# goethe-goethed-reboot" in cron_file.read_text(encoding="utf-8"):
            raise RuntimeError("crontab marker not removed on disable")
        off = subprocess.run(
            [str(goethe), "service", "restore", "--off"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if off.returncode != 0:
            raise RuntimeError(f"restore off: {off.stderr}")
        on = subprocess.run(
            [str(goethe), "service", "restore", "--on"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if on.returncode != 0:
            raise RuntimeError(f"restore on: {on.stderr}")
    print("stage3_service ok")


if __name__ == "__main__":
    main_exit(main)
