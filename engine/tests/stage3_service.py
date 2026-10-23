#!/usr/bin/env python3
"""Gate: goethe service enable/disable/status with redirected unit/crontab paths."""

from __future__ import annotations

import re
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


def _write_fake_systemctl(bin_dir: Path, log: Path, user_ok: bool) -> None:
    """Fake systemctl that logs every call; user_ok decides whether systemd --user looks available."""
    status = "exit 0" if user_ok else 'if [[ "$1" == "--user" && "$2" == "status" ]]; then exit 1; fi'
    path = bin_dir / "systemctl"
    path.write_text(f'#!/usr/bin/env bash\necho "$*" >>"{log}"\n{status}\nexit 0\n', encoding="utf-8")
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
        _write_fake_systemctl(fake_bin, home / "systemctl.log", user_ok=False)
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


def _user_units(unit_dir: Path) -> dict[str, str]:
    """Map each goethed user unit file name to the GOETHE_HOME it pins."""
    units = {}
    for unit in sorted(unit_dir.glob("goethed*.service")):
        pinned = re.search(r'GOETHE_HOME=([^"\n]+)', unit.read_text(encoding="utf-8"))
        units[unit.name] = pinned.group(1) if pinned else ""
    return units


def _run_two_homes_scenario(goethe: Path, goethed: Path) -> None:
    """Autostart is per home: uninstalling home B leaves home A's user unit untouched (F4)."""
    with temp_home(stub_python=False, tag="svc-a") as home_a, temp_home(stub_python=False, tag="svc-b") as home_b:
        user = home_a / "user-home"
        unit_dir = user / ".config" / "systemd" / "user"
        fake_bin = user / "fake-bin"
        unit_dir.mkdir(parents=True)
        fake_bin.mkdir()
        log = user / "systemctl.log"
        cron_file = user / "crontab.txt"
        cron_file.write_text("", encoding="utf-8")
        _write_fake_crontab(fake_bin, cron_file)
        _write_fake_systemctl(fake_bin, log, user_ok=True)
        for home in (home_a, home_b):
            (home / "bin").mkdir(parents=True, exist_ok=True)
            shutil.copy2(goethe, home / "bin" / "goethe")
            shutil.copy2(goethed, home / "bin" / "goethed")
            (home / ".goethe-home").write_text("goethe home\n", encoding="utf-8")

        def cli(home: Path, *args: str) -> str:
            env = {
                **home_goethe_env(home),
                "HOME": str(user),
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "GOETHE_SYSTEMD_USER_UNIT_DIR": str(unit_dir),
                "GOETHE_CRONTAB_FILE": str(cron_file),
            }
            return run_goethe([str(home / "bin" / "goethe"), *args], env=env).stdout

        cli(home_a, "service", "enable")
        cli(home_b, "service", "enable")
        units = _user_units(unit_dir)
        if sorted(units.values()) != sorted([str(home_a), str(home_b)]):
            raise RuntimeError(f"each home needs its own user unit: {units}")
        a_name = next(name for name, pinned in units.items() if pinned == str(home_a))
        a_text = (unit_dir / a_name).read_text(encoding="utf-8")
        log.write_text("", encoding="utf-8")
        cli(home_b, "uninstall", "--yes")
        left = _user_units(unit_dir)
        if left != {a_name: str(home_a)} or (unit_dir / a_name).read_text(encoding="utf-8") != a_text:
            raise RuntimeError(f"uninstalling home B must leave exactly home A's unit: {left}")
        calls = log.read_text(encoding="utf-8").splitlines()
        if any(a_name in c.split() for c in calls):
            raise RuntimeError(f"uninstalling home B issued systemctl for home A's unit: {calls}")
        status = cli(home_a, "service", "status")
        if "boot method: systemd" not in status:
            raise RuntimeError(f"home A lost its boot method after home B's uninstall: {status!r}")
        cli(home_a, "service", "disable")
        if _user_units(unit_dir):
            raise RuntimeError(f"disable left a user unit behind: {_user_units(unit_dir)}")


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
        fake_bin = home / "fake-bin"
        fake_bin.mkdir()
        log = home / "systemctl.log"
        _write_fake_crontab(fake_bin, cron_file)
        _write_fake_systemctl(fake_bin, log, user_ok=True)
        env = {
            **home_goethe_env(home),
            "HOME": str(home),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "GOETHE_SYSTEMD_USER_UNIT_DIR": str(unit_dir),
            "GOETHE_CRONTAB_FILE": str(cron_file),
        }
        run_goethe([str(goethe), "service", "enable"], env=env)
        units = _user_units(unit_dir)
        if list(units.values()) != [str(home)]:
            raise RuntimeError(f"enable with systemd --user must write one user unit pinning this home: {units}")
        unit = unit_dir / next(iter(units))
        out = run_goethe([str(goethe), "service", "status"], env=env).stdout
        if "boot method:" not in out or "goethed:" not in out or "systemd" not in out:
            raise RuntimeError(f"unexpected status output: {out!r}")
        run_goethe([str(goethe), "service", "enable"], env=env)
        run_goethe([str(goethe), "service", "disable"], env=env)
        if unit.is_file():
            raise RuntimeError("user unit not removed on disable")
        if "# goethe-goethed-reboot" in cron_file.read_text(encoding="utf-8"):
            raise RuntimeError("crontab marker not removed on disable")
        calls = log.read_text(encoding="utf-8").splitlines()
        if not any(f"enable --now {unit.name}" in c for c in calls) or not any("disable" in c for c in calls):
            raise RuntimeError(f"systemctl enable/disable not issued: {calls}")
        if any(not c.startswith("--user") for c in calls):
            raise RuntimeError(f"service touched the system manager instead of --user: {calls}")
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
    _run_two_homes_scenario(goethe, goethed)
    print("stage3_service ok")


if __name__ == "__main__":
    main_exit(main)
