#!/usr/bin/env python3
"""Stage 3 input validation and exit codes (F10, F11): bad input is refused with the right code and no side effect."""

from __future__ import annotations

from lib.goethe_test import (
    RunResult,
    home_goethe_env,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    temp_home,
    write_card,
)

FAILS: list[str] = []


def expect(rr: RunResult, ok: bool, what: str) -> None:
    """Collect a failed expectation with the command's output instead of stopping."""
    if not ok:
        FAILS.append(f"{what}: exit {rr.returncode} {(rr.stderr or rr.stdout).strip()[-160:]!r}")


def main() -> None:
    """Refused values change nothing; usage errors exit 2, missing targets exit 1."""
    goethe = str(require_cli("goethe"))
    gpu = roomiest_test_gpu()
    with temp_home(stub_python=True, tag="valid") as home:
        env = home_goethe_env(home)
        db = home / "state.db"
        card = write_card(home / "models" / "org__demo", "demo", 1024, 4)
        write_card(home / "models" / "org__other", "other", 1024, 4)
        seed_installed_model(home, "demo", "models/org__demo", 1.0, 1000, 1024, 4, 1)
        seed_installed_model(home, "other", "models/org__other", 1.0, 1000, 1024, 4, 1)
        card_before = card.read_text(encoding="utf-8")
        md = start_goethed(home)
        try:
            for value in ("0", "999999999"):
                rr = run([goethe, "config", "--model", "demo", "--ctx", value], env=env)
                expect(rr, rr.returncode != 0, f"config --ctx {value} refused")
            expect(rr, card.read_text(encoding="utf-8") == card_before, "refused config left the card unchanged")
            q = "SELECT default_ctx, default_max_conc FROM installed_models WHERE model_id='demo'"
            defaults = sqlite_query(db, q)
            expect(rr, defaults == [(1024, 4)], f"refused config left the DB defaults unchanged {defaults}")
            load = [goethe, "load", "--model", "demo", "--gpus", str(gpu), "--yes", "--no-wait"]
            for flag in (["--ctx", "0"], ["--max-conc", "0"]):
                rr = run([*load, *flag], env=env)
                expect(rr, rr.returncode != 0, f"load {flag[0]} 0 refused")
            for flag in (["--bogus"], ["--gpu", str(gpu)], ["--yess"]):
                rr = run([*load, *flag], env=env)
                expect(rr, rr.returncode == 2, f"load {flag[0]} is a usage error (exit 2)")
            rows = sqlite_query(db, "SELECT COUNT(*) FROM loaded_instances")[0][0]
            expect(rr, rows == 0, f"refused loads started nothing ({rows} instance rows)")
            for args in ([""], ["scoped", "--models", ""]):
                rr = run([goethe, "key", "--create", *args, "--raw"], env=env)
                expect(rr, rr.returncode != 0 and "goethe_sk_" not in rr.stdout, f"key create {args} refused")
            rr = run([goethe, "key", "--create", "zero", "--rpm", "0", "--raw"], env=env)
            text = run([goethe, "key"], env=env).stdout
            line = next((ln for ln in text.splitlines() if ln.split()[:1] == ["zero"]), "")
            unlimited = line.split()[3:4] == ["unlimited"]
            expect(rr, rr.returncode == 0 and unlimited, f"--rpm 0 is listed as unlimited: {line!r}")
            keys = sqlite_query(db, "SELECT name FROM api_keys WHERE name IN ('', 'scoped')")
            expect(rr, keys == [], f"refused keys were stored {keys}")
            for alias in (["", "demo"], ["other", "demo"]):
                rr = run([goethe, "alias", "--set", alias[0], "--model", alias[1]], env=env)
                expect(rr, rr.returncode != 0, f"alias set {alias} refused (empty name or a model id)")
            aliases = sqlite_query(db, "SELECT alias FROM model_aliases")
            expect(rr, aliases == [], f"refused aliases were stored {aliases}")
            rr = run([goethe, "service", "restore", "--on", "--off"], env=env)
            expect(rr, rr.returncode == 2, "service restore --on --off is a usage error")
            rr = run([goethe, "logs"], env=env)
            expect(rr, rr.returncode == 2, "logs without a target is a usage error")
            rr = run([goethe, "logs", "--model", ""], env=env)
            expect(rr, rr.returncode == 2, "logs with an empty target is a usage error")
            rr = run([goethe, "logs", "--model", "demo", "--daemon"], env=env)
            expect(rr, rr.returncode == 2, "logs <model> --daemon is a usage error, not a silently ignored model")
            rr = run([goethe, "unload", "--model", "no-such-model"], env=env)
            expect(rr, rr.returncode == 1, "unload of an unknown model exits 1")
            rr = run([goethe, "allowlist", "--remove", "10.9.9.9/32"], env=env)
            expect(rr, rr.returncode == 1, "allowlist remove of an absent entry exits 1")
        finally:
            run([goethe, "unload", "--all"], env=env)
            md.stop()
    if FAILS:
        raise RuntimeError(f"{len(FAILS)} validation checks failed: " + " | ".join(FAILS))
    print("stage3_validation ok")


if __name__ == "__main__":
    main_exit(main)
