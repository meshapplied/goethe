#!/usr/bin/env python3
"""Stage 2 foreign PID detection gate."""

from __future__ import annotations

import json

from lib.goethe_test import foreign_gpu_process, main_exit, require_cli, require_gpus, resolve_goethe_home, run_goethe


def main() -> None:
    """Ensure goethe hw profile lists a foreign GPU process."""
    goethe = require_cli("goethe")
    gpus = require_gpus(1)
    gpu = gpus[0]
    home = resolve_goethe_home()
    foreign = foreign_gpu_process(gpu, mb=10240)
    try:
        run_goethe([str(goethe), "hw", "profile", "--json"], env={"GOETHE_HOME": str(home)})
        state = json.loads((home / "hardware_state.json").read_text(encoding="utf-8"))
        found = False
        for entry in state.get("gpus", []):
            if int(entry["index"]) != gpu:
                continue
            for proc in entry.get("running_processes", []):
                if int(proc["pid"]) == foreign.popen.pid and int(proc["used_vram_mb"]) >= 9000:
                    found = True
        if not found:
            raise RuntimeError("foreign PID not reported with ~10240 MB")
        print("stage2_foreign_pid ok")
    finally:
        foreign.stop()


if __name__ == "__main__":
    main_exit(main)
