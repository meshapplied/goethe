#!/usr/bin/env python3
"""Stage 2 foreign PID detection gate."""

from __future__ import annotations

import json

from lib.goethe_test import (
    foreign_gpu_process,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run_goethe,
    temp_home,
    wait_gpu_pid,
)

FOREIGN_MB = 3072


def main() -> None:
    """goethe hw profile lists an unmanaged GPU process with its VRAM, written into a temp home."""
    goethe = require_cli("goethe")
    gpu = roomiest_test_gpu(8.0)
    with temp_home(stub_python=True, tag="foreign") as home:
        foreign = foreign_gpu_process(gpu, mb=FOREIGN_MB)
        try:
            if not wait_gpu_pid(gpu, foreign.popen.pid, int(FOREIGN_MB * 0.9), timeout=60):
                raise RuntimeError(f"foreign allocation of {FOREIGN_MB} MiB never became visible on GPU {gpu}")
            run_goethe([str(goethe), "hw", "profile", "--json"], env={"GOETHE_HOME": str(home)})
            state = json.loads((home / "hardware_state.json").read_text(encoding="utf-8"))
            entry = next((g for g in state.get("gpus", []) if int(g["index"]) == gpu), None)
            if entry is None:
                raise RuntimeError(f"GPU {gpu} missing from hardware_state.json")
            procs = {int(p["pid"]): int(p["used_vram_mb"]) for p in entry.get("running_processes", [])}
            used = procs.get(foreign.popen.pid, 0)
            if used < FOREIGN_MB * 0.9:
                raise RuntimeError(f"foreign PID {foreign.popen.pid} not reported with ~{FOREIGN_MB} MB: {procs}")
        finally:
            foreign.stop()
    print(f"stage2_foreign_pid ok gpu={gpu} reported_mb={used}")


if __name__ == "__main__":
    main_exit(main)
