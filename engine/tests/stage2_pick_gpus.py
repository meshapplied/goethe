#!/usr/bin/env python3
"""Stage 2: hw pick-gpus ranks free VRAM above link quality (N5); its tp=2 pair holds the roomiest GPU."""

from __future__ import annotations

import json
import re

from lib.goethe_test import gpu_free_gb, home_goethe_env, main_exit, require_cli, run_goethe, temp_home


def main() -> None:
    """Profile the host in a temp home, ask where tp=2 would go (nothing is loaded) and compare with live free VRAM."""
    goethe = str(require_cli("goethe"))
    with temp_home(stub_python=False, tag="pick") as home:
        env = home_goethe_env(home)
        run_goethe([goethe, "hw", "profile"], env=env, timeout=120)
        gpus = [int(g["index"]) for g in json.loads((home / "hardware_state.json").read_text())["gpus"]]
        if len(gpus) < 2:
            raise RuntimeError(f"pick-gpus --tp 2 needs a host with two GPUs, found {gpus}")
        out = run_goethe([goethe, "hw", "pick-gpus", "--tp", "2"], env=env, timeout=60).stdout
        found = re.search(r"place tp=2 on GPU ([\d,]+)", out)
        picked = {int(x) for x in found.group(1).split(",")} if found else set()
        free = {i: gpu_free_gb(i) for i in gpus}
        best = max(free.values())
        if len(picked) != 2 or not any(free[i] >= best - 1.0 for i in picked):
            shown = {i: round(f, 1) for i, f in free.items()}
            raise RuntimeError(f"pick-gpus --tp 2 chose {sorted(picked)} over the roomiest GPU (free {shown}): {out}")
    print(f"stage2_pick_gpus ok picked={sorted(picked)}")


if __name__ == "__main__":
    main_exit(main)
