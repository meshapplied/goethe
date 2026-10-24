#!/usr/bin/env python3
"""Stage 4: ps counts in-flight requests per instance, and models shows a loaded model as loaded, not unfit (N8)."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from lib.goethe_test import (
    gpu_flock,
    home_goethe_env,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sse_events,
    start_goethed,
    temp_home,
    test_mux_port,
    wait_until,
    write_card,
)

FAILS: list[str] = []


def _in_flight(goethe: str, env: dict[str, str], model: str) -> list[int]:
    """Sorted in_flight of every ps --json row of model."""
    rows = json.loads(run([goethe, "ps", "--json"], env=env, timeout=30).stdout or "[]")
    return sorted(int(r.get("in_flight") or 0) for r in rows if r.get("model_id") == model)


def _per_instance(goethe: str, env: dict[str, str], home_mux: int, token: str) -> None:
    """One slow stream to a model with two replicas shows IN_FLIGHT 1 on one replica and 0 on the other."""
    body = json.dumps({"model": "r", "messages": [{"role": "user", "content": "s"}], "stream": True, "max_tokens": 6})
    out: dict[str, Any] = {}

    def stream() -> None:
        url = f"http://127.0.0.1:{home_mux}/v1/chat/completions"
        out["events"] = sse_events(url, body.encode(), headers={"Authorization": f"Bearer {token}"}, max_chunks=12)

    th = threading.Thread(target=stream)
    th.start()
    seen: list[list[int]] = []
    try:
        wait_until(lambda: bool(seen.append(_in_flight(goethe, env, "r")) or sum(seen[-1]) >= 1), timeout=10)
    finally:
        th.join(timeout=30)
    busy = next((s for s in seen if sum(s) >= 1), [])
    if busy != [0, 1]:
        FAILS.append(f"one stream across two replicas must count on one instance only: in_flight {busy or seen[-3:]}")


def _models_line(goethe: str, env: dict[str, str], gpu: int) -> None:
    """The fit column of a loaded model lists its instances instead of judging whether another copy fits."""
    text = run_goethe([goethe, "models"], env=env).stdout
    rows = {cols[0]: cols for cols in (re.split(r"\s{2,}", ln.strip()) for ln in text.splitlines()) if cols}
    want = {"r": f"2 instances: GPU {gpu} RUNNING, GPU {gpu} RUNNING", "s": f"1 instance: GPU {gpu} RUNNING"}
    for model, prefix in want.items():
        cols = rows.get(model, [])
        if len(cols) < 4 or cols[1] != "loaded" or not cols[3].startswith(prefix):
            FAILS.append(f"models must show {model}'s instances ({prefix!r}), not a fit judgment: {cols}")


def main() -> None:
    """Two stub replicas of one model on the roomiest test GPU, with a hardware profile so models judges fit."""
    goethe = str(require_cli("goethe"))
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="psmodels") as home:
        env = home_goethe_env(home)
        write_card(home / "models" / "org__r", "r", 512, 2, env={"STUB_TOKEN_INTERVAL": "1.0"})
        seed_installed_model(home, "r", "models/org__r", 5.0, 512, 512, 2, 1)
        write_card(home / "models" / "org__s", "s", 512, 2)
        seed_installed_model(home, "s", "models/org__s", 5.0, 512, 512, 2, 1)
        run_goethe([goethe, "hw", "profile"], env=env, timeout=120)
        md = start_goethed(home)
        try:
            load = [goethe, "load", "--model", "r", "--gpus", str(gpu), "--yes"]
            run_goethe(load, env=env, timeout=120)
            run_goethe([*load, "--replica"], env=env, timeout=120)
            run_goethe([goethe, "load", "--model", "s", "--gpus", str(gpu), "--yes"], env=env, timeout=120)
            out = run_goethe([goethe, "key", "--create", "ps", "--raw"], env=env).stdout
            token = next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))
            _per_instance(goethe, env, test_mux_port(home), token)
            _models_line(goethe, env, gpu)
        finally:
            run([goethe, "unload", "--all"], env=env, timeout=60)
            md.stop()
    if FAILS:
        raise RuntimeError(" | ".join(FAILS))
    print("stage4_ps_models_stub ok")


if __name__ == "__main__":
    main_exit(main)
