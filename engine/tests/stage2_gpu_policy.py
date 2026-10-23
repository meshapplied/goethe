#!/usr/bin/env python3
"""Gate: GPU allow/deny policy and load rejection before engine spawn."""

from __future__ import annotations

import json
import subprocess

from lib.goethe_test import (
    assert_no_stub_engines,
    home_goethe_env,
    main_exit,
    require_cli,
    run,
    run_goethe,
    seed_installed_model,
    spawn,
    temp_home,
    test_gpus,
)


def _gpu_list_json(goethe, env: dict) -> list[dict]:
    out = subprocess.check_output(
        [str(goethe), "gpu", "list", "--json"],
        env=env,
        text=True,
    )
    return json.loads(out)


def _profile_indices(goethe, env: dict) -> list[int]:
    run_goethe([str(goethe), "hw", "profile"], env=env, timeout=120)
    rows = _gpu_list_json(goethe, env)
    return [int(r["index"]) for r in rows]


def _pick_test_gpu(indices: list[int]) -> int:
    allowed = set(test_gpus())
    for idx in sorted(indices):
        if idx in allowed:
            return idx
    raise RuntimeError(f"no profile GPU in GOETHE_TEST_GPUS among {indices}")


def main() -> None:
    """Exercise gpu policy CLI and load guardrails."""
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, tag="gpu-policy") as home:
        env = home_goethe_env(home)
        indices = _profile_indices(goethe, env)
        if not indices:
            raise RuntimeError("empty hardware profile")
        for row in _gpu_list_json(goethe, env):
            st = row.get("status")
            if st not in ("ALLOWED", "DENIED"):
                raise RuntimeError(f"bad status {st!r} for gpu {row.get('index')}")
        gpu_idx = _pick_test_gpu(indices)
        try:
            run_goethe([str(goethe), "gpu", "deny", str(gpu_idx)], env=env)
            rows = _gpu_list_json(goethe, env)
            st = next(r["status"] for r in rows if int(r["index"]) == gpu_idx)
            if st != "DENIED":
                raise RuntimeError(f"expected DENIED got {st}")
            rr99 = run([str(goethe), "gpu", "deny", "99"], env=env)
            if rr99.returncode == 0:
                raise RuntimeError("deny 99 should fail")
            goethed = spawn([str(require_cli("goethed"))], env=env)
            try:
                model_dir = home / "models" / "org__gp"
                model_dir.mkdir(parents=True)
                (model_dir / "goethe_card.toml").write_text(
                    'model_id="gp"\ncontext_length=512\nmax_running_requests=2\n',
                    encoding="utf-8",
                )
                seed_installed_model(
                    home, "gp", str(model_dir.relative_to(home)), 5.0, 512, 512, 2, 1
                )
                load_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu_idx)}
                rr_load = run(
                    [str(goethe), "load", "gp", "--gpus", str(gpu_idx), "--yes"],
                    env=load_env,
                    timeout=60,
                )
                if rr_load.returncode == 0:
                    raise RuntimeError("load on denied GPU should fail")
                err = (rr_load.stderr or rr_load.stdout or "").lower()
                if "denied by policy" not in err:
                    raise RuntimeError(f"missing denied by policy: {err[-400:]}")
                assert_no_stub_engines(home)
                rr_ep = run([str(goethe), "load", "gp", "--ep", "0"], env=env, timeout=30)
                if rr_ep.returncode == 0:
                    raise RuntimeError("--ep 0 should fail")
                if "at least 1" not in (rr_ep.stderr or rr_ep.stdout or ""):
                    raise RuntimeError(f"unexpected --ep 0 message: {rr_ep.stderr}")
            finally:
                goethed.stop()
        finally:
            run([str(goethe), "gpu", "allow", str(gpu_idx)], env=env)
    print("stage2_gpu_policy ok")


if __name__ == "__main__":
    main_exit(main)
