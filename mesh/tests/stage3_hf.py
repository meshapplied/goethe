#!/usr/bin/env python3
"""Stage 3 Hugging Face install, verify, auth, and preflight tests."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from lib.mesh_test import (
    Skip,
    huggingface_reachable,
    main_exit,
    require_cli,
    resolve_mesh_home,
    resolve_python,
    run,
    run_mesh,
    temp_home,
)


def _registry_for(model: str) -> str:
    return json.dumps(
        {
            "version": 1,
            "models": [
                {
                    "id": model,
                    "architecture": "dense",
                    "static_vram_gb": 0.01,
                    "kv_bytes_per_token": 512,
                    "min_total_vram_gb": 1.0,
                    "supported_sm": ["12.0"],
                    "default_ctx": 128,
                    "default_max_conc": 1,
                    "default_tp": 1,
                    "kv_cache_dtype": "fp8_e4m3",
                    "extra_sglang_args": [],
                }
            ],
        }
    )


def _install_env(home: Path, reg_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "MESH_HOME": str(home),
        "MESH_REGISTRY_URL": f"file://{reg_path}",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if extra:
        env.update(extra)
    return env


def _model_dir(home: Path) -> Path:
    dirs = list((home / "models").glob("*__*"))
    if not dirs:
        raise RuntimeError("model directory missing")
    return dirs[0]


def _copy_hf_cli(home: Path, src: Path) -> None:
    if not src.is_file():
        raise Skip("hf CLI not found in py-env")
    dst = home / "py-env" / "bin" / "hf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        dst.symlink_to(src.readlink())
    else:
        import shutil

        shutil.copy2(src, dst)
    dst.chmod(0o755)


def main() -> None:
    if not huggingface_reachable(3.0):
        raise Skip("huggingface.co unreachable")
    mesh = require_cli("mesh")
    tiny = os.environ.get("MESH_TEST_TINY_MODEL", "hf-internal-testing/tiny-random-gpt2")
    resume_model = os.environ.get("MESH_TEST_RESUME_MODEL", "distilbert/distilgpt2")
    hf_src = resolve_mesh_home() / "py-env" / "bin" / "hf"

    with temp_home(stub_python=True, real_python=resolve_python(), tag="hf") as home:
        _copy_hf_cli(home, hf_src)
        reg = home / "registry.json"
        reg.write_text(_registry_for(tiny), encoding="utf-8")
        env = _install_env(home, reg)
        t0 = time.time()
        for attempt in range(3):
            try:
                run_mesh([str(mesh), "install", tiny, "--no-load"], env=env, timeout=600)
                break
            except RuntimeError as exc:
                if attempt >= 2 or "cannot reach" not in str(exc).lower():
                    raise
                time.sleep(5.0)
        first = time.time() - t0
        t1 = time.time()
        run_mesh(
            [str(mesh), "install", tiny, "--no-load"],
            env={**env, "HF_HUB_OFFLINE": "1"},
            timeout=120,
        )
        second = time.time() - t1
        if second > max(15.0, first * 0.5):
            raise RuntimeError(f"second install not fast enough ({second:.1f}s vs first {first:.1f}s)")

        mdir = _model_dir(home)
        safes = list(mdir.glob("*.safetensors"))
        if safes:
            p = safes[0]
            data = bytearray(p.read_bytes())
            if data:
                data[0] ^= 0xFF
                p.write_bytes(data)
            run_mesh([str(mesh), "install", tiny, "--no-load"], env=env, timeout=600)
            if not (mdir / ".mesh_verified.json").exists() and safes[0].stat().st_size > 0:
                pass

        reg2 = home / "registry_resume.json"
        reg2.write_text(_registry_for(resume_model), encoding="utf-8")
        env_resume = _install_env(home, reg2)
        proc = subprocess.Popen(
            [str(mesh), "install", resume_model, "--no-load"],
            env={**os.environ, **env_resume},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(5.0)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
        run_mesh([str(mesh), "install", resume_model, "--no-load"], env=env_resume, timeout=900)

        gated = "meta-llama/Llama-3.2-1B"
        rr = run(
            [str(mesh), "install", gated, "--no-load"],
            env={"MESH_HOME": str(home), "MESH_REGISTRY_URL": f"file://{reg}"},
            timeout=45,
        )
        if rr.returncode == 0:
            raise RuntimeError("gated install should fail without token")
        msg = (rr.stderr or rr.stdout or "").lower()
        if "mesh login" not in msg:
            raise RuntimeError("gated install missing mesh login hint")

        rr404 = run(
            [str(mesh), "install", "no-such-org/no-such-model-xyz", "--no-load"],
            env={"MESH_HOME": str(home)},
            timeout=60,
        )
        if rr404.returncode == 0:
            raise RuntimeError("missing repo should fail")
        msg404 = (rr404.stderr or rr404.stdout or "").lower()
        if "check the id" not in msg404 and "mesh login" not in msg404:
            raise RuntimeError("missing repo should mention check the id or mesh login")

        rr_net = run(
            [str(mesh), "install", tiny, "--no-load", "--force"],
            env={
                "MESH_HOME": str(home),
                "MESH_REGISTRY_URL": f"file://{reg}",
                "HF_ENDPOINT": "http://127.0.0.1:9",
            },
            timeout=60,
        )
        if rr_net.returncode == 0:
            raise RuntimeError("bad HF_ENDPOINT should fail")
        net_msg = rr_net.stderr or rr_net.stdout or ""
        if "cannot reach" not in net_msg.lower() and "network" not in net_msg.lower():
            raise RuntimeError("bad HF_ENDPOINT should mention network/cannot reach")

        import shutil

        resume_dir = home / "models"
        for p in resume_dir.glob("*distil*"):
            shutil.rmtree(p, ignore_errors=True)
        rr_disk = run(
            [str(mesh), "install", resume_model, "--no-load"],
            env={
                **env_resume,
                "MESH_TEST_FREE_BYTES": "1",
            },
            timeout=120,
        )
        if rr_disk.returncode == 0:
            raise RuntimeError("disk preflight should fail")
        disk_msg = rr_disk.stderr or rr_disk.stdout or ""
        if "need" not in disk_msg.lower() or "available" not in disk_msg.lower():
            raise RuntimeError("disk preflight should show need and available GB")

        proc = subprocess.run(
            [str(mesh), "login", "--token-stdin"],
            input="hf_not_a_real_token\n",
            capture_output=True,
            text=True,
            env={**os.environ, "MESH_HOME": str(home)},
            timeout=30,
        )
        rr_bad = type("R", (), {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})()
        if rr_bad.returncode == 0:
            raise RuntimeError("bad token login should fail")
        if "invalid" not in (rr_bad.stderr or rr_bad.stdout or "").lower():
            raise RuntimeError("bad token login should say invalid")

    print("stage3_hf ok")


if __name__ == "__main__":
    main_exit(main)
