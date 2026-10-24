#!/usr/bin/env python3
"""Stage 3 remove safety, key create output, host profile, and the foreign-process load notice."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from lib.goethe_test import (
    foreign_gpu_process,
    gpu_pids,
    home_goethe_env,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    spawn,
    temp_home,
    wait_goethed_socket,
    wait_until,
)

_CFG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
}


def _tree(root: Path, name: str) -> Path:
    """Create a minimal local model folder."""
    src = root / name
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps(_CFG), encoding="utf-8")
    (src / "w.safetensors").write_bytes(b"x" * 256)
    return src


def _listing(home: Path) -> list[str]:
    """Every path under home, for before/after comparisons."""
    return sorted(str(p.relative_to(home)) for p in home.rglob("*"))


def _check_remove(goethe: Path, home: Path, env: dict[str, str]) -> None:
    """Managed copies are deleted, in-place folders are kept with an rm hint, bad ids change nothing."""
    managed_src = _tree(home / "src", "managed")
    run_goethe([str(goethe), "install", "--model", str(managed_src), "--copy", "--name", "org/managed"], env=env)
    copy_dir = home / "models" / "org__managed"
    if not (copy_dir / "w.safetensors").is_file() or (copy_dir / "w.safetensors").is_symlink():
        raise RuntimeError("--copy install did not copy the weights")
    rr = run([str(goethe), "remove", "--model", "org/managed"], env=env, input_text="")
    if rr.returncode != 3:
        raise RuntimeError(f"remove without --yes and without a tty should exit 3, got {rr.returncode}")
    run_goethe([str(goethe), "remove", "--model", "org/managed", "--yes"], env=env)
    if copy_dir.exists():
        raise RuntimeError("remove of a managed model left its files")
    if not (managed_src / "w.safetensors").is_file():
        raise RuntimeError("remove of a copy touched the source folder")
    inplace_src = _tree(home / "src", "inplace")
    run_goethe([str(goethe), "install", "--model", str(inplace_src), "--name", "local/inplace"], env=env)
    if (inplace_src / "goethe_card.toml").exists():
        raise RuntimeError("in-place install wrote its card into the user folder")
    rr = run_goethe([str(goethe), "remove", "--model", "local/inplace", "--yes"], env=env)
    if not (inplace_src / "w.safetensors").is_file() or not (inplace_src / "config.json").is_file():
        raise RuntimeError("remove deleted an in-place model's files")
    if f"rm -rf '{inplace_src.resolve()}'" not in rr.stdout:
        raise RuntimeError(f"remove did not print the rm command: {rr.stdout!r}")
    if (home / "models" / "local__inplace").exists():
        raise RuntimeError("remove left goethe's link directory")
    before = _listing(home)
    for bad in ("", "..", ".", "/", "org/..", "../models"):
        rr = run([str(goethe), "remove", "--model", bad, "--yes"], env=env)
        if rr.returncode != 2:
            raise RuntimeError(f"remove {bad!r} exited {rr.returncode}, expected usage error 2")
    if _listing(home) != before:
        raise RuntimeError("rejected remove ids changed GOETHE_HOME")


def _check_key_and_profile(goethe: Path, home: Path, env: dict[str, str]) -> None:
    """--raw prints exactly one token line; hw profile JSON carries host and environment."""
    rr = run_goethe([str(goethe), "key", "--create", "ci", "--raw"], env=env)
    if not re.fullmatch(r"goethe_sk_[0-9a-f]{32}\n", rr.stdout):
        raise RuntimeError(f"key create --raw stdout is not exactly the token: {rr.stdout!r}")
    rr = run_goethe([str(goethe), "key", "--create", "ci2"], env=env)
    if "goethe_sk_" not in rr.stdout or "goethe api shows it again" not in rr.stdout:
        raise RuntimeError(f"key create without --raw must show the token once: {rr.stdout!r}")
    rr = run_goethe([str(goethe), "hw", "profile", "--json"], env=env, timeout=120)
    state = json.loads(rr.stdout)
    host = state.get("host") or {}
    environment = state.get("environment") or {}
    for key in ("cpu", "memory", "disks", "os"):
        if key not in host:
            raise RuntimeError(f"hw profile host section lacks {key}")
    if host["cpu"].get("logical_threads", 0) < 1 or host["memory"].get("total_mb", 0) < 1:
        raise RuntimeError(f"hw profile host section looks empty: {host}")
    if "cloud" not in environment or "systemd_pid1" not in environment:
        raise RuntimeError(f"hw profile environment section incomplete: {environment}")
    for link in state.get("links", []):
        if link["type"] not in ("NVLINK", "PCIE_SWITCH", "SAME_NUMA", "CROSS_SOCKET") or not link.get("topo"):
            raise RuntimeError(f"link is not a CONTRACT C class with a raw topo code: {link}")


def _check_foreign_notice(goethe: Path, home: Path, env: dict[str, str]) -> None:
    """A non-tty load next to a process goethe did not start exits 3 with the notice (any foreign PID)."""
    src = _tree(home / "src", "tiny")
    run_goethe([str(goethe), "install", "--model", str(src), "--name", "org/tiny"], env=env)
    goethed = spawn([str(require_cli("goethed"))], env=env, label="goethed")
    alien = None
    try:
        wait_goethed_socket(home)
        gpu_index = roomiest_test_gpu(8.0)
        alien = foreign_gpu_process(gpu_index, mb=1024)
        pid = alien.popen.pid
        if not wait_until(lambda: any(p == pid for p, _ in gpu_pids(gpu_index)), timeout=90):
            raise RuntimeError("test allocation never appeared on the GPU")
        rr = run_goethe([str(goethe), "hw", "profile", "--json"], env=env, timeout=120)
        gpu = next(g for g in json.loads(rr.stdout)["gpus"] if int(g["index"]) == gpu_index)
        if not any(int(p["pid"]) == pid and int(p["used_vram_mb"]) >= 900 for p in gpu["running_processes"]):
            raise RuntimeError("hw profile does not list the foreign process with its VRAM")
        proc = subprocess.run(
            [str(goethe), "load", "--model", "org/tiny", "--gpus", str(gpu_index), "--no-wait"],
            env={**os.environ, **env},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 3:
            raise RuntimeError(f"load next to a foreign process exited {proc.returncode}: {proc.stderr[-400:]}")
        if f"unmanaged process PID {pid}" not in proc.stderr or f"kill {pid}" not in proc.stderr:
            raise RuntimeError(f"notice must name our foreign PID and the kill command: {proc.stderr[-400:]}")
    finally:
        if alien is not None:
            alien.stop()
        goethed.stop()


def main() -> None:
    """Run the remove, key, profile, and foreign-notice checks in one temp home."""
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, tag="remove") as home:
        env = home_goethe_env(home)
        _check_remove(goethe, home, env)
        _check_key_and_profile(goethe, home, env)
        _check_foreign_notice(goethe, home, env)
    print("stage3_remove_and_cli ok")


if __name__ == "__main__":
    main_exit(main)
