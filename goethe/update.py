"""Compare the local install to NINFER_PIN. Reads the disk, changes nothing, downloads nothing."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from goethe.doctor import Check
from goethe.engine import Install


def check(inst: Install, pin: dict[str, str]) -> list[Check]:
    src = inst.home / "ninfer"
    rev = _rev(src)
    want = pin["NINFER_REV"]
    rows = [_rev_row(rev, want)]
    rows += [_patch_row(src, patch, rev == want) for patch in sorted((inst.recipe / "patches").glob("*.patch"))]
    rows.append(Check("binary", "ok", str(inst.binary)) if inst.has_binary
                else Check("binary", "fail", "ninfer-serve is not built", "run goethe serve"))
    rows.append(_model_row(inst, pin))
    if "GREEDY_DIGEST" in pin:
        rows.append(Check("digest", "ok", f"{pin['GREEDY_DIGEST']}, replay is not run here"))
    return rows


def _rev(src: Path) -> str | None:
    if not (src / ".git").is_dir():
        return None
    done = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def _rev_row(rev: str | None, want: str) -> Check:
    if rev is None:
        return Check("engine", "fail", "NInfer source is not installed", "run goethe serve")
    if rev != want:
        return Check("engine", "fail", f"{rev[:12]}, pin is {want[:12]}")
    return Check("engine", "ok", rev[:12])


def _patch_row(src: Path, patch: Path, at_pin: bool) -> Check:
    if not at_pin:
        return Check("patch", "fail", "not checked, the source is not at the pin")
    applied = _apply_ok(src, patch, reverse=True)
    if applied:
        return Check("patch", "ok", patch.name)
    if _apply_ok(src, patch, reverse=False):
        return Check("patch", "fail", f"{patch.name} is not applied", "run goethe serve")
    return Check("patch", "fail", f"{patch.name} does not match this tree")


def _apply_ok(src: Path, patch: Path, reverse: bool) -> bool:
    cmd = ["git", "-C", str(src), "apply", "--check"]
    if reverse:
        cmd.append("--reverse")
    return subprocess.run([*cmd, str(patch)], capture_output=True).returncode == 0


def _model_row(inst: Install, pin: dict[str, str]) -> Check:
    path = inst.model_file
    if not path.is_file():
        return Check("model", "fail", "not downloaded", "run goethe serve")
    size = path.stat().st_size
    if size != int(pin["MODEL_BYTES"]):
        return Check("model", "fail", f"{size} bytes, pin says {pin['MODEL_BYTES']}")
    print("hashing the model file", file=sys.stderr, flush=True)
    digest = _sha256(path)
    if digest == pin["MODEL_PATCHED_SHA256"]:
        return Check("model", "ok", f"{digest[:12]}, patched")
    if digest == pin["MODEL_SHA256"]:
        return Check("model", "fail", f"{digest[:12]}, the head patch is not applied", "run goethe serve")
    return Check("model", "fail", f"{digest[:12]} matches neither hash in the pin")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
