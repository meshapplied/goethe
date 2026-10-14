"""Point `current` at a staged engine, or flip it back. The gate decides. This module does not boot one."""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path


class ReleaseError(Exception):
    pass


def switch(root: Path, staged: Path, gate: Callable[[Path], bool]) -> None:
    """Make `current` point at `staged` only if `gate` accepts it. A refusal leaves `current` as it was."""
    root.mkdir(parents=True, exist_ok=True)
    if staged.parent != root:
        raise ReleaseError(f"{staged} is not an engine directory inside {root}")
    staged = _dir(root, staged.name)
    if not gate(staged):
        raise ReleaseError("gate refused; current was left unchanged")
    current = root / "current"
    old = _target(current)
    if old:
        _component(old)
    hist = [n for n in _history(root) if n != staged.name]
    if old and old != staged.name:
        hist.insert(0, old)
    _point(current, staged)
    for name in hist[2:]:
        victim = root / name
        if victim.parent == root and victim.is_dir() and not victim.is_symlink():
            shutil.rmtree(victim)
    _write_history(root, hist[:2])


def rollback(root: Path) -> str:
    """Point `current` at the previous engine. The one that was current becomes the previous."""
    hist = _history(root)
    if not hist:
        raise ReleaseError("nothing to roll back")
    _dir(root, hist[0])
    current = root / "current"
    old = _target(current)
    _point(current, root / hist[0])
    rest = hist[1:]
    if old and old != hist[0]:
        rest.insert(0, old)
    _write_history(root, rest[:2])
    return hist[0]


def _target(link: Path) -> str | None:
    return os.readlink(link) if link.is_symlink() else None


def _point(link: Path, target: Path) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target.name)
    os.replace(tmp, link)


def _component(name: str) -> None:
    if name != Path(name).name or name in (".", ".."):
        raise ReleaseError(f"bad engine name {name!r}")


def _dir(root: Path, name: str) -> Path:
    _component(name)
    path = root / name
    if path.is_symlink() or not path.is_dir():
        raise ReleaseError(f"{name} is not an engine directory")
    return path


def _history(root: Path) -> list[str]:
    path = root / "history"
    if not path.is_file():
        return []
    names = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        _component(line)
        names.append(line)
    return names


def _write_history(root: Path, names: list[str]) -> None:
    (root / "history").write_text("".join(name + "\n" for name in names))
