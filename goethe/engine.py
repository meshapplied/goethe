"""NInfer through the nvfp4-kernels recipe: install.sh builds it, fetch_model.sh downloads and patches the model."""
from __future__ import annotations

import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from goethe import store
from goethe.registry import Model, read_pin


@dataclass(frozen=True)
class Install:
    recipe: Path
    home: Path
    binary: Path
    model_file: Path
    model_bytes: int

    @property
    def has_binary(self) -> bool:
        return os.access(self.binary, os.X_OK)

    @property
    def has_model(self) -> bool:
        return self.model_file.is_file() and self.model_file.stat().st_size == self.model_bytes


def locate(model: Model, env: Mapping[str, str] = os.environ) -> Install:
    recipe = store.REPO_ROOT / model.recipe
    pin = read_pin(recipe / "NINFER_PIN")
    home = store.recipe_home(env)
    return Install(recipe, home, home / "ninfer" / "build" / "apps" / "ninfer-serve",
                   home / "models" / pin["MODEL_FILE"], int(pin["MODEL_BYTES"]))


def commands(inst: Install) -> list[list[str]]:
    """What `ensure` would run: build the engine if missing, then fetch and patch the model."""
    steps = [] if inst.has_binary else [[str(inst.recipe / "install.sh")]]
    return steps + [[str(inst.recipe / "fetch_model.sh")]]


def ensure(inst: Install, env: Mapping[str, str] = os.environ) -> None:
    child_env = {**env, "NVFP4_HOME": str(inst.home)}
    for cmd in commands(inst):
        subprocess.run(cmd, env=child_env, check=True)


def healthy(host: str, port: int, timeout: float = 1.0) -> bool:
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else f"[{host}]" if ":" in host else host
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False
