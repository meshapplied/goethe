"""Where goethe keeps things. XDG locations, with the pre-CLI nvfp4-kernels cache reused when it exists."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _xdg(env: Mapping[str, str], var: str, fallback: str) -> Path:
    return Path(env.get(var) or Path(env.get("HOME", str(Path.home()))) / fallback)


def data_dir(env: Mapping[str, str] = os.environ) -> Path:
    return _xdg(env, "XDG_DATA_HOME", ".local/share") / "goethe"


def state_dir(env: Mapping[str, str] = os.environ) -> Path:
    return _xdg(env, "XDG_STATE_HOME", ".local/state") / "goethe"


def config_path(env: Mapping[str, str] = os.environ) -> Path:
    return _xdg(env, "XDG_CONFIG_HOME", ".config") / "goethe" / "config.toml"


def recipe_home(env: Mapping[str, str] = os.environ) -> Path:
    """NVFP4_HOME of the nvfp4-kernels scripts: explicit, else the existing cache, else under the data dir."""
    if env.get("NVFP4_HOME"):
        return Path(env["NVFP4_HOME"])
    legacy = Path(env.get("HOME", str(Path.home()))) / ".cache" / "goethe" / "nvfp4"
    return legacy if legacy.is_dir() else data_dir(env) / "nvfp4"
