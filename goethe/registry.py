"""models.toml: load and validate. Unknown or missing keys are errors, not silently ignored."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MODELS_TOML = Path(__file__).with_name("models.toml")
ENGINES = {"ninfer"}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    lanes: int
    context: int
    why: str


@dataclass(frozen=True)
class Model:
    id: str
    title: str
    engine: str
    served_name: str
    native_context: int
    recipe: str
    min_vram_gib: float
    tested: tuple[str, ...]
    base: dict[str, object]
    profiles: dict[str, Profile]


def _table(data: object, where: str, required: set[str]) -> dict:
    if not isinstance(data, dict):
        raise RegistryError(f"{where}: expected a table")
    missing, unknown = required - data.keys(), data.keys() - required
    if missing or unknown:
        raise RegistryError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return data


def _model(model_id: str, raw: object) -> Model:
    where = f"models.{model_id}"
    t = _table(raw, where, {"title", "engine", "served_name", "native_context", "recipe",
                            "min_vram_gib", "tested", "base", "profiles"})
    if t["engine"] not in ENGINES:
        raise RegistryError(f"{where}: engine must be one of {sorted(ENGINES)}")
    base = _table(t["base"], f"{where}.base", {"kv_dtype", "spec", "draft_tokens", "prefill_chunk"})
    profiles = {}
    for name, body in _table(t["profiles"], f"{where}.profiles", {"agent", "chat", "long"}).items():
        i = _table(body, f"{where}.profiles.{name}", {"lanes", "context", "why"})
        profiles[name] = Profile(name, i["lanes"], i["context"], i["why"])
    return Model(model_id, t["title"], t["engine"], t["served_name"], t["native_context"], t["recipe"],
                 t["min_vram_gib"], tuple(t["tested"]), base, profiles)


def load(path: Path = MODELS_TOML) -> dict[str, Model]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    models = _table(raw, "root", {"models"})["models"]
    return {mid: _model(mid, body) for mid, body in models.items()}


def find(models: dict[str, Model], name: str) -> Model:
    for m in models.values():
        if name in (m.id, m.served_name):
            return m
    raise RegistryError(f"unknown model {name!r}; known: {', '.join(models)}")


def read_pin(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a NINFER_PIN file."""
    return dict(line.strip().split("=", 1) for line in path.read_text().splitlines()
                if "=" in line and not line.startswith("#"))
