"""Hardware x model x profile -> the exact server settings, each with where it came from and why."""
from __future__ import annotations

import os
import secrets
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from goethe import store
from goethe.hw import Gpu
from goethe.registry import Model

PROFILES = ("agent", "chat", "long")
KV_DTYPES = ("fp8", "bf16", "int8", "nvfp4", "k8v4")
LOOPBACK = ("127.0.0.1", "localhost", "::1")
DRAFT_TOKENS = {"dflash2": (1, 15), "mtp": (1, 5)}


class ResolveError(Exception):
    pass


def _int(low: int, high: int | None = None):
    def check(value: object) -> int:
        try:
            n = int(str(value))
        except ValueError:
            raise ResolveError(f"{value!r} is not an integer") from None
        if n < low or (high is not None and n > high):
            raise ResolveError(f"{n} is outside {low}..{high if high is not None else 'inf'}")
        return n
    return check


def _choice(options: tuple[str, ...]):
    def check(value: object) -> str:
        if value not in options:
            raise ResolveError(f"{value!r} is not one of {', '.join(options)}")
        return str(value)
    return check


def _text(value: object) -> str:
    if not str(value):
        raise ResolveError("must not be empty")
    return str(value)


VALIDATORS = {
    "host": _text,
    "port": _int(1, 65535),
    "lanes": _int(1, 8),
    "context": _int(2048),
    "spec": _choice(tuple(DRAFT_TOKENS)),
    "draft_tokens": _int(1, 15),
    "kv_dtype": _choice(KV_DTYPES),
    "api_key": _text,
}
CONFIG_SERVE_KEYS = {"host", "port", "api_key"}
CONFIG_PROFILE_KEYS = {"lanes", "context", "spec", "draft_tokens", "kv_dtype"}


@dataclass
class Setting:
    value: int | str
    source: str
    why: str = ""


@dataclass(frozen=True)
class Tier:
    name: str
    reason: str


@dataclass
class Plan:
    model: Model
    profile: str
    gpu: Gpu
    tier: Tier
    settings: dict[str, Setting]
    extra_args: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def value(self, key: str):
        return self.settings[key].value


def tier_for(model: Model, gpu: Gpu) -> Tier:
    if model.engine == "ninfer" and gpu.compute_cap != "12.0":
        return Tier("unsupported", f"NInfer builds only for sm_120a (compute capability 12.0); "
                                   f"{gpu.name} is {gpu.compute_cap}")
    if model.min_vram_gib is not None and gpu.total_gib < model.min_vram_gib:
        return Tier("unsupported", f"{model.title} needs a GPU with at least {model.min_vram_gib:g} GiB, "
                                   f"{gpu.name} has {gpu.total_gib:.1f}")
    if gpu.name in model.tested:
        return Tier("tested", "we ran this")
    if model.engine == "ninfer":
        return Tier("compatible", "Blackwell, enough memory. we have not run it")
    return Tier("compatible", "we have not run it")


def pick_gpu(model: Model, gpus: list[Gpu]) -> tuple[Gpu, Tier]:
    if not gpus:
        raise ResolveError("no NVIDIA GPU found")
    tiers = [(g, tier_for(model, g)) for g in gpus]
    usable = [(g, t) for g, t in tiers if t.name != "unsupported"]
    if not usable:
        raise ResolveError(tiers[0][1].reason)
    return usable[0]


def load_config(path: Path) -> dict:
    """config.toml: [serve] host/port/api_key and [profiles.<name>] overrides. Anything else is an error."""
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    serve, profiles = raw.pop("serve", {}), raw.pop("profiles", {})
    if raw:
        raise ResolveError(f"{path}: unknown top-level keys {sorted(raw)}")
    tables = {"serve": (serve, CONFIG_SERVE_KEYS)}
    for name, body in profiles.items():
        if name not in PROFILES:
            raise ResolveError(f"{path}: unknown profile [profiles.{name}]; profiles: {', '.join(PROFILES)}")
        tables[f"profiles.{name}"] = (body, CONFIG_PROFILE_KEYS)
    for name, (body, allowed) in tables.items():
        if set(body) - allowed:
            raise ResolveError(f"{path}: [{name}] has unknown keys {sorted(set(body) - allowed)}; "
                               f"allowed: {', '.join(sorted(allowed))}")
    return {name: body for name, (body, _) in tables.items()}


def env_overrides(env: Mapping[str, str]) -> dict[str, str]:
    return {k: env[f"GOETHE_{k.upper()}"] for k in VALIDATORS if f"GOETHE_{k.upper()}" in env}


def resolve(model: Model, gpus: list[Gpu], profile: str, *, flags: Mapping[str, str] | None = None,
            env: Mapping[str, str] = os.environ, config: dict | None = None,
            extra_args: list[str] | None = None, probe: bool = True) -> Plan:
    """Layers, lowest first: model base, profile, config.toml, environment, command-line flags."""
    if gpus:
        gpu, tier = pick_gpu(model, gpus)
    elif probe:
        raise ResolveError("no NVIDIA GPU found")
    else:
        gpu = Gpu(0, "not probed", 0, 0, "", "0.0", None, None)
        tier = Tier("unknown", "not probed")
    chosen = model.profiles[profile]
    merged: dict[str, Setting] = {
        "host": Setting("127.0.0.1", "default", "loopback only"),
        "port": Setting(8080, "default"),
        "lanes": Setting(chosen.lanes, "profile", chosen.why),
        "context": Setting(chosen.context, "profile", chosen.why),
        "kv_dtype": Setting(model.base["kv_dtype"], "default"),
        "spec": Setting(model.base["spec"], "default"),
        "draft_tokens": Setting(model.base["draft_tokens"], "default"),
    }
    config = config or {}
    layers = [("config", {**config.get("serve", {}), **config.get(f"profiles.{profile}", {})}),
              ("env", env_overrides(env)), ("flag", dict(flags or {}))]
    for source, values in layers:
        for key, raw in values.items():
            try:
                merged[key] = Setting(VALIDATORS[key](raw), source)
            except KeyError:
                raise ResolveError(f"unknown setting {key!r}; settings: {', '.join(VALIDATORS)}") from None
            except ResolveError as e:
                raise ResolveError(f"{key} from {source}: {e}") from None

    spec = merged["spec"].value
    low, high = DRAFT_TOKENS[spec]
    if not low <= merged["draft_tokens"].value <= high:
        raise ResolveError(f"draft_tokens {merged['draft_tokens'].value} is outside {low}..{high} for spec {spec}")
    if merged["context"].value > model.native_context:
        raise ResolveError(f"context {merged['context'].value} is above the model's {model.native_context}")

    plan = Plan(model, profile, gpu, tier, merged, list(extra_args or []))
    if merged["kv_dtype"].value != "fp8":
        plan.warnings.append(f"kv_dtype {merged['kv_dtype'].value} changes numerics; only fp8 is measured")
    if merged["host"].value not in LOOPBACK and "api_key" not in merged:
        key_file = store.state_dir(env) / "api_key"
        merged["api_key"] = Setting(_api_key(key_file), "generated", f"host is not loopback; key kept in {key_file}")
    return plan


def _api_key(path: Path) -> str:
    if path.is_file() and path.read_text().strip():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = "gk-" + secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key


def server_argv(plan: Plan, binary: Path, model_file: Path, redact: bool = False) -> list[str]:
    """ninfer-serve command line; same flags and order as nvfp4-kernels/serve.sh."""
    s, b = plan.settings, plan.model.base
    argv = [str(binary), str(model_file), "--host", str(s["host"].value), "--port", str(s["port"].value),
            "--model-id", plan.model.served_name, "--max-context", str(s["context"].value),
            "--kv-capacity", "auto", "--max-concurrency", str(s["lanes"].value),
            "--kv-dtype", str(s["kv_dtype"].value), "--spec", str(s["spec"].value),
            "--draft-tokens", str(s["draft_tokens"].value), "--lm-head-draft",
            "--prefill-chunk", str(b["prefill_chunk"])]
    if "api_key" in s:
        argv += ["--api-key", "<hidden>" if redact else str(s["api_key"].value)]
    if plan.gpu.index != 0:
        argv += ["--device", str(plan.gpu.index)]
    return argv + plan.extra_args
