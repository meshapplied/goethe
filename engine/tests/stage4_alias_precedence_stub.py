#!/usr/bin/env python3
"""Stage 4: a model id beats an alias spelled the same; install and scan drop such an alias with a warning (N2)."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

from lib.goethe_test import (
    gpu_flock,
    home_goethe_env,
    http_post,
    main_exit,
    require_cli,
    roomiest_test_gpu,
    run,
    run_goethe,
    seed_installed_model,
    sqlite_query,
    start_goethed,
    stub_requests,
    temp_home,
    test_mux_port,
    write_card,
)

CFG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "vocab_size": 128,
}
FAILS: list[str] = []


def _weights(folder: Path) -> Path:
    """A minimal model folder goethe can install or scan: config.json plus one valid safetensors file."""
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
    header = json.dumps({"w": {"dtype": "BF16", "shape": [32], "data_offsets": [0, 64]}}).encode()
    header += b" " * (-len(header) % 8)
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(64))
    return folder


def _plant_alias(home: Path, alias: str, target: str) -> None:
    """An alias row as an older goethe could have left it, before a model with the same id existed."""
    con = sqlite3.connect(str(home / "state.db"))
    try:
        con.execute("INSERT OR REPLACE INTO model_aliases (alias, model_id) VALUES (?, ?)", (alias, target))
        con.commit()
    finally:
        con.close()


def _drops_alias(home: Path, alias: str, rr_text: str, rc: int, what: str) -> None:
    """After installing a model named like an alias, the alias is gone and the user was told."""
    left = sqlite_query(home / "state.db", "SELECT model_id FROM model_aliases WHERE alias=?", (alias,))
    if rc != 0 or left or "alias" not in rr_text.lower():
        FAILS.append(f"{what} must drop alias {alias!r} with a warning: exit {rc}, left {left}, {rr_text[-200:]!r}")


def _routing(goethe: str, home: Path, env: dict[str, str], gpu: int) -> None:
    """With alias b -> a and model b both present, a request for b reaches b's engine, not a's."""
    _plant_alias(home, "b", "a")
    seed_installed_model(home, "b", "models/org__b", 5.0, 512, 512, 2, 1)
    for mid in ("a", "b"):
        run_goethe([goethe, "load", "--model", mid, "--gpus", str(gpu), "--yes"], env=env, timeout=120)
    out = run_goethe([goethe, "key", "--create", "prec", "--raw"], env=env).stdout
    token = next(ln for ln in out.splitlines() if ln.startswith("goethe_sk_"))
    body = json.dumps({"model": "b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}).encode()
    url = f"http://127.0.0.1:{test_mux_port(home)}/v1/chat/completions"
    code, reply = http_post(url, body, headers={"Authorization": f"Bearer {token}"})
    served = {m: sum(r["path"].startswith("/v1/chat") for r in stub_requests(home / "models" / f"org__{m}"))
              for m in ("a", "b")}
    if code != 200 or served != {"a": 0, "b": 1}:
        raise RuntimeError(f"model id b must win over alias b -> a: code {code}, chats {served} {reply[:120]!r}")


def main() -> None:
    """Routing precedence first, then install and scan of models spelled like existing aliases."""
    goethe = str(require_cli("goethe"))
    gpu = roomiest_test_gpu()
    with gpu_flock([gpu]), temp_home(stub_python=True, tag="aliasprec") as home:
        env = home_goethe_env(home)
        for mid in ("a", "b"):
            write_card(home / "models" / f"org__{mid}", mid, 512, 2)
        seed_installed_model(home, "a", "models/org__a", 5.0, 512, 512, 2, 1)
        md = start_goethed(home)
        try:
            _routing(goethe, home, env, gpu)
            _plant_alias(home, "c", "a")
            src = _weights(home / "src-c")
            rr = run([goethe, "install", "--model", str(src), "--name", "c", "--no-load"], env=env, timeout=120)
            _drops_alias(home, "c", rr.stdout + rr.stderr, rr.returncode, "goethe install")
            _plant_alias(home, "local/scanned", "a")
            _weights(home / "models" / "scanned")
            rr = run([goethe, "scan", "--yes"], env=env, timeout=120)
            _drops_alias(home, "local/scanned", rr.stdout + rr.stderr, rr.returncode, "goethe scan --yes")
        finally:
            run([goethe, "unload", "--all"], env=env, timeout=60)
            md.stop()
    if FAILS:
        raise RuntimeError(" | ".join(FAILS))
    print("stage4_alias_precedence_stub ok")


if __name__ == "__main__":
    main_exit(main)
