#!/usr/bin/env python3
"""Stage 3 ternary model install gate."""

from __future__ import annotations

import json
import os
import re

from lib.goethe_test import main_exit, require_cli, resolve_python, run_goethe, temp_home


def _parse_card(text: str) -> dict[str, str]:
    """Parse simple goethe_card.toml key = value lines."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r'^(\w+)\s*=\s*"([^"]*)"\s*$', line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> None:
    """Install a ternary registry entry and verify card fields and both HF trees."""
    goethe = require_cli("goethe")
    tiny = os.environ.get("GOETHE_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
    model_id = "goethe-test-ternary"
    with temp_home(stub_python=True, real_python=resolve_python()) as home:
        ternary_root = home / "goethe" / "ternary-kernels"
        ternary_root.mkdir(parents=True)
        (ternary_root / ".keep").write_text("", encoding="utf-8")
        reg = home / "registry_ternary.json"
        reg.write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": model_id,
                            "architecture": "moe",
                            "runtime": "ternary",
                            "pack_id": tiny,
                            "base_model_id": tiny,
                            "static_vram_gb": 1.0,
                            "kv_bytes_per_token": 512,
                            "min_total_vram_gb": 4.0,
                            "supported_sm": ["12.0", "12.1"],
                            "default_ctx": 512,
                            "default_max_conc": 2,
                            "default_tp": 1,
                            "kv_cache_dtype": "fp8_e4m3",
                            "env": {
                                "TERNARY_PACK_DIR": "$GOETHE_HOME/models/pack",
                                "TERNARY_MODEL_DIR": "$GOETHE_HOME/models/base",
                                "TERNARY_PROFILE": "moe",
                            },
                            "pythonpath": ["$GOETHE_HOME/goethe/ternary-kernels"],
                            "extra_sglang_args": ["--trust-remote-code"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_goethe(
            [str(goethe), "install", model_id, "--no-load"],
            env={"GOETHE_HOME": str(home), "GOETHE_REGISTRY_URL": f"file://{reg}"},
            timeout=600,
        )
        cards = list((home / "models").glob("*__*"))
        if len(cards) < 2:
            raise RuntimeError("expected pack and base under models/")
        card_path = next(p for p in cards if (p / "goethe_card.toml").is_file())
        card = _parse_card((card_path / "goethe_card.toml").read_text(encoding="utf-8"))
        for key in ("runtime", "pack_id", "base_model_id"):
            if card.get(key) in (None, ""):
                raise RuntimeError(f"goethe_card missing {key}")
        if card.get("runtime") != "ternary":
            raise RuntimeError("runtime must be ternary")
        if card.get("pack_id") != tiny or card.get("base_model_id") != tiny:
            raise RuntimeError("pack_id/base_model_id mismatch")
        body = (card_path / "goethe_card.toml").read_text(encoding="utf-8")
        if "TERNARY_PACK_DIR" not in body or "pythonpath" not in body.lower():
            raise RuntimeError("card missing env or pythonpath")
    print("stage3_ternary ok")


if __name__ == "__main__":
    main_exit(main)
