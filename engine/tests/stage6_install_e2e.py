#!/usr/bin/env python3
"""Stage 6 install.sh end-to-end gate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    Pending,
    Skip,
    free_port,
    home_goethe_env,
    http_get,
    http_post,
    main_exit,
    require_cli,
    run_goethe,
    spawn,
    temp_home,
    test_gpus,
    test_public_port,
)


def _goethe_bin(home: Path) -> Path:
    local = home / "bin" / "goethe"
    return local if local.is_file() else require_cli("goethe")


def _stage_hub_model(home: Path, model: str, hf_home: str) -> None:
    """Copy a cached Hugging Face model into GOETHE_HOME/models for offline load."""
    dest = home / "models" / model.replace("/", "__")
    if dest.is_dir():
        return
    real_py = Path(os.environ.get("GOETHE_PYTHON", "/scratch/ternary-flash/goethe-home/py-env/bin/python"))
    code = (
        "import os\n"
        "from huggingface_hub import snapshot_download\n"
        "from pathlib import Path\n"
        f"os.environ['HF_HOME'] = {hf_home!r}\n"
        f"p = Path(snapshot_download({model!r}))\n"
        "print(p)\n"
    )
    rr = subprocess.run([str(real_py), "-c", code], capture_output=True, text=True, timeout=600)
    if rr.returncode != 0:
        raise RuntimeError(f"snapshot_download failed: {rr.stderr[-400:]}")
    src = Path(rr.stdout.strip().splitlines()[-1])
    shutil.copytree(src, dest)


def main() -> None:
    """Serve CDN install.sh and verify rootless install layout."""
    if os.environ.get("GOETHE_TEST_E2E", "0") != "1":
        raise Skip("set GOETHE_TEST_E2E=1 for installer e2e gate")
    cdn = Path(os.environ.get("GOETHE_TEST_CDN", f"{GOETHE_PKG}/dist/cdn"))
    install_sh = cdn / "install.sh"
    if not install_sh.is_file():
        raise Pending("engine/cdn/install.sh not packaged yet")
    port = free_port()
    srv = spawn(
        [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(cdn.parent if (cdn / "SHA256SUMS").is_file() else cdn),
        label="cdn-http",
    )
    pkg_url = f"http://127.0.0.1:{port}/{cdn.name if (cdn / 'SHA256SUMS').is_file() else ''}".rstrip("/")
    time.sleep(0.3)
    try:
        with temp_home(stub_python=True) as home:
            env = {**home_goethe_env(home), "GOETHE_PKG_URL": pkg_url, "GOETHE_NO_LINK": "1"}
            try:
                subprocess.run(["bash", str(install_sh)], env=env, check=True, timeout=900, capture_output=True)
                for sub in ("py-env", "bin", "nginx", "models", "logs"):
                    if not (home / sub).exists():
                        raise RuntimeError(f"missing {sub} after install")
                goethe = _goethe_bin(home)
                menv = home_goethe_env(home)
                hf_home = os.environ.get(
                    "GOETHE_TEST_HF_HOME",
                    "/scratch/ternary-flash/goethe-home/.cache/huggingface",
                )
                menv["HF_HOME"] = hf_home
                menv["HF_HUB_OFFLINE"] = "1"
                run_goethe([str(goethe), "hw", "profile"], env=menv)
                if not (home / "hardware_state.json").is_file():
                    raise RuntimeError("hardware_state.json missing after install")
                model = os.environ.get("GOETHE_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
                _stage_hub_model(home, model, hf_home)
                gpu = test_gpus()[-1]
                run_goethe(
                    [str(goethe), "install", model, "--load", "--gpus", str(gpu)],
                    env=menv,
                    timeout=900,
                    input_text="y\n",
                )
                key_rr = run_goethe([str(goethe), "key", "create", "--name", "e2e", "--raw"], env=menv)
                token = [ln for ln in key_rr.stdout.splitlines() if ln.startswith("goethe_sk_")][0]
                run_goethe([str(goethe), "allowlist", "add", "127.0.0.1/32"], env=menv)
                pub = test_public_port(home)
                code, _ = http_get(f"https://127.0.0.1:{pub}/v1/models", headers={"Authorization": f"Bearer {token}"})
                if code != 200:
                    raise RuntimeError(f"public API after install returned {code}")
                body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
                code, out = http_post(
                    f"https://127.0.0.1:{pub}/v1/chat/completions",
                    body,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if code != 200 or b"choices" not in out:
                    raise RuntimeError(f"completion failed after install: code={code} body={out[:200]!r}")
                run_goethe([str(goethe), "unload", "--all"], env=menv)
            finally:
                run_goethe([str(_goethe_bin(home)), "stop"], env=home_goethe_env(home))
        with temp_home(stub_python=True, tag="fallback") as home2:
            env2 = {"GOETHE_HOME": str(home2), "GOETHE_PKG_URL": pkg_url, "GOETHE_NO_LINK": "1"}
            try:
                subprocess.run(["bash", str(install_sh)], env=env2, check=True, timeout=900, capture_output=True)
                if not (home2 / "goethe.sock").exists():
                    raise RuntimeError("goethed socket missing after port-fallback install")
                goethe2 = _goethe_bin(home2)
                rr = run_goethe([str(goethe2), "ps", "--json"], env={"GOETHE_HOME": str(home2)})
                json.loads(rr.stdout or "[]")
            finally:
                run_goethe([str(_goethe_bin(home2)), "stop"], env={"GOETHE_HOME": str(home2)})
    finally:
        srv.stop()
    print("stage6_install_e2e ok")


if __name__ == "__main__":
    main_exit(main)
