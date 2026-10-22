#!/usr/bin/env python3
"""Stage 1 hermetic py-env and SGLang fork gate."""

from __future__ import annotations

import os
import subprocess
from lib.goethe_test import gpu_flock, main_exit, require_gpus, resolve_goethe_home, resolve_python, run, test_gpus


def main() -> None:
    """Verify Python 3.11 py-env, CUDA torch, SGLang fork, and hf CLI."""
    gpus = require_gpus(1)
    gpu = gpus[0]
    home = resolve_goethe_home()
    py = resolve_python()
    if not py.is_file():
        raise RuntimeError(f"missing py-env python at {py}")
    ver = run([str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
    if ver.stdout.strip() != "3.11":
        raise RuntimeError(f"expected Python 3.11, got {ver.stdout.strip()}")
    with gpu_flock([gpu]):
        cuda = run(
            [
                str(py),
                "-c",
                "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
    if "True" not in cuda.stdout.splitlines()[0]:
        raise RuntimeError("torch.cuda not available on test GPU")
    if not cuda.stdout.splitlines()[1].strip():
        raise RuntimeError("torch.version.cuda empty")
    imp = run([str(py), "-c", "import sglang; from sglang.version import __version__; print(__version__ or sglang.__file__)"], env=os.environ.copy())
    if imp.returncode != 0:
        raise RuntimeError(f"import sglang failed: {imp.stderr}")
    help_rr = run([str(py), "-m", "sglang.launch_server", "--help"], timeout=60)
    if help_rr.returncode != 0:
        raise RuntimeError("sglang.launch_server --help failed")
    hf = home / "py-env" / "bin" / "hf"
    if not hf.is_file():
        raise RuntimeError("hf CLI missing in py-env")
    hf_rr = run([str(hf), "download", "--help"], env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    if hf_rr.returncode != 0:
        raise RuntimeError("hf download --help failed")
    uv = home / "bin" / "uv"
    if not uv.is_file():
        uv_path = run(["bash", "-lc", "command -v uv"])
        if uv_path.returncode != 0 or not uv_path.stdout.strip():
            raise RuntimeError("uv not found")
    marlin = run(
        [
            str(py),
            "-c",
            "from sglang.srt.layers.quantization import goethe_marlin as m; "
            "import os; os.environ.pop('GOETHE_MARLIN', None); "
            "assert m.mode() in ('marlin_small', 'marlin_all'), m.mode()",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
    )
    if marlin.returncode != 0:
        raise RuntimeError(f"goethe Marlin patch not active: {marlin.stderr}")
    print("stage1_pyenv ok")


if __name__ == "__main__":
    main_exit(main)
