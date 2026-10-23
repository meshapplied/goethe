#!/usr/bin/env python3
"""Stage 1 hermetic py-env and SGLang fork gate (the interpreter named by GOETHE_PYTHON)."""

from __future__ import annotations

import shutil

from lib.goethe_test import gpu_flock, main_exit, resolve_python, roomiest_test_gpu, run


def main() -> None:
    """Verify Python 3.11, CUDA torch, the patched SGLang fork and the hf CLI of the py-env under test."""
    gpu = roomiest_test_gpu(6.0)
    py = resolve_python()
    if not py.is_file():
        raise RuntimeError(f"missing py-env python at {py}")
    ver = run([str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
    if ver.stdout.strip() != "3.11":
        raise RuntimeError(f"expected Python 3.11, got {ver.stdout.strip()}")
    with gpu_flock([gpu]):
        cuda = run(
            [str(py), "-c", "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"],
            env={"CUDA_VISIBLE_DEVICES": str(gpu)},
        )
    lines = cuda.stdout.splitlines()
    if len(lines) < 2 or lines[0] != "True" or not lines[1].startswith("13."):
        raise RuntimeError(f"torch must see the test GPU with CUDA 13: {cuda.stdout} {cuda.stderr[-300:]}")
    imp = run([str(py), "-c", "import sglang; from sglang.version import __version__; print(__version__)"])
    if imp.returncode != 0:
        raise RuntimeError(f"import sglang failed: {imp.stderr[-400:]}")
    if run([str(py), "-m", "sglang.launch_server", "--help"], timeout=120).returncode != 0:
        raise RuntimeError("sglang.launch_server --help failed")
    hf = py.parent / "hf"
    if not hf.is_file() or run([str(hf), "download", "--help"]).returncode != 0:
        raise RuntimeError(f"hf CLI missing or broken at {hf}")
    if not (py.parent / "uv").is_file() and shutil.which("uv") is None:
        raise RuntimeError("uv not found next to the py-env or on PATH")
    marlin = run(
        [
            str(py),
            "-c",
            "from sglang.srt.layers.quantization import goethe_marlin as m; "
            "assert m.mode() in ('marlin_small', 'marlin_all'), m.mode(); print(m.mode(), m.max_rows())",
        ],
        env={"CUDA_VISIBLE_DEVICES": str(gpu), "GOETHE_MARLIN": "1"},
    )
    if marlin.returncode != 0:
        raise RuntimeError(f"goethe Marlin patch not active: {marlin.stderr[-400:]}")
    print(f"stage1_pyenv ok python={py} route={marlin.stdout.strip()}")


if __name__ == "__main__":
    main_exit(main)
