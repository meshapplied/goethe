#!/usr/bin/env python3
"""Optional real NVFP4 model gate (MESH_TEST_REAL=1)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    ("capital", "What is the capital of France? Answer in one word.", ["Paris"]),
    ("arith", "What is 12 times 13? Reply with the number only.", ["156"]),
    ("code", "Write a Python function that returns the sum of two integers. One line.", ["def", "return"]),
    ("color", "What color is the sky on a clear day? One word.", ["blue"]),
    ("year", "In what year did World War II end? Four digits.", ["1945"]),
    ("json", 'Reply with valid JSON only: {"ok": true}', ['"ok"', "true"]),
]

NEW_TOKENS = 64
PORT = int(os.environ.get("MESH_TEST_E2E_PORT", "30142"))
GPU = os.environ.get("MESH_TEST_GPUS", "2,3").split(",")[0].strip()


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}")
    sys.exit(75)


def _pending(msg: str) -> None:
    print(f"PENDING: {msg}")
    sys.exit(77)


def _gpu_mem_mb() -> float:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            GPU,
        ],
        text=True,
    )
    return float(out.strip().split()[0])


def _model_id(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
        data = json.load(r)
    return data["data"][0]["id"]


def _wait_health(base: str, timeout_s: float = 900.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            _model_id(base)
            return True
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            pass
        time.sleep(3.0)
    return False


def _kill_pgid(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(4.0)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stream_chat(base: str, model: str, prompt: str) -> tuple[str, float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": NEW_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    tokens = 0
    text_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            obj = json.loads(chunk)
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                text_parts.append(piece)
                tokens += 1
    elapsed = max(time.perf_counter() - t0, 1e-6)
    return "".join(text_parts), tokens / elapsed, tokens


def _run_config(
    py: Path,
    model_path: Path,
    marlin_off: bool,
    force_w4a16: bool,
    mesh_home: Path,
) -> dict:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MESH_HOME"] = str(mesh_home)
    env.pop("PYTHONPATH", None)
    if marlin_off:
        env["MESH_MARLIN"] = "0"
    else:
        env.pop("MESH_MARLIN", None)
    if force_w4a16:
        env["MESH_FORCE_PATH"] = "w4a16"
    else:
        env.pop("MESH_FORCE_PATH", None)

    tag = "force" if force_w4a16 else ("stock" if marlin_off else "default")
    cmd = [
        str(py),
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--quantization",
        "compressed-tensors",
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "4096",
        "--max-running-requests",
        "8",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    log_dir = Path(tempfile.mkdtemp(prefix=f"mesh-test-{os.environ.get('MESH_GATE_RUN_ID', os.getpid())}-e2e-"))
    log_path = log_dir / f"{PORT}_{tag}.log"
    mem_before = _gpu_mem_mb()
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_file,
        preexec_fn=os.setsid,
    )
    base = f"http://127.0.0.1:{PORT}"
    try:
        if not _wait_health(base):
            err = log_path.read_text()[-4000:] if log_path.is_file() else ""
            raise RuntimeError(f"server failed to start: {err}")
        mem_loaded = _gpu_mem_mb()
        model = _model_id(base)
        for _name, prompt, needles in PROMPTS:
            text, _, _ = _stream_chat(base, model, prompt)
            low = text.lower()
            if not any(n.lower() in low for n in needles):
                raise RuntimeError(f"incoherent output for {prompt!r}: {text!r}")

        _, tps_c1, _ = _stream_chat(base, model, PROMPTS[0][1])
        _, tps_c4, _ = _stream_chat(base, model, PROMPTS[1][1])
        return {
            "config": tag,
            "mem_loaded_mb": mem_loaded,
            "mem_delta_mb": mem_loaded - mem_before,
            "tok_s_c1": tps_c1,
            "tok_s_c4": tps_c4,
        }
    finally:
        _kill_pgid(proc.pid)
        log_file.close()
        shutil.rmtree(log_dir, ignore_errors=True)
        time.sleep(8.0)


def main() -> None:
    if os.environ.get("MESH_TEST_REAL", "") != "1":
        _skip("set MESH_TEST_REAL=1 to run real-model gate")

    mesh_home = Path(os.environ.get("MESH_HOME", "/scratch/ternary-flash/mesh-home"))
    py = mesh_home / "py-env/bin/python"
    if not py.is_file():
        print("missing py-env", file=sys.stderr)
        sys.exit(1)

    model = Path(
        os.environ.get(
            "MESH_TEST_REAL_MODELS",
            "/scratch/models/staged/Qwen3.8-27B-NVFP4-NInfer",
        )
    )
    if not model.is_dir():
        _pending(f"model path missing: {model}")

    results = []
    for marlin_off, force in ((True, False), (False, False), (False, True)):
        print(f"starting config marlin_off={marlin_off} force={force}", flush=True)
        results.append(_run_config(py, model, marlin_off, force, mesh_home))

    print("\n| config | mem_loaded_mb | tok/s c=1 | tok/s c=4 |")
    print("|--------|---------------|-----------|-----------|")
    for row in results:
        print(
            f"| {row['config']} | {row['mem_loaded_mb']:.0f} | {row['tok_s_c1']:.1f} | {row['tok_s_c4']:.1f} |"
        )

    stock_mb = results[0]["mem_loaded_mb"]
    default_mb = results[1]["mem_loaded_mb"]
    dual_gb = max(0.0, (default_mb - stock_mb) / 1024.0)
    print(f"\ndual_layout_overhead_gb={dual_gb:.2f}")

    print("stage1_e2e_model ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
