#!/usr/bin/env python3
"""Stage 1 NVFP4 Marlin routing (default-on, env switches)."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.parameter import Parameter

from lib.mesh_test import gpu_flock, main_exit, require_gpus

E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

SHAPES = [(17408, 5120), (5120, 17408), (34816, 5120), (100, 256)]
SMALL_M = (1, 3, 17, 32, 128)
LARGE_M = (129, 1024)
MARLIN_ALL_M_FULL = (1, 3, 17, 128, 129, 1024, 4096)
GRAPH_M = (1, 4, 16, 64)


def _publish_fp4(fp4_backend: str = "flashinfer_cutlass") -> None:
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy", fp4_gemm_runner_backend=fp4_backend)
    args.resolve_once()
    publish(args, role="scheduler")
    initialize_fp4_gemm_config()


def _decode_nvfp4_packed(weight_packed: torch.Tensor) -> torch.Tensor:
    low = (weight_packed & 0x0F).to(torch.long)
    high = ((weight_packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(weight_packed.shape[0], -1)
    mag = E2M1_LUT.to(codes.device)[codes & 7]
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    return (mag * sign).to(torch.float32)


def _reference_fp32(x, weight_packed, scales, weight_global_scale, bias):
    w = _decode_nvfp4_packed(weight_packed)
    k = w.shape[1]
    scale_rep = scales.to(torch.float32).repeat_interleave(16, dim=1)[:, :k]
    out = torch.matmul(x.to(torch.float32), (w * scale_rep).T)
    out = out / weight_global_scale.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def _within_tol(actual: torch.Tensor, expected: torch.Tensor, frac: float = 1e-2) -> bool:
    ref_f = expected.float()
    act_f = actual.float()
    out_max = max(ref_f.abs().max().item(), 1e-6)
    return (act_f - ref_f).abs().max().item() <= frac * out_max


def _marlin_sidecar_bytes(layer: nn.Module) -> int:
    total = 0
    for name in ("marlin_weight", "marlin_scale", "marlin_global_scale", "marlin_workspace"):
        if hasattr(layer, name):
            t = getattr(layer, name)
            total += t.numel() * t.element_size()
    return total


def _param_bytes(layer: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in layer.parameters())


class _Nvfp4Layer(nn.Module):
    def __init__(self, n: int, k: int) -> None:
        super().__init__()
        self.output_size_per_partition = n
        self.input_size_per_partition = k
        self.weight_packed = Parameter(torch.empty(n, k // 2, dtype=torch.uint8), requires_grad=False)
        self.weight_global_scale = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.weight_scale = Parameter(torch.empty(n, k // 16, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.input_global_scale = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.params_dtype = torch.bfloat16


def _fill_random(layer: _Nvfp4Layer, seed: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        layer.weight_packed.copy_(
            torch.randint(0, 256, layer.weight_packed.shape, dtype=torch.uint8, device="cuda", generator=gen)
        )
        layer.weight_scale.copy_(
            (torch.rand(layer.weight_scale.shape, device="cuda", generator=gen) * 1.5 + 0.25).to(torch.float8_e4m3fn)
        )
        layer.weight_global_scale.fill_(1.25)
        layer.input_global_scale.fill_(2.0)


@contextmanager
def _env(**kwargs: str | None):
    saved = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _scheme():
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
        CompressedTensorsW4A4Fp4,
    )

    return CompressedTensorsW4A4Fp4()


def _build(marlin_on: bool, n: int, k: int, seed: int, dtype: torch.dtype, force: bool):
    with _env(
        MESH_MARLIN=None if marlin_on else "0",
        MESH_FORCE_PATH="w4a16" if force else None,
    ):
        _publish_fp4()
        layer = _Nvfp4Layer(n, k).cuda()
        layer.params_dtype = dtype
        _fill_random(layer, seed)
        raw = {
            "weight_packed": layer.weight_packed.data.clone(),
            "weight_scale": layer.weight_scale.data.clone(),
            "weight_global_scale": layer.weight_global_scale.data.clone(),
        }
        _scheme().process_weights_after_loading(layer)
        return layer, raw


def _apply(
    layer,
    batch: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    seed: int,
    marlin_on: bool,
    force_all: bool = False,
):
    gen = torch.Generator(device="cuda").manual_seed(seed + batch)
    x = torch.randn(batch, k, device="cuda", dtype=dtype, generator=gen)
    bias = torch.randn(n, device="cuda", dtype=dtype, generator=gen)
    with _env(
        MESH_MARLIN=None if marlin_on else "0",
        MESH_FORCE_PATH="w4a16" if force_all else None,
    ):
        y = _scheme().apply_weights(layer, x, bias=bias)
    return y, x, bias


def _marlin_all_ms(n: int, k: int) -> tuple[int, ...]:
    if n * k <= 5120 * 5120:
        return MARLIN_ALL_M_FULL
    return (1, 17, 128, 1024)


def _graph_replay(
    layer, x, bias, marlin_on: bool, force_all: bool = False, reps: int = 20
) -> None:
    scheme = _scheme()
    g = torch.cuda.CUDAGraph()
    sx, sb = x.clone(), bias.clone()
    torch.cuda.synchronize()
    with _env(
        MESH_MARLIN=None if marlin_on else "0",
        MESH_FORCE_PATH="w4a16" if force_all else None,
    ):
        with torch.cuda.graph(g):
            out = scheme.apply_weights(layer, sx, bias=sb)
        torch.cuda.synchronize()
        eager = scheme.apply_weights(layer, x, bias=bias)
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    if not torch.equal(out, eager):
        raise RuntimeError("CUDA graph replay differs from eager")


def _large_m(n: int, k: int) -> tuple[int, ...]:
    if (n, k) == (100, 256):
        return ()
    if n * k > 5120 * 17408:
        return (129,)
    return LARGE_M


def _run_dtype(dtype: torch.dtype, seed: int, shapes: tuple[tuple[int, int], ...] | None = None) -> None:
    if shapes is None:
        shapes = tuple(SHAPES)
    n0, k0 = SHAPES[0]
    y_a, _, _ = _apply(_build(False, n0, k0, seed, dtype, False)[0], 4, n0, k0, dtype, seed, False)
    y_b, _, _ = _apply(_build(False, n0, k0, seed, dtype, False)[0], 4, n0, k0, dtype, seed, False)
    if not torch.equal(y_a, y_b):
        raise RuntimeError("MESH_MARLIN=0 not deterministic")

    layer_stock, _ = _build(False, n0, k0, seed, dtype, False)
    layer_dual, _ = _build(True, n0, k0, seed, dtype, False)
    if hasattr(layer_stock, "marlin_weight"):
        raise RuntimeError("MESH_MARLIN=0 must not allocate Marlin tensors")
    extra = _param_bytes(layer_dual) - _param_bytes(layer_stock)
    sidecar = _marlin_sidecar_bytes(layer_dual)
    if abs(extra - sidecar) > 256 * 1024:
        raise RuntimeError(f"dual layout overhead {extra} != sidecar {sidecar}")

    for n, k in shapes:
        layer_stock, _ = _build(False, n, k, seed, dtype, False)
        layer_mesh, raw = _build(True, n, k, seed, dtype, False)
        for m in SMALL_M:
            y, x, bias = _apply(layer_mesh, m, n, k, dtype, seed + m, True)
            ref = _reference_fp32(
                x, raw["weight_packed"], raw["weight_scale"], raw["weight_global_scale"], bias
            )
            if not _within_tol(y, ref):
                raise RuntimeError(f"default marlin tol fail {n}x{k} m={m} dtype={dtype}")
        for m in _large_m(n, k):
            y_mesh, _, _ = _apply(layer_mesh, m, n, k, dtype, seed + m, True)
            y_stock, _, _ = _apply(layer_stock, m, n, k, dtype, seed + m, False)
            if not torch.equal(y_mesh, y_stock):
                raise RuntimeError(f"large M stock path mismatch {n}x{k} m={m}")

        with _env(MESH_FORCE_PATH="w4a16"):
            layer_all, raw = _build(True, n, k, seed + 7, dtype, True)
            if layer_all.weight_packed.numel() != 0:
                raise RuntimeError("marlin_all must release stock tensors")
            ws_bytes = layer_all.marlin_workspace.numel() * layer_all.marlin_workspace.element_size()
            if _param_bytes(layer_all) != _marlin_sidecar_bytes(layer_all) - ws_bytes:
                raise RuntimeError("marlin_all memory must be Marlin tensors only")
            for m in _marlin_all_ms(n, k):
                y, x, bias = _apply(layer_all, m, n, k, dtype, seed + m, True, True)
                ref = _reference_fp32(
                    x, raw["weight_packed"], raw["weight_scale"], raw["weight_global_scale"], bias
                )
                if not _within_tol(y, ref):
                    raise RuntimeError(f"marlin_all tol fail {n}x{k} m={m}")

    layer_g, _ = _build(True, n0, k0, seed, dtype, False)
    _publish_fp4()
    for m in GRAPH_M:
        x = torch.randn(m, k0, device="cuda", dtype=dtype)
        bias = torch.randn(n0, device="cuda", dtype=dtype)
        _graph_replay(layer_g, x, bias, True)
    with _env(MESH_FORCE_PATH="w4a16"):
        layer_ga, _ = _build(True, n0, k0, seed + 9, dtype, True)
        for m in GRAPH_M:
            x = torch.randn(m, k0, device="cuda", dtype=dtype)
            bias = torch.randn(n0, device="cuda", dtype=dtype)
            _graph_replay(layer_ga, x, bias, True, True)


def main() -> None:
    gpu = require_gpus(1)[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    try:
        from sglang.srt.layers.quantization import marlin_utils_fp4

        _ = marlin_utils_fp4.apply_fp4_marlin_linear
    except ImportError as exc:
        raise RuntimeError("Marlin FP4 kernels missing") from exc
    from sglang.srt.layers.quantization.mesh_marlin import mode

    if mode() == "stock":
        raise RuntimeError("expected default-on Marlin routing")

    seed = 42
    with gpu_flock([gpu]):
        _run_dtype(torch.bfloat16, seed)
        _run_dtype(torch.float16, seed + 100, shapes=((100, 256),))
        n0, k0 = SHAPES[0]
        layer, raw = _build(True, n0, k0, seed + 50, torch.float16, False)
        y, x, bias = _apply(layer, 1, n0, k0, torch.float16, seed, True)
        ref = _reference_fp32(
            x, raw["weight_packed"], raw["weight_scale"], raw["weight_global_scale"], bias
        )
        if not _within_tol(y, ref):
            raise RuntimeError("fp16 spot check failed")
    print("stage1_layer ok")


if __name__ == "__main__":
    main_exit(main)
