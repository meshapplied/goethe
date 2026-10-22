#!/usr/bin/env python3
"""Stage 1 compressed-tensors NVFP4 MoE Marlin vs dense Marlin and fp32 dequant refs."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from lib.mesh_test import gpu_flock, main_exit, require_gpus

E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

NUM_EXPERTS = 4
HIDDEN = 128
INTER = 256
TOPK = 2
BATCH_SIZES = (1, 2, 8, 64)
SEEDS = (11, 22, 33)
REL_L2_MAX = 0.03
COS_MIN = 0.999


def _decode_nvfp4_packed(weight_packed: torch.Tensor) -> torch.Tensor:
    low = (weight_packed & 0x0F).to(torch.long)
    high = ((weight_packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(weight_packed.shape[0], -1)
    mag = E2M1_LUT.to(codes.device)[codes & 7]
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    return (mag * sign).to(torch.float32)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    ref = expected.float().reshape(-1)
    act = actual.float().reshape(-1)
    denom = max(ref.norm().item(), 1e-6)
    rel_l2 = (act - ref).norm().item() / denom
    cos = torch.nn.functional.cosine_similarity(act.unsqueeze(0), ref.unsqueeze(0)).item()
    return rel_l2, cos


def _marlin_global_scale(stored: torch.Tensor) -> torch.Tensor:
    return (1.0 / stored.to(torch.float32)).reshape(1)


def _swap_up_gate_rows(w: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    up_w, gate_w = w.chunk(2, dim=0)
    up_s, gate_s = s.chunk(2, dim=0)
    return torch.cat([gate_w, up_w], dim=0), torch.cat([gate_s, up_s], dim=0)


def _prepare_dense_sidecar(
    n: int,
    k: int,
    packed: torch.Tensor,
    scale: torch.Tensor,
    stored_global: torch.Tensor,
    dtype: torch.dtype,
) -> nn.Module:
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_nvfp4_layer_for_marlin

    tmp = nn.Module()
    tmp.output_size_per_partition = n
    tmp.input_size_per_partition = k
    tmp.params_dtype = dtype
    tmp.weight = Parameter(packed.clone(), requires_grad=False)
    tmp.weight_scale = Parameter(scale.clone(), requires_grad=False)
    tmp.weight_global_scale = Parameter(
        _marlin_global_scale(stored_global).to(device=packed.device),
        requires_grad=False,
    )
    prepare_nvfp4_layer_for_marlin(tmp)
    return tmp


def _marlin_linear(x: torch.Tensor, side: nn.Module) -> torch.Tensor:
    from sglang.srt.layers.quantization.marlin_utils_fp4 import apply_fp4_marlin_linear

    n = side.output_size_per_partition
    k = side.input_size_per_partition
    return apply_fp4_marlin_linear(
        input=x,
        weight=side.weight,
        weight_scale=side.weight_scale,
        weight_global_scale=side.weight_global_scale,
        workspace=side.workspace,
        size_n=n,
        size_k=k,
        bias=None,
    ).view(x.shape[0], n)


def _reference_marlin_dense_moe(
    x: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    from sglang.kernels.ops.activation.activation import silu_and_mul

    side13: dict[int, nn.Module] = {}
    side2: dict[int, nn.Module] = {}
    for e in range(NUM_EXPERTS):
        w13_p = raw["w13_packed"][e]
        w13_s = raw["w13_scale"][e]
        w13_p, w13_s = _swap_up_gate_rows(w13_p, w13_s)
        g_stored = raw["w13_global"][e, 0]
        side13[e] = _prepare_dense_sidecar(2 * INTER, HIDDEN, w13_p, w13_s, g_stored, dtype)
        side2[e] = _prepare_dense_sidecar(
            HIDDEN, INTER, raw["w2_packed"][e], raw["w2_scale"][e], raw["w2_global"][e], dtype
        )

    topk = topk_ids.shape[1]
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        acc = torch.zeros(HIDDEN, device=x.device, dtype=dtype)
        xt = x[t : t + 1]
        for j in range(topk):
            e = int(topk_ids[t, j].item())
            wt = topk_weights[t, j].to(dtype)
            h13 = _marlin_linear(xt, side13[e])
            mid = torch.empty(1, INTER, device=x.device, dtype=dtype)
            silu_and_mul(h13, mid)
            acc = acc + wt * _marlin_linear(mid, side2[e]).squeeze(0)
        out[t] = acc
    return out


def _reference_fp32_dequant_moe(
    x: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    topk = topk_ids.shape[1]
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        acc = torch.zeros(HIDDEN, device=x.device, dtype=torch.float32)
        for j in range(topk):
            e = int(topk_ids[t, j].item())
            wt = topk_weights[t, j].to(torch.float32)
            w13_p = raw["w13_packed"][e]
            w13_s = raw["w13_scale"][e]
            w13 = _decode_nvfp4_packed(w13_p)
            scale_rep = w13_s.to(torch.float32).repeat_interleave(16, dim=1)[:, : w13.shape[1]]
            w13 = w13 * scale_rep / raw["w13_global"][e, 0].to(torch.float32)
            up, gate = w13.chunk(2, dim=0)
            mid = torch.nn.functional.silu(x[t].float() @ gate.T) * (x[t].float() @ up.T)
            w2 = _decode_nvfp4_packed(raw["w2_packed"][e])
            s2 = raw["w2_scale"][e].to(torch.float32).repeat_interleave(16, dim=1)[:, : w2.shape[1]]
            w2 = w2 * s2 / raw["w2_global"][e].to(torch.float32)
            acc = acc + wt * (mid @ w2.T)
        out[t] = acc.to(x.dtype)
    return out


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


class _QuantCfg:
    group_size = 16


class _MoeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN
        self.intermediate_size_per_partition = INTER
        self.params_dtype = torch.bfloat16
        self.quant_config = _QuantCfg()
        self.moe_ep_size = 1
        self.moe_ep_rank = 0
        self.moe_tp_size = 1
        self.moe_tp_rank = 0


def _fill_weights(layer: _MoeLayer, scheme, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scheme.create_weights(
        layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.w13_weight_packed.copy_(
            torch.randint(
                0,
                256,
                layer.w13_weight_packed.shape,
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
        )
        layer.w2_weight_packed.copy_(
            torch.randint(
                0,
                256,
                layer.w2_weight_packed.shape,
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
        )
        layer.w13_weight_scale.copy_(
            (torch.rand(layer.w13_weight_scale.shape, device="cuda", generator=gen) * 1.5 + 0.25).to(
                torch.float8_e4m3fn
            )
        )
        layer.w2_weight_scale.copy_(
            (torch.rand(layer.w2_weight_scale.shape, device="cuda", generator=gen) * 1.5 + 0.25).to(
                torch.float8_e4m3fn
            )
        )
        layer.w13_weight_global_scale.fill_(1.25)
        layer.w2_weight_global_scale.fill_(1.1)
        layer.w13_input_global_scale.fill_(2.0)
        layer.w2_input_global_scale.fill_(1.75)
    return {
        "w13_packed": layer.w13_weight_packed.data.clone(),
        "w13_scale": layer.w13_weight_scale.data.clone(),
        "w13_global": layer.w13_weight_global_scale.data.clone(),
        "w2_packed": layer.w2_weight_packed.data.clone(),
        "w2_scale": layer.w2_weight_scale.data.clone(),
        "w2_global": layer.w2_weight_global_scale.data.clone(),
    }


def _run_fused(
    scheme,
    layer: _MoeLayer,
    moe_cfg,
    batch: int,
    seed: int,
    raw: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    gen = torch.Generator(device="cuda").manual_seed(seed + batch)
    x = torch.randn(batch, HIDDEN, device="cuda", dtype=dtype, generator=gen)
    router = torch.randn(batch, NUM_EXPERTS, device="cuda", dtype=dtype, generator=gen)
    topk_w, topk_ids = torch.topk(router, TOPK, dim=-1)
    topk_w = torch.softmax(topk_w.float(), dim=-1)
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_w, topk_ids.to(torch.int32), router),
    )
    y = scheme.apply_weights(layer, dispatch).hidden_states
    raw_cuda = {k: v.cuda() for k, v in raw.items()}
    ref_m = _reference_marlin_dense_moe(x, raw_cuda, topk_ids, topk_w, dtype)
    ref_d = _reference_fp32_dequant_moe(x, raw_cuda, topk_ids, topk_w)
    return y, ref_m, ref_d, topk_w


def main() -> None:
    gpu = require_gpus(1)[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
        CompressedTensorsW4A4Nvfp4MoE,
    )

    stats: list[tuple[int, int, float, float, float, float]] = []
    with gpu_flock([gpu]), _env(MESH_MARLIN="1", MESH_FORCE_PATH="w4a16"):
        dtype = torch.bfloat16
        for seed in SEEDS:
            scheme = CompressedTensorsW4A4Nvfp4MoE()
            layer = _MoeLayer()
            moe_cfg = MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                top_k=TOPK,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTER,
                activation="silu",
                is_gated=True,
            )
            layer.moe_runner_config = moe_cfg
            raw = _fill_weights(layer, scheme, seed)
            layer.cuda()
            scheme.create_moe_runner(layer, moe_cfg)
            scheme.process_weights_after_loading(layer)

            for batch in BATCH_SIZES:
                y, ref_m, ref_d, _ = _run_fused(scheme, layer, moe_cfg, batch, seed, raw, dtype)
                rl_m, cos_m = _metrics(y, ref_m)
                rl_d, cos_d = _metrics(y, ref_d)
                stats.append((batch, seed, rl_m, cos_m, rl_d, cos_d))
                if rl_m > REL_L2_MAX or cos_m < COS_MIN:
                    raise RuntimeError(
                        f"fused vs dense Marlin fail M={batch} seed={seed} "
                        f"rel_l2={rl_m:.4f} cos={cos_m:.6f}"
                    )

    for batch, seed, rl_m, cos_m, rl_d, cos_d in stats:
        print(
            f"M={batch:2d} seed={seed} marlin rel_l2={rl_m:.4f} cos={cos_m:.6f} "
            f"dequant rel_l2={rl_d:.4f} cos={cos_d:.6f}"
        )
    print("stage1_moe ok")


if __name__ == "__main__":
    main_exit(main)
