"""Stage 1 mesh_kernels verification: exactness, graphs, and bandwidth benchmark."""

from __future__ import annotations

import glob
import os
import sys
import time

import torch

MESH_HOME = os.environ.get("MESH_HOME", "/scratch/ternary-flash/mesh-home")
MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
TARGET_GBS = {1: 1000.0, 2: 1000.0, 3: 1000.0, 4: 1000.0, 8: 1000.0, 16: 900.0, 32: 700.0}
TC_EXACT_B = {3, 4, 5, 8, 9, 16, 17, 32}


def _so_path() -> str:
    paths = glob.glob(os.path.join(MESH_HOME, "kernels", "sm120", "libmesh_kernels_sm120.so"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one sm120 .so, got {paths}")
    return paths[0]


def _load() -> None:
    torch.ops.load_library(_so_path())


def _dequant(qweight: torch.Tensor, scales: torch.Tensor, k: int) -> torch.Tensor:
    lo = (qweight & 0x0F).to(torch.int64)
    hi = ((qweight >> 4) & 0x0F).to(torch.int64)
    codes = torch.stack((lo, hi), dim=-1).reshape(qweight.shape[0], k)
    vals = MAG.to(device=qweight.device)[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    g = k // 16
    sc = scales.float()
    return (vals.view(qweight.shape[0], g, 16) * sc.unsqueeze(-1)).reshape(qweight.shape[0], k)


def ref_gemv_fp32(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    k = x.shape[1]
    w = _dequant(qweight, scales, k)
    return x.float() @ w.T


def ref_gemv(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return ref_gemv_fp32(x, qweight, scales).to(dtype=x.dtype)


def _exact_inputs(
    b: int, n: int, k: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32)
    x = xs[torch.randint(0, 5, (b, k), device=device)].to(dtype)
    qweight = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
    exps = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=torch.float32)
    sc_f = (2.0 ** exps[torch.randint(0, 5, (n, k // 16), device=device)]).clamp(0.25, 2.0)
    scales = sc_f.to(torch.float8_e4m3fn)
    return x, qweight, scales


def test_schemas() -> None:
    names = [s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("mesh_kernels::")]
    assert any(n == "mesh_kernels::fused_nvfp4_gemv" for n in names), names
    gemv = [s for s in torch._C._jit_get_all_schemas() if s.name == "mesh_kernels::fused_nvfp4_gemv"]
    assert len(gemv) == 2 and any(len(s.arguments) == 4 for s in gemv), gemv
    assert any("custom_moe_dispatch" in n for n in names), names
    print("mesh_kernels schemas:", names)


def _run_gemv(
    x: torch.Tensor,
    qw: torch.Tensor,
    sc: torch.Tensor,
    alpha: torch.Tensor | None,
    scaled: bool,
) -> torch.Tensor:
    if scaled:
        return torch.ops.mesh_kernels.fused_nvfp4_gemv.scaled(x, qw, sc, alpha)
    return torch.ops.mesh_kernels.fused_nvfp4_gemv(x, qw, sc)


def test_gemv_exact(device: torch.device) -> None:
    grid_nk = [(5120, 5120), (17408, 5120), (5120, 17408), (1000, 64), (4097, 192)]
    alphas = [None, torch.tensor([4.0], device=device, dtype=torch.float32)]
    for dtype in (torch.float16, torch.bfloat16):
        for b in sorted(TC_EXACT_B | {1, 2}):
            for n, k in grid_nk:
                x, qw, sc = _exact_inputs(b, n, k, dtype, device)
                ref = ref_gemv(x, qw, sc)
                for alpha in alphas:
                    got = _run_gemv(x, qw, sc, alpha, scaled=alpha is not None)
                    if alpha is not None:
                        ref_a = (ref.float() * alpha.item()).to(dtype)
                        assert torch.equal(got, ref_a), (b, n, k, dtype, alpha.item())
                    else:
                        assert torch.equal(got, ref), (b, n, k, dtype)
    print("gemv exact ok")


def test_gemv_random(device: torch.device) -> None:
    torch.manual_seed(7)
    grid = [(4, 5120, 5120), (8, 17408, 5120), (16, 5120, 17408), (32, 1000, 64)]
    for dtype in (torch.float16, torch.bfloat16):
        for b, n, k in grid:
            x = torch.randn(b, k, device=device, dtype=dtype)
            qw = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
            sc = torch.randn(n, k // 16, device=device).clamp(-448, 448).to(torch.float8_e4m3fn)
            got = torch.ops.mesh_kernels.fused_nvfp4_gemv(x, qw, sc)
            ref = ref_gemv_fp32(x, qw, sc)
            if dtype == torch.float16:
                ulp = torch.clamp(ref.abs(), min=1e-4) * (2**-10)
            else:
                ulp = torch.clamp(ref.abs(), min=1e-4) * (2**-7)
            delta = torch.abs(got.float() - ref)
            assert torch.all(delta <= ulp + 1e-3), (b, n, k, dtype, float(delta.max()))
    print("gemv random ok")


def ref_moe(hidden: torch.Tensor, router_logits: torch.Tensor, topk: int) -> torch.Tensor:
    t, _h = hidden.shape
    e = router_logits.shape[1]
    logits = router_logits.float()
    pairs: list[tuple[int, int]] = []
    for tok in range(t):
        row = logits[tok]
        order = sorted(range(e), key=lambda ex: (-row[ex].item(), ex))
        take = min(topk, e)
        for j in range(take):
            pairs.append((order[j], tok))
    pairs.sort()
    idx = torch.tensor([p[1] for p in pairs], device=hidden.device, dtype=torch.long)
    return hidden.index_select(0, idx)


def test_moe(device: torch.device) -> None:
    for t, e, topk in ((7, 8, 2), (128, 64, 8), (512, 512, 32)):
        hidden = torch.randn(t, 128, device=device, dtype=torch.float16)
        logits = torch.randn(t, e, device=device, dtype=torch.float32)
        got = torch.ops.mesh_kernels.custom_moe_dispatch(hidden, logits, topk)
        ref = ref_moe(hidden, logits, topk)
        assert torch.equal(got, ref), (t, e, topk)
    print("moe ok")


def test_cuda_graphs(device: torch.device) -> None:
    cases = [
        (1, 512, 512, False, False),
        (4, 512, 512, True, False),
        (8, 512, 512, False, True),
    ]
    for b, n, k, scaled, bf16 in cases:
        dtype = torch.bfloat16 if bf16 else torch.float16
        x, qw, sc = _exact_inputs(b, n, k, dtype, device)
        alpha = torch.tensor([0.25], device=device, dtype=torch.float32)
        ref = _run_gemv(x, qw, sc, alpha if scaled else None, scaled=scaled)
        out_g = torch.empty_like(ref)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_g = _run_gemv(x, qw, sc, alpha if scaled else None, scaled=scaled)
        g.replay()
        torch.cuda.synchronize()
        first = out_g.clone()
        for _ in range(100):
            g.replay()
            assert torch.equal(out_g, first)
        assert torch.equal(out_g, ref)
    print("cuda graph ok")


def _weight_bytes(n: int, k: int) -> int:
    return n * (k // 2 + k // 16)


def _median_us(fn, warmup: int, iters: int, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def bench_gemv(device: torch.device) -> None:
    for n, k in ((17408, 5120), (5120, 17408), (34816, 5120)):
        num_sets = max(6, (256 * 1024 * 1024) // _weight_bytes(n, k) + 1)
        weights = []
        dense_ws = []
        for _i in range(num_sets):
            qw = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
            sc = torch.randn(n, k // 16, device=device).clamp(-448, 448).to(torch.float8_e4m3fn)
            weights.append((qw, sc))
            dense_ws.append(_dequant(qw, sc, k).to(torch.bfloat16))
        byte_count = _weight_bytes(n, k)
        print("gemv benchmark N=%d K=%d weight_bytes=%d sets=%d" % (n, k, byte_count, num_sets))
        for b in (1, 2, 3, 4, 8, 16, 32):
            x = torch.randn(b, k, device=device, dtype=torch.float16)
            step = [0]

            def custom_fn(step=step, b=b, x=x):
                qw, sc = weights[step[0] % num_sets]
                step[0] += 1
                torch.ops.mesh_kernels.fused_nvfp4_gemv(x, qw, sc)

            us = _median_us(custom_fn, 20, 200)
            gbs = byte_count / (us * 1000.0)
            step_r = [0]
            x_bf = x.to(torch.bfloat16)

            def dense_fn(step=step_r, x_bf=x_bf):
                w = dense_ws[step[0] % num_sets]
                step[0] += 1
                torch.matmul(x_bf, w.T)

            ref_us = _median_us(dense_fn, 20, 200)
            speedup = ref_us / us
            print(
                "  B=%d custom %.1f us (%.1f GB/s) dense_bf16 %.1f us speedup %.2fx"
                % (b, us, gbs, ref_us, speedup)
            )
            if n == 17408 and k == 5120:
                tgt = TARGET_GBS.get(b, 0.0)
                if tgt > 0 and gbs < tgt:
                    print("WARNING: fused_nvfp4_gemv below TARGET %.0f GB/s for B=%d" % (tgt, b))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")
    t0 = time.time()
    _load()
    test_schemas()
    test_gemv_exact(device)
    test_gemv_random(device)
    test_moe(device)
    test_cuda_graphs(device)
    bench_gemv(device)
    print("wall_s %.1f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
