// Shared NVFP4 GEMV checks and device decode helpers.

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
#endif

inline void check_nvfp4_gemv(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    int64_t& batch,
    int64_t& k,
    int64_t& n) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  batch = x.size(0);
  k = x.size(1);
  n = qweight.size(0);
  TORCH_CHECK(batch >= 1 && batch <= 32, "batch must be in [1, 32]");
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(qweight.size(1) == k / 2, "qweight K/2 mismatch");
  TORCH_CHECK(scales.size(0) == n, "scales N mismatch");
  TORCH_CHECK(scales.size(1) == k / 16, "scales K/16 mismatch");
  TORCH_CHECK(qweight.scalar_type() == at::kByte, "qweight must be uint8");
  TORCH_CHECK(scales.scalar_type() == at::kFloat8_e4m3fn, "scales must be float8_e4m3fn");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "x must be fp16 or bf16");
}

inline void check_nvfp4_alpha(const at::Tensor& alpha) {
  TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
  TORCH_CHECK(alpha.numel() == 1, "alpha must be scalar");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be fp32");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
}
