// PyTorch entry points for fused NVFP4 GEMV.

#include "nvfp4_common.cuh"
#include "nvfp4_launch.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {

at::Tensor fused_nvfp4_gemv_impl(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& alpha) {
  int64_t batch = 0;
  int64_t k = 0;
  int64_t n = 0;
  check_nvfp4_gemv(x, qweight, scales, batch, k, n);
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  auto out = at::empty({batch, n}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float* alpha_dev = nullptr;
  if (alpha.has_value()) {
    check_nvfp4_alpha(*alpha);
    alpha_dev = alpha->const_data_ptr<float>();
  }
  launch_nvfp4_fma(x, qweight, scales, out, alpha_dev, stream);
  return out;
}

at::Tensor fused_nvfp4_gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales) {
  return fused_nvfp4_gemv_impl(x, qweight, scales, c10::nullopt);
}

at::Tensor fused_nvfp4_gemv_scaled_cuda(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const at::Tensor& alpha) {
  return fused_nvfp4_gemv_impl(x, qweight, scales, alpha);
}

at::Tensor fused_nvfp4_gemv_meta(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales) {
  int64_t batch = 0;
  int64_t k = 0;
  int64_t n = 0;
  check_nvfp4_gemv(x, qweight, scales, batch, k, n);
  return at::empty({batch, n}, x.options());
}

at::Tensor fused_nvfp4_gemv_scaled_meta(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    const at::Tensor& alpha) {
  (void)alpha;
  return fused_nvfp4_gemv_meta(x, qweight, scales);
}

}

TORCH_LIBRARY_IMPL(mesh_kernels, CUDA, m) {
  m.impl("fused_nvfp4_gemv", fused_nvfp4_gemv_cuda);
  m.impl("fused_nvfp4_gemv.scaled", fused_nvfp4_gemv_scaled_cuda);
}

TORCH_LIBRARY_IMPL(mesh_kernels, Meta, m) {
  m.impl("fused_nvfp4_gemv", fused_nvfp4_gemv_meta);
  m.impl("fused_nvfp4_gemv.scaled", fused_nvfp4_gemv_scaled_meta);
}
