// NVFP4 GEMV launch declarations (mesh_kernels).
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

void launch_nvfp4_fma(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    at::Tensor& out,
    const float* alpha_dev,
    cudaStream_t stream);
