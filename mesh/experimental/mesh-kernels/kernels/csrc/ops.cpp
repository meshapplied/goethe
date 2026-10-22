// PyTorch custom op declarations for mesh_kernels.

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(mesh_kernels, m) {
  m.def("fused_nvfp4_gemv(Tensor x, Tensor qweight, Tensor scales) -> Tensor");
  m.def("fused_nvfp4_gemv.scaled(Tensor x, Tensor qweight, Tensor scales, Tensor alpha) -> Tensor");
  m.def("custom_moe_dispatch(Tensor hidden, Tensor router_logits, int topk) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
