// MoE top-k routing, expert sort, and hidden gather.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

namespace {

template <typename T>
__device__ float logit_f(T v);

template <>
__device__ float logit_f(float v) {
  return v;
}

template <>
__device__ float logit_f(__half v) {
  return __half2float(v);
}

template <>
__device__ float logit_f(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename LogitT>
__device__ void warp_topk(
    const LogitT* row,
    int num_experts,
    int topk,
    int token,
    int* experts_out,
    int* tokens_out) {
  const int lane = threadIdx.x;
  int sel[32];
  for (int j = 0; j < topk; ++j) {
    float best = -INFINITY;
    int best_e = 0;
    for (int e = lane; e < num_experts; e += 32) {
      float lv = logit_f(row[e]);
      for (int t = 0; t < j; ++t) {
        if (e == sel[t]) {
          lv = -INFINITY;
        }
      }
      if (lv > best || (lv == best && e < best_e)) {
        best = lv;
        best_e = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffffffff, best, o);
      const int oe = __shfl_down_sync(0xffffffff, best_e, o);
      if (ov > best || (ov == best && oe < best_e)) {
        best = ov;
        best_e = oe;
      }
    }
    sel[j] = __shfl_sync(0xffffffff, best_e, 0);
  }
  if (lane == 0) {
    const int base = token * topk;
    for (int j = 0; j < topk; ++j) {
      experts_out[base + j] = sel[j];
      tokens_out[base + j] = token;
    }
  }
}

template <typename LogitT>
__global__ void topk_kernel(
    const LogitT* logits,
    int* experts,
    int* tokens,
    int num_tokens,
    int num_experts,
    int topk) {
  const int t = blockIdx.x;
  if (t >= num_tokens) {
    return;
  }
  warp_topk(logits + static_cast<long long>(t) * num_experts, num_experts, topk, t, experts, tokens);
}

__global__ void fill_keys(const int* experts, int* keys, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    keys[i] = experts[i];
  }
}

__global__ void fill_iota(int* v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    v[i] = i;
  }
}

__global__ void permute_idx(const int* ord, const int* in, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[ord[i]];
  }
}

template <typename ActT>
__global__ void gather_rows(
    const ActT* hidden,
    const int* tok,
    ActT* out,
    int rows,
    int dim) {
  const int r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const ActT* src = hidden + static_cast<long long>(tok[r]) * dim;
  ActT* dst = out + static_cast<long long>(r) * dim;
  const int bytes = dim * static_cast<int>(sizeof(ActT));
  if (bytes % 16 == 0) {
    const int n = bytes / 16;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      reinterpret_cast<uint4*>(dst)[i] = __ldg(reinterpret_cast<const uint4*>(src) + i);
    }
  } else {
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      dst[d] = src[d];
    }
  }
}

int sort_bits(int e) {
  int b = 0;
  for (int v = e - 1; v > 0; v >>= 1) {
    ++b;
  }
  return b > 0 ? b : 1;
}

template <typename LogitT>
void run_topk(
    const LogitT* logits,
    int* experts,
    int* tokens,
    int t,
    int e,
    int topk,
    cudaStream_t stream) {
  topk_kernel<<<t, 32, 0, stream>>>(logits, experts, tokens, t, e, topk);
}

void check_moe(
    const at::Tensor& hidden,
    const at::Tensor& router_logits,
    int64_t topk,
    int64_t& t,
    int64_t& h,
    int64_t& e) {
  TORCH_CHECK(hidden.is_cuda() && router_logits.is_cuda());
  TORCH_CHECK(hidden.dim() == 2 && router_logits.dim() == 2);
  TORCH_CHECK(hidden.is_contiguous() && router_logits.is_contiguous());
  t = hidden.size(0);
  h = hidden.size(1);
  e = router_logits.size(1);
  TORCH_CHECK(router_logits.size(0) == t);
  TORCH_CHECK(topk >= 1 && topk <= 32 && e >= 1 && e <= 512);
  TORCH_CHECK(hidden.scalar_type() == at::kHalf || hidden.scalar_type() == at::kBFloat16);
  const auto lt = router_logits.scalar_type();
  TORCH_CHECK(lt == at::kFloat || lt == at::kHalf || lt == at::kBFloat16);
}

at::Tensor custom_moe_dispatch_cuda(
    const at::Tensor& hidden,
    const at::Tensor& router_logits,
    int64_t topk) {
  int64_t nt = 0;
  int64_t hd = 0;
  int64_t ne = 0;
  check_moe(hidden, router_logits, topk, nt, hd, ne);
  const at::cuda::OptionalCUDAGuard guard(hidden.device());
  const int count = static_cast<int>(nt * topk);
  auto experts = at::empty({count}, hidden.options().dtype(at::kInt));
  auto tokens = at::empty({count}, hidden.options().dtype(at::kInt));
  auto keys = at::empty({count}, hidden.options().dtype(at::kInt));
  auto keys_out = at::empty({count}, hidden.options().dtype(at::kInt));
  auto vals = at::empty({count}, hidden.options().dtype(at::kInt));
  auto vals_out = at::empty({count}, hidden.options().dtype(at::kInt));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto lt = router_logits.scalar_type();
  if (lt == at::kFloat) {
    run_topk(router_logits.data_ptr<float>(), experts.data_ptr<int>(), tokens.data_ptr<int>(), static_cast<int>(nt), static_cast<int>(ne), static_cast<int>(topk), stream);
  } else if (lt == at::kHalf) {
    run_topk(reinterpret_cast<const __half*>(router_logits.const_data_ptr()), experts.data_ptr<int>(), tokens.data_ptr<int>(), static_cast<int>(nt), static_cast<int>(ne), static_cast<int>(topk), stream);
  } else {
    run_topk(reinterpret_cast<const __nv_bfloat16*>(router_logits.const_data_ptr()), experts.data_ptr<int>(), tokens.data_ptr<int>(), static_cast<int>(nt), static_cast<int>(ne), static_cast<int>(topk), stream);
  }
  const int tpb = 256;
  const int nb = (count + tpb - 1) / tpb;
  fill_keys<<<nb, tpb, 0, stream>>>(experts.data_ptr<int>(), keys.data_ptr<int>(), count);
  fill_iota<<<nb, tpb, 0, stream>>>(vals.data_ptr<int>(), count);
  size_t temp_bytes = 0;
  const int bits = sort_bits(static_cast<int>(ne));
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, keys.data_ptr<int>(), keys_out.data_ptr<int>(), vals.data_ptr<int>(), vals_out.data_ptr<int>(), count, 0, bits, stream);
  auto temp = at::empty({static_cast<int64_t>(temp_bytes)}, hidden.options().dtype(at::kByte));
  cub::DeviceRadixSort::SortPairs(
      temp.data_ptr<uint8_t>(), temp_bytes, keys.data_ptr<int>(), keys_out.data_ptr<int>(), vals.data_ptr<int>(), vals_out.data_ptr<int>(), count, 0, bits, stream);
  auto sorted_tok = at::empty({count}, hidden.options().dtype(at::kInt));
  permute_idx<<<nb, tpb, 0, stream>>>(vals_out.data_ptr<int>(), tokens.data_ptr<int>(), sorted_tok.data_ptr<int>(), count);
  auto out = at::empty({count, hd}, hidden.options());
  const dim3 ggrid(static_cast<unsigned>(count));
  const dim3 gblock(static_cast<unsigned>(std::min<int64_t>(256, hd)));
  if (hidden.scalar_type() == at::kHalf) {
    gather_rows<<<ggrid, gblock, 0, stream>>>(
        reinterpret_cast<const __half*>(hidden.const_data_ptr()),
        sorted_tok.data_ptr<int>(),
        reinterpret_cast<__half*>(out.data_ptr()),
        count,
        static_cast<int>(hd));
  } else {
    gather_rows<<<ggrid, gblock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr()),
        sorted_tok.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        count,
        static_cast<int>(hd));
  }
  return out;
}

at::Tensor custom_moe_dispatch_meta(
    const at::Tensor& hidden,
    const at::Tensor& router_logits,
    int64_t topk) {
  int64_t nt = 0;
  int64_t hd = 0;
  int64_t ne = 0;
  check_moe(hidden, router_logits, topk, nt, hd, ne);
  (void)ne;
  return at::empty({nt * topk, hd}, hidden.options());
}

}

TORCH_LIBRARY_IMPL(goethe_kernels, CUDA, m) {
  m.impl("custom_moe_dispatch", custom_moe_dispatch_cuda);
}

TORCH_LIBRARY_IMPL(goethe_kernels, Meta, m) {
  m.impl("custom_moe_dispatch", custom_moe_dispatch_meta);
}
