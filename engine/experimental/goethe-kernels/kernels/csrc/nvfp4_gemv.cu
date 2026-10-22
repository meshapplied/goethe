// Fused NVFP4 E2M1 grouped GEMV FMA path for batch size 1..2.

#include "nvfp4_common.cuh"
#include "nvfp4_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
#endif

namespace {

constexpr int kWarpsPerBlock = 8;

template <int R>
__device__ __host__ constexpr int rows_per_block() {
  return kWarpsPerBlock * R;
}
constexpr float kFp4ExpBias = 16384.f;

template <typename ActT>
struct ActOps;

template <>
struct ActOps<__half> {
  static __device__ inline float to_float(__half v) {
    return __half2float(v);
  }
  static __device__ inline __half from_float(float v) {
    return __float2half_rn(v);
  }
};

template <>
struct ActOps<__nv_bfloat16> {
  static __device__ inline float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  static __device__ inline __nv_bfloat16 from_float(float v) {
    return __float2bfloat16_rn(v);
  }
};

__device__ inline float2 fp4_pair_from_word(uint32_t word, int j) {
  const uint32_t v = (word >> (4 * j)) & 0x000F000Fu;
  const uint32_t h_bits = ((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12);
  const __half2 h = *reinterpret_cast<const __half2*>(&h_bits);
  return __half22float2(h);
}

__device__ inline float2 fp8_pair_to_scaled_float2(uint16_t pair) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const __half2 h2 = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  float2 f = __half22float2(h2);
  f.x *= kFp4ExpBias;
  f.y *= kFp4ExpBias;
  return f;
#else
  return make_float2(0.f, 0.f);
#endif
}

template <int R, int B>
__device__ inline void warp_reduce_acc(float acc[R][B], int lane) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float sum = acc[r][b];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      if (lane == 0) {
        acc[r][b] = sum;
      }
    }
  }
}

template <typename ActT, int B, int KC, int R>
__global__ __launch_bounds__(256, 2) void nvfp4_gemv_kernel(
    const ActT* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const uint8_t* __restrict__ scales,
    ActT* __restrict__ out,
    const float* __restrict__ alpha,
    int batch,
    int n_rows,
    int k,
    int k_half,
    int k_groups) {
  extern __shared__ float xs[];
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_k = lane * 32;
  const int row0 = blockIdx.x * rows_per_block<R>() + warp_id * R;
  int rows[R];
  bool store_row[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = row0 + r;
    store_row[r] = row < n_rows;
    rows[r] = store_row[r] ? row : (n_rows > 0 ? n_rows - 1 : 0);
  }
  const int chunks = (k + KC - 1) / KC;
  float acc[R][B];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      acc[r][b] = 0.f;
    }
  }
  for (int c = 0; c < chunks; ++c) {
    const int k_chunk = (c == chunks - 1) ? k - c * KC : KC;
    const int vec_k = (k_chunk + 7) & ~7;
    for (int tid = threadIdx.x; tid < B * (vec_k / 8); tid += blockDim.x) {
      const int b = tid / (vec_k / 8);
      const int kl = (tid % (vec_k / 8)) * 8;
      const int gk = c * KC + kl;
      float4 s0 = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 s1 = make_float4(0.f, 0.f, 0.f, 0.f);
      if (b < batch && kl < k_chunk) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(x + static_cast<long long>(b) * k + gk));
        const ActT* p = reinterpret_cast<const ActT*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          reinterpret_cast<float*>(&s0)[i] = ActOps<ActT>::to_float(p[i]);
          reinterpret_cast<float*>(&s1)[i] = ActOps<ActT>::to_float(p[i + 4]);
        }
      }
      if (kl + 4 <= KC) {
        reinterpret_cast<float4*>(&xs[b * KC + kl])[0] = s0;
        reinterpret_cast<float4*>(&xs[b * KC + kl + 4])[0] = s1;
      }
    }
    __syncthreads();
    if (lane_k < k_chunk) {
      const int k_base = c * KC + lane_k;
      const int byte_off = k_base >> 1;
      const int scale_off = k_base >> 4;
      uint4 w_reg[R];
      uint16_t sc_reg[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const long long row = rows[r];
        w_reg[r] = __ldcs(reinterpret_cast<const uint4*>(qweight + row * k_half + byte_off));
        sc_reg[r] = __ldcs(reinterpret_cast<const uint16_t*>(scales + row * k_groups + scale_off));
      }
      const uint32_t* words_r[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        words_r[r] = reinterpret_cast<const uint32_t*>(&w_reg[r]);
      }
      float gscale_r[R][2];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float2 gs = fp8_pair_to_scaled_float2(sc_reg[r]);
        gscale_r[r][0] = gs.x;
        gscale_r[r][1] = gs.y;
      }
      float gsum[R][B];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const int k_off = lane_k + g * 16;
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
          for (int b = 0; b < B; ++b) {
            gsum[r][b] = 0.f;
          }
        }
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          const int k0 = k_off + w * 8;
#pragma unroll
          for (int b = 0; b < B; ++b) {
            const float4 xa = *reinterpret_cast<const float4*>(&xs[b * KC + k0]);
            const float4 xb = *reinterpret_cast<const float4*>(&xs[b * KC + k0 + 4]);
            const float* xaf = reinterpret_cast<const float*>(&xa);
            const float* xbf = reinterpret_cast<const float*>(&xb);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
              for (int r = 0; r < R; ++r) {
                const float2 fp = fp4_pair_from_word(words_r[r][g * 2 + w], j);
                gsum[r][b] = fmaf(fp.x, xaf[j], gsum[r][b]);
                gsum[r][b] = fmaf(fp.y, xbf[j], gsum[r][b]);
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const float group_scale = gscale_r[r][g];
#pragma unroll
          for (int b = 0; b < B; ++b) {
            acc[r][b] += gsum[r][b] * group_scale;
          }
        }
      }
    }
    __syncthreads();
  }
  warp_reduce_acc<R, B>(acc, lane);
  const float ascale = alpha ? alpha[0] : 1.f;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (!store_row[r]) {
        continue;
      }
#pragma unroll
      for (int b = 0; b < B; ++b) {
        if (b < batch) {
          out[static_cast<long long>(b) * n_rows + row0 + r] =
              ActOps<ActT>::from_float(acc[r][b] * ascale);
        }
      }
    }
  }
}

template <typename ActT, int B, int KC, int R>
void launch_gemv(
    const ActT* x,
    const uint8_t* qweight,
    const uint8_t* scales,
    ActT* out,
    const float* alpha,
    int batch,
    int n_rows,
    int k,
    cudaStream_t stream) {
  const int blocks = (n_rows + rows_per_block<R>() - 1) / rows_per_block<R>();
  const size_t smem = static_cast<size_t>(B) * KC * sizeof(float);
  nvfp4_gemv_kernel<ActT, B, KC, R><<<blocks, 256, smem, stream>>>(
      x,
      qweight,
      scales,
      out,
      alpha,
      batch,
      n_rows,
      k,
      k / 2,
      k / 16);
}

template <typename ActT>
void dispatch_gemv(
    const ActT* x,
    const uint8_t* qweight,
    const uint8_t* scales,
    ActT* out,
    const float* alpha,
    int batch,
    int n_rows,
    int k,
    cudaStream_t stream) {
  int bucket = 32;
  if (batch <= 1) {
    bucket = 1;
  } else if (batch <= 2) {
    bucket = 2;
  } else if (batch <= 4) {
    bucket = 4;
  } else if (batch <= 8) {
    bucket = 8;
  } else if (batch <= 16) {
    bucket = 16;
  }
  switch (bucket) {
    case 1:
      launch_gemv<ActT, 1, 1024, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
    case 2:
      launch_gemv<ActT, 2, 1024, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
    case 4:
      launch_gemv<ActT, 4, 1024, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
    case 8:
      launch_gemv<ActT, 8, 1024, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
    case 16:
      launch_gemv<ActT, 16, 512, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
    default:
      launch_gemv<ActT, 32, 256, 4>(x, qweight, scales, out, alpha, batch, n_rows, k, stream);
      break;
  }
}

}

void launch_nvfp4_fma(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scales,
    at::Tensor& out,
    const float* alpha_dev,
    cudaStream_t stream) {
  const int batch = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(qweight.size(0));
  const uint8_t* scale_ptr = reinterpret_cast<const uint8_t*>(scales.const_data_ptr());
  if (x.scalar_type() == at::kHalf) {
    dispatch_gemv<__half>(
        reinterpret_cast<const __half*>(x.const_data_ptr()),
        qweight.data_ptr<uint8_t>(),
        scale_ptr,
        reinterpret_cast<__half*>(out.data_ptr()),
        alpha_dev,
        batch,
        n,
        k,
        stream);
  } else {
    dispatch_gemv<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        qweight.data_ptr<uint8_t>(),
        scale_ptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        alpha_dev,
        batch,
        n,
        k,
        stream);
  }
}
