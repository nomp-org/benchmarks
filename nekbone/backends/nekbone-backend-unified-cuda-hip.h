#define LOCAL_SIZE 512

#define bIdx(N) ((int)blockIdx.N)
#define tIdx(N) ((int)threadIdx.N)
#define gIdx(N) (bIdx(N) * LOCAL_SIZE + tIdx(N))

__global__ void __launch_bounds__(LOCAL_SIZE)
    mask_kernel(scalar *__restrict__ v) {
  if (gIdx(x) == 0) v[gIdx(x)] = 0;
}

inline static void mask(scalar *d_v, const int n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  mask_kernel<<<global_size, LOCAL_SIZE>>>(d_v);
}

__global__ void __launch_bounds__(LOCAL_SIZE)
    zero_kernel(scalar *__restrict__ v, const int n) {
  if (gIdx(x) < n) v[gIdx(x)] = 0;
}

inline static void zero(scalar *d_v, const int n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  zero_kernel<<<global_size, LOCAL_SIZE>>>(d_v, n);
}

__global__ void __launch_bounds__(LOCAL_SIZE)
    copy_kernel(scalar *__restrict__ a, const scalar *__restrict__ b,
                const int n) {
  if (gIdx(x) < n) a[gIdx(x)] = b[gIdx(x)];
}

inline static void copy(scalar *d_a, const scalar *d_b, const int n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  copy_kernel<<<global_size, LOCAL_SIZE>>>(d_a, d_b, n);
}

__global__ void __launch_bounds__(LOCAL_SIZE)
    add2s1_kernel(scalar *__restrict__ a, const scalar *__restrict__ b,
                  const scalar c, const int n) {
  if (gIdx(x) < n) a[gIdx(x)] = c * a[gIdx(x)] + b[gIdx(x)];
}

inline static void add2s1(scalar *d_a, const scalar *d_b, const scalar c,
                          const int n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  add2s1_kernel<<<global_size, LOCAL_SIZE>>>(d_a, d_b, c, n);
}

__global__ void __launch_bounds__(LOCAL_SIZE)
    add2s2_kernel(scalar *__restrict__ a, const scalar *__restrict__ b,
                  const scalar c, const int n) {
  if (gIdx(x) < n) a[gIdx(x)] += c * b[gIdx(x)];
}

inline static void add2s2(scalar *d_a, const scalar *d_b, const scalar c,
                          const int n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  add2s2_kernel<<<global_size, LOCAL_SIZE>>>(d_a, d_b, c, n);
}

#if LOCAL_SIZE == 512
__global__ void __launch_bounds__(LOCAL_SIZE)
    glsc3_kernel_loopy(double *__restrict__ wrk, double const *__restrict__ a,
                       double const *__restrict__ b,
                       double const *__restrict__ c, int const n) {
  __shared__ double acc_i_inner[512];
  double            neutral_i_inner;
  __shared__ double tmp_sum_0[512];

  if (-1 + -512 * bIdx(x) + -1 * tIdx(x) + n >= 0)
    tmp_sum_0[tIdx(x)] = a[512 * bIdx(x) + tIdx(x)] *
                         b[512 * bIdx(x) + tIdx(x)] *
                         c[512 * bIdx(x) + tIdx(x)];
  acc_i_inner[tIdx(x)] = 0.0;
  neutral_i_inner      = 0.0;
  __syncthreads();
  if (-1 + -512 * bIdx(x) + -1 * tIdx(x) + n >= 0)
    acc_i_inner[tIdx(x)] = neutral_i_inner + tmp_sum_0[tIdx(x)];
  __syncthreads();
  if (255 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[256 + tIdx(x)];
  __syncthreads();
  if (127 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[128 + tIdx(x)];
  __syncthreads();
  if (63 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[64 + tIdx(x)];
  __syncthreads();
  if (31 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[32 + tIdx(x)];
  __syncthreads();
  if (15 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[16 + tIdx(x)];
  __syncthreads();
  if (7 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[8 + tIdx(x)];
  __syncthreads();
  if (3 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[4 + tIdx(x)];
  __syncthreads();
  if (1 + -1 * tIdx(x) >= 0)
    acc_i_inner[tIdx(x)] = acc_i_inner[tIdx(x)] + acc_i_inner[2 + tIdx(x)];
  __syncthreads();
  if (tIdx(x) == 0) {
    acc_i_inner[0] = acc_i_inner[0] + acc_i_inner[1];
    wrk[bIdx(x)]   = acc_i_inner[0];
  }
}
#else
__global__ void __launch_bounds__(LOCAL_SIZE)
    glsc3_kernel_v0(scalar *wrk, const scalar *__restrict__ a,
                    const scalar *__restrict__ b, const scalar *__restrict__ c,
                    const uint n) {
  extern __shared__ scalar s_abc[];

  if (gIdx(x) < n)
    s_abc[threadIdx.x] = a[gIdx(x)] * b[gIdx(x)] * c[gIdx(x)];
  else
    s_abc[threadIdx.x] = 0;

  for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadIdx.x < s) s_abc[threadIdx.x] += s_abc[threadIdx.x + s];
  }
  __syncthreads();
  wrk[blockIdx.x] = s_abc[0];
}
#endif

inline static scalar glsc3(const scalar *d_a, const scalar *d_b,
                           const scalar *d_c, const uint n) {
  const size_t global_size = (n + LOCAL_SIZE - 1) / LOCAL_SIZE;
#if LOCAL_SIZE == 512
  glsc3_kernel_loopy<<<global_size, LOCAL_SIZE, LOCAL_SIZE * sizeof(scalar)>>>(
      d_wrk, d_a, d_b, d_c, n);
#else
  glsc3_kernel_v0<<<global_size, LOCAL_SIZE, LOCAL_SIZE * sizeof(scalar)>>>(
      d_wrk, d_a, d_b, d_c, n);
#endif
  check_runtime(unified_device_synchronize());

  check_runtime(unified_memcpy(wrk, d_wrk, global_size * sizeof(scalar),
                               unified_memcpy_device_to_host));
  for (uint i = 1; i < global_size; i++) wrk[0] += wrk[i];

  return wrk[0];
}

#define NX1 2
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 3
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 4
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 5
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 6
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 7
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 8
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 9
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1
#define NX1 10
#include "nekbone-backend-unified-cuda-hip-ax.h"
#undef NX1

inline static void ax(scalar *d_w, const scalar *d_u, const scalar *d_g,
                      const scalar *d_D, const uint nelt, const uint nx1) {
  static dim3 local  = dim3(nx1, nx1, nx1);
  static dim3 global = dim3(nelt);

  switch (nx1) {
  case 2: ax_kernel_v00_2<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 3: ax_kernel_v00_3<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 4: ax_kernel_v00_4<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 5: ax_kernel_v00_5<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 6: ax_kernel_v00_6<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 7: ax_kernel_v00_7<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 8: ax_kernel_v00_8<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 9: ax_kernel_v00_9<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  case 10: ax_kernel_v00_10<<<global, local>>>(d_w, d_u, d_g, d_D); break;
  default:
    nekbone_error("Ax kernel for nx1 = %d is not implemented.", nx1);
    break;
  }
}

__global__ void __launch_bounds__(LOCAL_SIZE)
    gs_kernel(scalar *__restrict__ v, const uint *__restrict__ gs_off,
              const uint *__restrict__ gs_idx, const int n) {
  if (gIdx(x) < n) {
    scalar s = 0;
    for (uint j = gs_off[gIdx(x)]; j < gs_off[gIdx(x) + 1]; ++j)
      s += v[gs_idx[j]];
    for (uint j = gs_off[gIdx(x)]; j < gs_off[gIdx(x) + 1]; ++j)
      v[gs_idx[j]] = s;
  }
}

inline static void gs(scalar *d_v, const uint *d_gs_off, const uint *d_gs_idx,
                      const uint gs_n) {
  const size_t global_size = (gs_n + LOCAL_SIZE - 1) / LOCAL_SIZE;
  gs_kernel<<<global_size, LOCAL_SIZE>>>(d_v, d_gs_off, d_gs_idx, gs_n);
}

#undef LOCAL_SIZE
#undef bIdx
#undef tIdx
#undef gIdx
