__global__ static void mask_kernel(scalar *v) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) v[i] = 0;
}

inline static void mask(scalar *d_v, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  mask_kernel<<<global_size, local_size>>>(d_v);
}

__global__ static void zero_kernel(scalar *v, const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = 0;
}

inline static void zero(scalar *d_v, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  zero_kernel<<<global_size, local_size>>>(d_v, n);
}

__global__ static void copy_kernel(scalar *out, const scalar *in,
                                   const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

inline static void copy(scalar *d_out, const scalar *d_in, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  copy_kernel<<<global_size, local_size>>>(d_out, d_in, n);
}

__global__ static void add2s1_kernel(scalar *a, const scalar *b, const scalar c,
                                     const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = c * a[i] + b[i];
}

inline static void add2s1(scalar *d_a, const scalar *d_b, const scalar c,
                          const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  add2s1_kernel<<<global_size, local_size>>>(d_a, d_b, c, n);
}

__global__ static void add2s2_kernel(scalar *a, const scalar *b, const scalar c,
                                     const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += c * b[i];
}

inline static void add2s2(scalar *d_a, const scalar *d_b, const scalar c,
                          const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  add2s2_kernel<<<global_size, local_size>>>(d_a, d_b, c, n);
}

__global__ static void glsc3_kernel(scalar *out, const scalar *a,
                                    const scalar *b, const scalar *c,
                                    const uint n) {
  extern __shared__ scalar s_abc[];

  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    s_abc[threadIdx.x] = a[i] * b[i] * c[i];
  else
    s_abc[threadIdx.x] = 0;

  for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadIdx.x < s) s_abc[threadIdx.x] += s_abc[threadIdx.x + s];
  }
  __syncthreads();
  out[blockIdx.x] = s_abc[0];
}

inline static scalar glsc3(const scalar *d_a, const scalar *d_b,
                           const scalar *d_c, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  glsc3_kernel<<<global_size, local_size, local_size * sizeof(scalar)>>>(
      d_wrk, d_a, d_b, d_c, n);
  check_driver(unifiedDeviceSynchronize());

  check_driver(unifiedMemcpy(wrk, d_wrk, global_size * sizeof(scalar),
                             unifiedMemcpyDeviceToHost));
  for (uint i = 1; i < global_size; i++) wrk[0] += wrk[i];

  return wrk[0];
}

__global__ static void gs_kernel(scalar *v, const uint *gs_off,
                                 const uint *gs_idx, const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    scalar s = 0;
    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j) s += v[gs_idx[j]];
    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j) v[gs_idx[j]] = s;
  }
}

inline static void gs(scalar *d_v, const uint *d_gs_off, const uint *d_gs_idx,
                      const uint gs_n) {
  const size_t global_size = (gs_n + local_size - 1) / local_size;
  gs_kernel<<<global_size, local_size>>>(d_v, d_gs_off, d_gs_idx, gs_n);
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
  dim3 local_dim(nx1, nx1, nx1);
  dim3 global_dim(nelt);
  const size_t shared_size = (3 * nx1 * nx1 * nx1 + nx1 * nx1) * sizeof(scalar);
  switch (nx1) {
  case 2:
    ax_kernel_v00_2<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 3:
    ax_kernel_v00_3<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 4:
    ax_kernel_v00_4<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 5:
    ax_kernel_v00_5<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 6:
    ax_kernel_v00_6<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 7:
    ax_kernel_v00_7<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 8:
    ax_kernel_v00_8<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 9:
    ax_kernel_v00_9<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D);
    break;
  case 10:
    ax_kernel_v00_10<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g,
                                                             d_D);
    break;
  default:
    nekbone_error("Ax kernel for nx1 = %d is not implemented.", nx1);
    break;
  }
}
