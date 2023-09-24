#include "bp5-backend.h"

static uint initialized = 0;
static const size_t local_size = 512;
static const char *ERR_STR_CUDA_FAILURE = "%s:%d CUDA %s failure: %s.\n";

#define check_error(FNAME, LINE, CALL, ERR_T, SUCCES, GET_ERR, OP)             \
  {                                                                            \
    ERR_T result_ = (CALL);                                                    \
    if (result_ != SUCCES) {                                                   \
      const char *msg = GET_ERR(result_);                                      \
      bp5_error(ERR_STR_CUDA_FAILURE, FNAME, LINE, OP, msg);                   \
    }                                                                          \
  }

#define check_driver(call)                                                     \
  check_error(__FILE__, __LINE__, call, cudaError_t, cudaSuccess,              \
              cudaGetErrorName, "driver");

static scalar *d_r, *d_x, *d_z, *d_p, *d_w;
static scalar *d_c, *d_g, *d_D;
static uint *d_gs_off, *d_gs_idx;
static scalar *d_wrk, *wrk;

static void cuda_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "cuda_mem_init: copy problem data to device ... ");

  const uint n = bp5_get_local_dofs(bp5);

  // Allocate device buffers and copy problem data to device.
  check_driver(cudaMalloc(&d_r, n * sizeof(scalar)));
  check_driver(cudaMalloc(&d_x, n * sizeof(scalar)));
  check_driver(cudaMalloc(&d_z, n * sizeof(scalar)));
  check_driver(cudaMalloc(&d_p, n * sizeof(scalar)));
  check_driver(cudaMalloc(&d_w, n * sizeof(scalar)));

  // Copy multiplicity array.
  check_driver(cudaMalloc(&d_c, n * sizeof(scalar)));
  check_driver(
      cudaMemcpy(d_c, bp5->c, n * sizeof(scalar), cudaMemcpyHostToDevice));

  // Copy geometric factors and derivative matrix.
  check_driver(cudaMalloc(&d_g, 6 * n * sizeof(scalar)));
  check_driver(
      cudaMemcpy(d_g, bp5->g, 6 * n * sizeof(scalar), cudaMemcpyHostToDevice));

  check_driver(cudaMalloc(&d_D, bp5->nx1 * bp5->nx1 * sizeof(scalar)));
  check_driver(cudaMemcpy(d_D, bp5->D, bp5->nx1 * bp5->nx1 * sizeof(scalar),
                          cudaMemcpyHostToDevice));

  // Copy gather-scatter offsets and indices.
  check_driver(cudaMalloc(&d_gs_off, (bp5->gs_n + 1) * sizeof(uint)));
  check_driver(cudaMemcpy(d_gs_off, bp5->gs_off, (bp5->gs_n + 1) * sizeof(uint),
                          cudaMemcpyHostToDevice));

  check_driver(cudaMalloc(&d_gs_idx, bp5->gs_off[bp5->gs_n] * sizeof(uint)));
  check_driver(cudaMemcpy(d_gs_idx, bp5->gs_idx,
                          bp5->gs_off[bp5->gs_n] * sizeof(uint),
                          cudaMemcpyHostToDevice));

  // Work array.
  wrk = bp5_calloc(scalar, n);
  check_driver(cudaMalloc(&d_wrk, n * sizeof(scalar)));

  bp5_debug(bp5->verbose, "done.\n");
}

__global__ static void mask_kernel(scalar *v) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0)
    v[i] = 0;
}

inline static void mask(scalar *d_v, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  mask_kernel<<<global_size, local_size>>>(d_v);
}

__global__ static void zero_kernel(scalar *v, const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i] = 0;
}

inline static void zero(scalar *d_v, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  zero_kernel<<<global_size, local_size>>>(d_v, n);
}

__global__ static void copy_kernel(scalar *out, const scalar *in,
                                   const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i];
}

inline static void copy(scalar *d_out, const scalar *d_in, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  copy_kernel<<<global_size, local_size>>>(d_out, d_in, n);
}

__global__ static void add2s1_kernel(scalar *a, const scalar *b, const scalar c,
                                     const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = c * a[i] + b[i];
}

inline static void add2s1(scalar *d_a, const scalar *d_b, const scalar c,
                          const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  add2s1_kernel<<<global_size, local_size>>>(d_a, d_b, c, n);
}

__global__ static void add2s2_kernel(scalar *a, const scalar *b, const scalar c,
                                     const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] += c * b[i];
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
    if (threadIdx.x < s)
      s_abc[threadIdx.x] += s_abc[threadIdx.x + s];
  }
  __syncthreads();
  out[blockIdx.x] = s_abc[0];
}

inline static scalar glsc3(const scalar *d_a, const scalar *d_b,
                           const scalar *d_c, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  glsc3_kernel<<<global_size, local_size, local_size * sizeof(scalar)>>>(
      d_wrk, d_a, d_b, d_c, n);
  check_driver(cudaDeviceSynchronize());

  check_driver(cudaMemcpy(wrk, d_wrk, global_size * sizeof(scalar),
                          cudaMemcpyDeviceToHost));
  for (uint i = 1; i < global_size; i++)
    wrk[0] += wrk[i];

  return wrk[0];
}

__global__ static void gs_kernel(scalar *v, const uint *gs_off,
                                 const uint *gs_idx, const uint n) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    scalar s = 0;
    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j)
      s += v[gs_idx[j]];
    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j)
      v[gs_idx[j]] = s;
  }
}

inline static void gs(scalar *d_v, const uint *d_gs_off, const uint *d_gs_idx,
                      const uint gs_n) {
  const size_t global_size = (gs_n + local_size - 1) / local_size;
  gs_kernel<<<global_size, local_size>>>(d_v, d_gs_off, d_gs_idx, gs_n);
}

__global__ static void ax_kernel_v00(scalar *w, const scalar *u,
                                     const scalar *G, const scalar *D,
                                     const uint nelt, const uint nx1) {
  const uint ebase = blockIdx.x * nx1 * nx1 * nx1;
  const uint i = threadIdx.x;
  const uint j = threadIdx.y;
  const uint k = threadIdx.z;

  extern __shared__ scalar smem[];
  scalar *s_D = (scalar *)smem;
  scalar *s_ur = (scalar *)&s_D[nx1 * nx1];
  scalar *s_us = (scalar *)&s_ur[nx1 * nx1 * nx1];
  scalar *s_ut = (scalar *)&s_us[nx1 * nx1 * nx1];

  s_ur[BP5_IDX3(i, j, k)] = 0;
  s_us[BP5_IDX3(i, j, k)] = 0;
  s_ut[BP5_IDX3(i, j, k)] = 0;
  if (k == 0)
    s_D[BP5_IDX2(i, j)] = D[BP5_IDX2(i, j)];
  __syncthreads();

  for (uint l = 0; l < nx1; ++l) {
    s_ur[BP5_IDX3(i, j, k)] +=
        s_D[BP5_IDX2(l, i)] * u[ebase + BP5_IDX3(l, j, k)];
    s_us[BP5_IDX3(i, j, k)] +=
        s_D[BP5_IDX2(l, j)] * u[ebase + BP5_IDX3(i, l, k)];
    s_ut[BP5_IDX3(i, j, k)] +=
        s_D[BP5_IDX2(l, k)] * u[ebase + BP5_IDX3(i, j, l)];
  }
  __syncthreads();

  const uint gbase = 6 * (ebase + BP5_IDX3(i, j, k));
  scalar r_G00 = G[gbase + 0];
  scalar r_G01 = G[gbase + 1];
  scalar r_G02 = G[gbase + 2];
  scalar r_G11 = G[gbase + 3];
  scalar r_G12 = G[gbase + 4];
  scalar r_G22 = G[gbase + 5];

  scalar wr = r_G00 * s_ur[BP5_IDX3(i, j, k)] +
              r_G01 * s_us[BP5_IDX3(i, j, k)] + r_G02 * s_ut[BP5_IDX3(i, j, k)];
  scalar ws = r_G01 * s_ur[BP5_IDX3(i, j, k)] +
              r_G11 * s_us[BP5_IDX3(i, j, k)] + r_G12 * s_ut[BP5_IDX3(i, j, k)];
  scalar wt = r_G02 * s_ur[BP5_IDX3(i, j, k)] +
              r_G12 * s_us[BP5_IDX3(i, j, k)] + r_G22 * s_ut[BP5_IDX3(i, j, k)];
  __syncthreads();

  s_ur[BP5_IDX3(i, j, k)] = wr;
  s_us[BP5_IDX3(i, j, k)] = ws;
  s_ut[BP5_IDX3(i, j, k)] = wt;
  __syncthreads();

  scalar wo = 0;
  for (uint l = 0; l < nx1; l++) {
    wo += s_D[BP5_IDX2(i, l)] * s_ur[BP5_IDX3(l, j, k)] +
          s_D[BP5_IDX2(j, l)] * s_us[BP5_IDX3(i, l, k)] +
          s_D[BP5_IDX2(k, l)] * s_ut[BP5_IDX3(i, j, l)];
  }
  w[ebase + BP5_IDX3(i, j, k)] = wo;
}

inline static void ax(scalar *d_w, const scalar *d_u, const scalar *d_g,
                      const scalar *d_D, const uint nelt, const uint nx1) {
  dim3 local_dim(nx1, nx1, nx1);
  dim3 global_dim(nelt);
  const size_t shared_size = (3 * nx1 * nx1 * nx1 + nx1 * nx1) * sizeof(scalar);
  ax_kernel_v00<<<global_dim, local_dim, shared_size>>>(d_w, d_u, d_g, d_D,
                                                        nelt, nx1);
}

static void cuda_init(const struct bp5_t *bp5) {
  if (initialized)
    return;
  bp5_debug(bp5->verbose, "cuda_init: initializing Cuda backend ...\n");

  int num_devices = 0;
  check_driver(cudaGetDeviceCount(&num_devices));
  if (bp5->device_id >= (uint)num_devices) {
    bp5_error("cuda_init: Invalid device id %d, only %d devices available.",
              bp5->device_id, num_devices);
  }

  check_driver(cudaSetDeviceFlags(cudaDeviceMapHost));
  check_driver(cudaFree(0));

  cuda_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "cuda_init: done.\n");
}

static scalar cuda_run(const struct bp5_t *bp5, const scalar *r) {
  if (!initialized)
    bp5_error("cuda_run: CUDA backend is not initialized.\n");

  const uint n = bp5_get_local_dofs(bp5);
  bp5_debug(bp5->verbose, "cuda_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  check_driver(cudaMemcpy(d_r, r, n * sizeof(scalar), cudaMemcpyHostToDevice));

  scalar pap = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(d_x, n);

  // Apply Dirichlet BCs to RHS.
  mask(d_r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(d_r, d_c, d_r, n)), r0 = rnorm;
  for (uint i = 0; i < bp5->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(d_z, d_r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(d_r, d_c, d_z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(d_p, d_z, beta, n);

    ax(d_w, d_p, d_g, d_D, bp5->nelt, bp5->nx1);
    gs(d_w, d_gs_off, d_gs_idx, bp5->gs_n);
    add2s2(d_w, d_p, 0.1, n);
    mask(d_w, n);

    pap = glsc3(d_w, d_c, d_p, n);

    scalar alpha = rtz1 / pap;
    add2s2(d_x, d_p, alpha, n);
    add2s2(d_r, d_w, -alpha, n);

    scalar rtr = glsc3(d_r, d_c, d_r, n);
    rnorm = sqrt(rtr);
    bp5_debug(bp5->verbose, "cuda_run: iteration %d, rnorm = %e\n", i, rnorm);
  }
  check_driver(cudaDeviceSynchronize());
  clock_t t1 = clock() - t0;

  bp5_debug(bp5->verbose, "cuda_run: done.\n");
  bp5_debug(bp5->verbose, "cuda_run: iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "cuda_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1) / CLOCKS_PER_SEC;
}

static void cuda_finalize(void) {
  if (!initialized)
    return;

  check_driver(cudaFree(d_r));
  check_driver(cudaFree(d_x));
  check_driver(cudaFree(d_z));
  check_driver(cudaFree(d_p));
  check_driver(cudaFree(d_w));
  check_driver(cudaFree(d_c));
  check_driver(cudaFree(d_g));
  check_driver(cudaFree(d_D));
  check_driver(cudaFree(d_gs_off));
  check_driver(cudaFree(d_gs_idx));
  check_driver(cudaFree(d_wrk));
  bp5_free(&wrk);

  initialized = 0;
}

BP5_INTERN void bp5_cuda_init(void) {
  bp5_register_backend("CUDA", cuda_init, cuda_run, cuda_finalize);
}
