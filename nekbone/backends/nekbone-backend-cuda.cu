#include "nekbone-backend.h"

static uint initialized = 0;

#define check_error(FNAME, LINE, CALL, ERR_T, SUCCES, GET_ERR, OP)             \
  {                                                                            \
    ERR_T result_ = (CALL);                                                    \
    if (result_ != SUCCES) {                                                   \
      const char *msg = GET_ERR(result_);                                      \
      nekbone_error("%s:%d CUDA %s failure: %s.\n", FNAME, LINE, OP, msg);     \
    }                                                                          \
  }

#define check_runtime(call)                                                    \
  check_error(__FILE__, __LINE__, call, cudaError_t, cudaSuccess,              \
              cudaGetErrorName, "driver");

static scalar *d_r, *d_x, *d_z, *d_p, *d_w;
static scalar *d_wrk, *wrk;
static scalar *d_c, *d_g, *d_D;
static uint   *d_gs_off, *d_gs_idx;

static void cuda_mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "cuda_mem_init: copy problem data to device ... ");

  const uint n = nekbone_get_local_dofs(nekbone);

  // Allocate device buffers and copy problem data to device.
  check_runtime(cudaMalloc(&d_r, n * sizeof(scalar)));
  check_runtime(cudaMalloc(&d_x, n * sizeof(scalar)));
  check_runtime(cudaMalloc(&d_z, n * sizeof(scalar)));
  check_runtime(cudaMalloc(&d_p, n * sizeof(scalar)));
  check_runtime(cudaMalloc(&d_w, n * sizeof(scalar)));

  // Copy multiplicity array.
  check_runtime(cudaMalloc(&d_c, n * sizeof(scalar)));
  check_runtime(
      cudaMemcpy(d_c, nekbone->c, n * sizeof(scalar), cudaMemcpyHostToDevice));

  // Copy geometric factors and derivative matrix.
  check_runtime(cudaMalloc(&d_g, 6 * n * sizeof(scalar)));
  check_runtime(cudaMemcpy(d_g, nekbone->g, 6 * n * sizeof(scalar),
                           cudaMemcpyHostToDevice));

  check_runtime(cudaMalloc(&d_D, nekbone->nx1 * nekbone->nx1 * sizeof(scalar)));
  check_runtime(cudaMemcpy(d_D, nekbone->D,
                           nekbone->nx1 * nekbone->nx1 * sizeof(scalar),
                           cudaMemcpyHostToDevice));

  // Copy gather-scatter offsets and indices.
  check_runtime(cudaMalloc(&d_gs_off, (nekbone->gs_n + 1) * sizeof(uint)));
  check_runtime(cudaMemcpy(d_gs_off, nekbone->gs_off,
                           (nekbone->gs_n + 1) * sizeof(uint),
                           cudaMemcpyHostToDevice));

  check_runtime(
      cudaMalloc(&d_gs_idx, nekbone->gs_off[nekbone->gs_n] * sizeof(uint)));
  check_runtime(cudaMemcpy(d_gs_idx, nekbone->gs_idx,
                           nekbone->gs_off[nekbone->gs_n] * sizeof(uint),
                           cudaMemcpyHostToDevice));

  // Work array.
  wrk = nekbone_calloc(scalar, n);
  check_runtime(cudaMalloc(&d_wrk, n * sizeof(scalar)));

  nekbone_debug(nekbone->verbose, "done.\n");
}

#define unified_device_synchronize    cudaDeviceSynchronize
#define unified_memcpy                cudaMemcpy
#define unified_memcpy_device_to_host cudaMemcpyDeviceToHost
#include "nekbone-backend-unified-cuda-hip.h"
#undef unified_device_synchronize
#undef unified_memcpy
#undef unified_memcpy_device_to_host

static void cuda_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "cuda_init: initializing cuda backend ...\n");

  int num_devices = 0;
  check_runtime(cudaGetDeviceCount(&num_devices));
  if (nekbone->device >= (uint)num_devices) {
    nekbone_error("cuda_init: Invalid device id %d, only %d devices available.",
                  nekbone->device, num_devices);
  }

  check_runtime(cudaSetDeviceFlags(cudaDeviceMapHost));
  check_runtime(cudaFree(0));

  cuda_mem_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "cuda_init: done.\n");
}

static scalar cuda_run(const struct nekbone_t *nekbone, const scalar *r) {
  if (!initialized)
    nekbone_error("cuda_run: cuda backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "cuda_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  check_runtime(cudaMemcpy(d_r, r, n * sizeof(scalar), cudaMemcpyHostToDevice));

  scalar pap  = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(d_x, n);

  // Apply Dirichlet BCs to RHS.
  mask(d_r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(d_r, d_c, d_r, n));
  nekbone_debug(nekbone->verbose, "cuda_run: iteration 0, rnorm = %e\n", rnorm);
  scalar r0 = rnorm;
  for (uint i = 0; i < nekbone->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(d_z, d_r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(d_r, d_c, d_z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0) beta = 0;
    add2s1(d_p, d_z, beta, n);

    ax(d_w, d_p, d_g, d_D, nekbone->nelt, nekbone->nx1);
    gs(d_w, d_gs_off, d_gs_idx, nekbone->gs_n);
    add2s2(d_w, d_p, 0.1, n);
    mask(d_w, n);

    pap = glsc3(d_w, d_c, d_p, n);

    scalar alpha = rtz1 / pap;
    add2s2(d_x, d_p, alpha, n);
    add2s2(d_r, d_w, -alpha, n);

    scalar rtr = glsc3(d_r, d_c, d_r, n);
    rnorm      = sqrt(rtr);
    nekbone_debug(nekbone->verbose - 1, "cuda_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }

  check_runtime(cudaDeviceSynchronize());
  clock_t t1 = clock();

  nekbone_debug(nekbone->verbose, "cuda_run: done.\n");
  nekbone_debug(nekbone->verbose, "cuda_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "cuda_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void cuda_finalize(void) {
  if (!initialized) return;

  check_runtime(cudaFree(d_r));
  check_runtime(cudaFree(d_x));
  check_runtime(cudaFree(d_z));
  check_runtime(cudaFree(d_p));
  check_runtime(cudaFree(d_w));
  check_runtime(cudaFree(d_c));
  check_runtime(cudaFree(d_g));
  check_runtime(cudaFree(d_D));
  check_runtime(cudaFree(d_gs_off));
  check_runtime(cudaFree(d_gs_idx));
  check_runtime(cudaFree(d_wrk));
  nekbone_free(&wrk);

  initialized = 0;
}

NEKBONE_INTERN void nekbone_cuda_init(void) {
  nekbone_register_backend("CUDA", cuda_init, cuda_run, cuda_finalize);
}
