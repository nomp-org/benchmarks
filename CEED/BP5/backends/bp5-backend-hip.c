#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "bp5-backend.h"

static uint initialized = 0;
static const size_t local_size = 512;
static const char *ERR_STR_HIP_FAILURE = "%s:%d HIP %s failure: %s.\n";

#define check_error(FNAME, LINE, CALL, ERR_T, SUCCES, GET_ERR, OP)             \
  {                                                                            \
    ERR_T result_ = (CALL);                                                    \
    if (result_ != SUCCES) {                                                   \
      const char *msg = GET_ERR(result_);                                      \
      bp5_error(ERR_STR_HIP_FAILURE, FNAME, LINE, OP, msg);                    \
    }                                                                          \
  }

#define check_driver(call)                                                     \
  check_error(__FILE__, __LINE__, call, hipError_t, hipSuccess,                \
              hipGetErrorName, "driver")

static scalar *d_r, *d_x, *d_z, *d_p, *d_w;
static scalar *d_c, *d_g, *d_D;
static uint *d_gs_off, *d_gs_idx;
static scalar *d_wrk, *wrk;

static void hip_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "hip_mem_init: copy problem data to device ... ");

  const uint n = bp5_get_local_dofs(bp5);

  // Allocate device buffers and copy problem data to device.
  check_driver(hipMalloc((void **)&d_r, n * sizeof(scalar)));
  check_driver(hipMalloc((void **)&d_x, n * sizeof(scalar)));
  check_driver(hipMalloc((void **)&d_z, n * sizeof(scalar)));
  check_driver(hipMalloc((void **)&d_p, n * sizeof(scalar)));
  check_driver(hipMalloc((void **)&d_w, n * sizeof(scalar)));

  // Copy multiplicity array.
  check_driver(hipMalloc((void **)&d_c, n * sizeof(scalar)));
  check_driver(
      hipMemcpy(d_c, bp5->c, n * sizeof(scalar), hipMemcpyHostToDevice));

  // Copy geometric factors and derivative matrix.
  check_driver(hipMalloc((void **)&d_g, 6 * n * sizeof(scalar)));
  check_driver(
      hipMemcpy(d_g, bp5->g, 6 * n * sizeof(scalar), hipMemcpyHostToDevice));

  check_driver(hipMalloc((void **)&d_D, bp5->nx1 * bp5->nx1 * sizeof(scalar)));
  check_driver(hipMemcpy(d_D, bp5->D, bp5->nx1 * bp5->nx1 * sizeof(scalar),
                         hipMemcpyHostToDevice));

  // Copy gather-scatter offsets and indices.
  check_driver(hipMalloc((void **)&d_gs_off, (bp5->gs_n + 1) * sizeof(uint)));
  check_driver(hipMemcpy(d_gs_off, bp5->gs_off, (bp5->gs_n + 1) * sizeof(uint),
                         hipMemcpyHostToDevice));

  check_driver(
      hipMalloc((void **)&d_gs_idx, bp5->gs_off[bp5->gs_n] * sizeof(uint)));
  check_driver(hipMemcpy(d_gs_idx, bp5->gs_idx,
                         bp5->gs_off[bp5->gs_n] * sizeof(uint),
                         hipMemcpyHostToDevice));

  // Work array.
  wrk = bp5_calloc(scalar, n);
  check_driver(hipMalloc((void **)&d_wrk, n * sizeof(scalar)));

  bp5_debug(bp5->verbose, "done.\n");
}

#define unifiedDeviceSynchronize hipDeviceSynchronize
#define unifiedMemcpy hipMemcpy
#define unifiedMemcpyDeviceToHost hipMemcpyDeviceToHost
#include "bp5-backend-unified-cuda-hip.h"
#undef unifiedDeviceSynchronize
#undef unifiedMemcpy
#undef unifiedMemcpyDeviceToHost

static void hip_init(const struct bp5_t *bp5) {
  if (initialized)
    return;
  bp5_debug(bp5->verbose, "hip_init: initializing hip backend ...\n");

  int num_devices = 0;
  check_driver(hipGetDeviceCount(&num_devices));
  if (bp5->device >= (uint)num_devices) {
    bp5_error("hip_init: Invalid device id %d, only %d devices available.",
              bp5->device, num_devices);
  }

  check_driver(hipSetDeviceFlags(hipDeviceMapHost));
  check_driver(hipFree(0));

  hip_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "hip_init: done.\n");
}

static scalar hip_run(const struct bp5_t *bp5, const scalar *r) {
  if (!initialized)
    bp5_error("hip_run: hip backend is not initialized.\n");

  const uint n = bp5_get_local_dofs(bp5);
  bp5_debug(bp5->verbose, "hip_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  check_driver(hipMemcpy(d_r, r, n * sizeof(scalar), hipMemcpyHostToDevice));

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

#if 0
    ax(d_w, d_p, d_g, d_D, bp5->nelt, bp5->nx1);
#else
    copy(d_w, d_p, n);
#endif
    gs(d_w, d_gs_off, d_gs_idx, bp5->gs_n);
    add2s2(d_w, d_p, 0.1, n);
    mask(d_w, n);

    pap = glsc3(d_w, d_c, d_p, n);

    scalar alpha = rtz1 / pap;
    add2s2(d_x, d_p, alpha, n);
    add2s2(d_r, d_w, -alpha, n);

    scalar rtr = glsc3(d_r, d_c, d_r, n);
    rnorm = sqrt(rtr);
    bp5_debug(bp5->verbose, "hip_run: iteration %d, rnorm = %e\n", i, rnorm);
  }
  check_driver(hipDeviceSynchronize());
  clock_t t1 = clock() - t0;

  bp5_debug(bp5->verbose, "hip_run: done.\n");
  bp5_debug(bp5->verbose, "hip_run: iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "hip_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1) / CLOCKS_PER_SEC;
}

static void hip_finalize(void) {
  if (!initialized)
    return;

  check_driver(hipFree(d_r));
  check_driver(hipFree(d_x));
  check_driver(hipFree(d_z));
  check_driver(hipFree(d_p));
  check_driver(hipFree(d_w));
  check_driver(hipFree(d_c));
  check_driver(hipFree(d_g));
  check_driver(hipFree(d_D));
  check_driver(hipFree(d_gs_off));
  check_driver(hipFree(d_gs_idx));
  check_driver(hipFree(d_wrk));
  bp5_free(&wrk);

  initialized = 0;
}

BP5_INTERN void bp5_hip_init(void) {
  bp5_register_backend("HIP", hip_init, hip_run, hip_finalize);
}
