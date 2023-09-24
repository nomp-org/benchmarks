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

__global__ static void mask_kernel(scalar *v) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0)
    v[i] = 0;
}

inline static void mask(scalar *d_v, const uint n) {
  const size_t global_size = (n + local_size - 1) / local_size;
  mask_kernel<<<global_size, local_size>>>(d_v);
}
