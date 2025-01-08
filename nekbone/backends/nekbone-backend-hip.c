#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "nekbone-backend.h"

static uint initialized = 0;

#define check_error(FNAME, LINE, CALL, ERR_T, SUCCES, GET_ERR, OP)             \
  {                                                                            \
    ERR_T result_ = (CALL);                                                    \
    if (result_ != SUCCES) {                                                   \
      const char *msg = GET_ERR(result_);                                      \
      nekbone_error("%s:%d HIP %s failure: %s.\n", FNAME, LINE, OP, msg);      \
    }                                                                          \
  }

#define check_runtime(call)                                                    \
  check_error(__FILE__, __LINE__, call, hipError_t, hipSuccess,                \
              hipGetErrorName, "runtime")

static scalar *d_r, *d_x, *d_z, *d_p, *d_w;
static scalar *d_wrk, *wrk;
static scalar *d_c, *d_g, *d_D;
static uint   *d_gs_off, *d_gs_idx;

static void hip_mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "hip_mem_init: copy problem data to device ... ");

  const uint n = nekbone_get_local_dofs(nekbone);

  // Allocate device buffers and copy problem data to device.
  check_runtime(hipMalloc((void **)&d_r, n * sizeof(scalar)));
  check_runtime(hipMalloc((void **)&d_x, n * sizeof(scalar)));
  check_runtime(hipMalloc((void **)&d_z, n * sizeof(scalar)));
  check_runtime(hipMalloc((void **)&d_p, n * sizeof(scalar)));
  check_runtime(hipMalloc((void **)&d_w, n * sizeof(scalar)));

  // Copy multiplicity array.
  check_runtime(hipMalloc((void **)&d_c, n * sizeof(scalar)));
  check_runtime(
      hipMemcpy(d_c, nekbone->c, n * sizeof(scalar), hipMemcpyHostToDevice));

  // Copy geometric factors and derivative matrix.
  check_runtime(hipMalloc((void **)&d_g, 6 * n * sizeof(scalar)));
  check_runtime(hipMemcpy(d_g, nekbone->g, 6 * n * sizeof(scalar),
                          hipMemcpyHostToDevice));

  check_runtime(
      hipMalloc((void **)&d_D, nekbone->nx1 * nekbone->nx1 * sizeof(scalar)));
  check_runtime(hipMemcpy(d_D, nekbone->D,
                          nekbone->nx1 * nekbone->nx1 * sizeof(scalar),
                          hipMemcpyHostToDevice));

  // Copy gather-scatter offsets and indices.
  check_runtime(
      hipMalloc((void **)&d_gs_off, (nekbone->gs_n + 1) * sizeof(uint)));
  check_runtime(hipMemcpy(d_gs_off, nekbone->gs_off,
                          (nekbone->gs_n + 1) * sizeof(uint),
                          hipMemcpyHostToDevice));

  check_runtime(hipMalloc((void **)&d_gs_idx,
                          nekbone->gs_off[nekbone->gs_n] * sizeof(uint)));
  check_runtime(hipMemcpy(d_gs_idx, nekbone->gs_idx,
                          nekbone->gs_off[nekbone->gs_n] * sizeof(uint),
                          hipMemcpyHostToDevice));

  // Work array.
  wrk = nekbone_calloc(scalar, n);
  check_runtime(hipMalloc((void **)&d_wrk, n * sizeof(scalar)));

  nekbone_debug(nekbone->verbose, "done.\n");
}

#define unified_device_synchronize    hipDeviceSynchronize
#define unified_memcpy                hipMemcpy
#define unified_memcpy_device_to_host hipMemcpyDeviceToHost

#define unified_rtc_program              hiprtcProgram
#define unified_rtc_create_program       hiprtcCreateProgram
#define unified_rtc_result               hiprtcResult
#define unified_rtc_compile_program      hiprtcCompileProgram
#define unified_rtc_get_program_log_size hiprtcGetProgramLogSize
#define unified_rtc_get_program_log      hiprtcGetProgramLog
#define unified_rtc_get_error_string     hiprtcGetErrorString
#define unified_rtc_get_code_size        hiprtcGetCodeSize
#define unified_rtc_get_code             hiprtcGetCode
#define unified_rtc_destroy_program      hiprtcDestroyProgram

#define unified_module_t             hipModule_t
#define unified_function_t           hipFunction_t
#define unified_module_load_data     hipModuleLoadData
#define unified_module_get_function  hipModuleGetFunction
#define unified_module_unload        hipModuleUnload
#define unified_module_launch_kernel hipModuleLaunchKernel

#define UNIFIED_RTC_SUCCESS HIPRTC_SUCCESS

#define check_rtc(call)                                                        \
  check_error(__FILE__, __LINE__, call, hiprtcResult, HIPRTC_SUCCESS,          \
              hiprtcGetErrorString, "runtime compilation")

#include "nekbone-backend-unified-cuda-hip.h"

#undef unified_device_synchronize
#undef unified_memcpy
#undef unified_memcpy_device_to_host

#undef unified_rtc_program
#undef unified_rtc_create_program
#undef unified_rtc_result
#undef unified_rtc_compile_program
#undef unified_rtc_get_program_log_size
#undef unified_rtc_get_program_log
#undef unified_rtc_get_error_string
#undef unified_rtc_get_code_size
#undef unified_rtc_get_code
#undef unified_rtc_destroy_program

#undef unified_module_t
#undef unified_function_t
#undef unified_module_load_data
#undef unified_module_get_function
#undef unified_module_unload
#undef unified_module_launch_kernel

#undef UNIFIED_RTC_SUCCESS

#undef check_rtc

static void hip_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "hip_init: initializing hip backend ...\n");

  int num_devices = 0;
  check_runtime(hipGetDeviceCount(&num_devices));
  if (nekbone->device >= (uint)num_devices) {
    nekbone_error("hip_init: Invalid device id %d, only %d devices available.",
                  nekbone->device, num_devices);
  }

  check_runtime(hipSetDeviceFlags(hipDeviceMapHost));
  check_runtime(hipFree(0));

  hip_mem_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "hip_init: done.\n");
}

static scalar hip_run(const struct nekbone_t *nekbone, const scalar *r) {
  if (!initialized) nekbone_error("hip_run: hip backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "hip_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  check_runtime(hipMemcpy(d_r, r, n * sizeof(scalar), hipMemcpyHostToDevice));

  scalar pap  = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(d_x, n);

  // Apply Dirichlet BCs to RHS.
  mask(d_r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(d_r, d_c, d_r, n));
  nekbone_debug(nekbone->verbose, "hip_run: iteration 0, rnorm = %e\n", rnorm);
  scalar r0 = rnorm;
  for (uint i = 0; i < nekbone->max_iter; ++i) {
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
    nekbone_debug(nekbone->verbose, "hip_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }
  check_runtime(hipDeviceSynchronize());
  clock_t t1 = clock() - t0;

  nekbone_debug(nekbone->verbose, "hip_run: done.\n");
  nekbone_debug(nekbone->verbose, "hip_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "hip_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1) / CLOCKS_PER_SEC;
}

static void hip_finalize(void) {
  if (!initialized) return;

  check_runtime(hipFree(d_r));
  check_runtime(hipFree(d_x));
  check_runtime(hipFree(d_z));
  check_runtime(hipFree(d_p));
  check_runtime(hipFree(d_w));
  check_runtime(hipFree(d_c));
  check_runtime(hipFree(d_g));
  check_runtime(hipFree(d_D));
  check_runtime(hipFree(d_gs_off));
  check_runtime(hipFree(d_gs_idx));
  check_runtime(hipFree(d_wrk));

  nekbone_free(&wrk);

  initialized = 0;
}

NEKBONE_INTERN void nekbone_hip_init(void) {
  nekbone_register_backend("HIP", hip_init, hip_run, hip_finalize);
}
