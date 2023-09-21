#include <string.h>

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "bp5-backend.h"

static uint initialized = 0;

static const char *ERR_STR_OPENCL_FAILURE = "%s failed with error: %d (%s).";

#define CASE(MSG, VAL, STR)                                                    \
  case VAL:                                                                    \
    bp5_error(ERR_STR_OPENCL_FAILURE, MSG, VAL, STR);                          \
    break;

// clang-format off
#define FOR_EACH_ERROR(S)                                                      \
  CASE(S, CL_INVALID_BINARY, "CL_INVALID_BINARY")                              \
  CASE(S, CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE")                    \
  CASE(S, CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE")                \
  CASE(S, CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT")                            \
  CASE(S, CL_INVALID_DEVICE, "CL_INVALID_DEVICE")                              \
  CASE(S, CL_INVALID_EVENT, "CL_INVALID_EVENT")                                \
  CASE(S, CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST")            \
  CASE(S, CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET")                \
  CASE(S, CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE")          \
  CASE(S, CL_INVALID_KERNEL, "CL_INVALID_KERNEL")                              \
  CASE(S, CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS")                    \
  CASE(S, CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION")        \
  CASE(S, CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME")                    \
  CASE(S, CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT")                      \
  CASE(S, CL_INVALID_OPERATION, "CL_INVALID_OPERATION")                        \
  CASE(S, CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM")                            \
  CASE(S, CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE")      \
  CASE(S, CL_INVALID_VALUE, "CL_INVALID_VALUE")                                \
  CASE(S, CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION")              \
  CASE(S, CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE")            \
  CASE(S, CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE")              \
  CASE(S, CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE")\
  CASE(S, CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET")  \
  CASE(S, CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES")                          \
  CASE(S, CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY")
// clang-format on

#define check(call, msg)                                                       \
  {                                                                            \
    cl_int err_ = (call);                                                      \
    if (err_ != CL_SUCCESS) {                                                  \
      switch (err_) {                                                          \
        FOR_EACH_ERROR(msg)                                                    \
      default:                                                                 \
        bp5_error(ERR_STR_OPENCL_FAILURE, msg, err_, "UNKNOWN");               \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

static const char *header_src =
    "#define scalar double                                                 \n"
    "                                                                      \n"
    "#define IDX2(i, j) ((i) + nx1 * (j))                                  \n"
    "#define IDX3(i, j, k) ((i) + nx1 * ((j) + nx1 * (k)))                 \n"
    "                                                                      \n"
    "#define gid get_global_id(0)                                          \n"
    "#define lid get_local_id(0)                                           \n";

static const char *stream_knl_src =
    "__kernel void mask(__global scalar *v) {                              \n"
    "  if (gid == 0)                                                       \n"
    "    v[gid] = 0.0;                                                     \n"
    "}                                                                     \n"
    "__kernel void zero(__global scalar *v, const uint n) {                \n"
    "  if (gid < n)                                                        \n"
    "    v[gid] = 0.0;                                                     \n"
    "}                                                                     \n"
    "__kernel void copy(__global scalar *dst, __global const scalar *src,  \n"
    "                   const uint n) {                                    \n"
    "  if (gid < n)                                                        \n"
    "    dst[gid] = src[gid];                                              \n"
    "}                                                                     \n"
    "__kernel void add2s1(__global scalar *a, __global const scalar *b,    \n"
    "                     const scalar c, const uint n) {                  \n"
    "  if (gid < n)                                                        \n"
    "    a[gid] = c * a[gid] + b[gid];                                     \n"
    "}                                                                     \n"
    "__kernel void add2s2(__global scalar *a, __global const scalar *b,    \n"
    "                     const scalar c, const uint n) {                  \n"
    "  if (gid < n)                                                        \n"
    "    a[gid] += c * b[gid];                                             \n"
    "}                                                                     \n"
    "__kernel void glsc3(__global scalar *out,                             \n"
    "                    __global const scalar *a,                         \n"
    "                    __global const scalar *b,                         \n"
    "                    __global const scalar *c,                         \n"
    "                    __local scalar *s_abc,                            \n"
    "                    const uint n) {                                   \n"
    "  if (gid < n)                                                        \n"
    "    s_abc[lid] = a[gid] * b[gid] * c[gid];                            \n"
    "  else                                                                \n"
    "   s_abc[lid] = 0.0;                                                  \n"
    "                                                                      \n"
    "  for (uint s = get_local_size(0) / 2; s > 0; s >>= 1) {              \n"
    "    barrier(CLK_LOCAL_MEM_FENCE);                                     \n"
    "    if (lid < s)                                                      \n"
    "      s_abc[lid] += s_abc[lid + s];                                   \n"
    "  }                                                                   \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  if (lid == 0)                                                       \n"
    "    out[get_group_id(0)] = s_abc[0];                                  \n"
    "}                                                                     \n"
    "__kernel void gs(__global scalar *v, __global const uint *gs_off,     \n"
    "                 __global const uint *gs_idx, const uint gs_n) {      \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < gs_n) {                                                     \n"
    "    scalar s = 0.0;                                                   \n"
    "    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++)                  \n"
    "      s += v[gs_idx[j]];                                              \n"
    "    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++)                  \n"
    "      v[gs_idx[j]] = s;                                               \n"
    "  }                                                                   \n"
    "}                                                                     \n";

static const char *ax_knl_src =
    "__kernel void ax_kernel_v00(__global scalar *w,                       \n"
    "                            __global const scalar *u,                 \n"
    "                            __global const scalar *G,                 \n"
    "                            __global const scalar *D,                 \n"
    "                            const uint nx1,                           \n"
    "                            __local scalar *smem) {                   \n"
    "  const uint ebase = get_group_id(0) * nx1 * nx1 * nx1;               \n"
    "  const uint i = get_local_id(0);                                     \n"
    "  const uint j = get_local_id(1);                                     \n"
    "  const uint k = get_local_id(2);                                     \n"
    "                                                                      \n"
    "  __local scalar *s_D = smem;                                         \n"
    "  __local scalar *s_ur = (__local scalar *)&s_D[nx1 * nx1 * nx1];     \n"
    "  __local scalar *s_us = (__local scalar *)&s_ur[nx1 * nx1 * nx1];    \n"
    "  __local scalar *s_ut = (__local scalar *)&s_us[nx1 * nx1 * nx1];    \n"
    "                                                                      \n"
    "  s_ur[IDX3(i, j, k)] = 0;                                            \n"
    "  s_us[IDX3(i, j, k)] = 0;                                            \n"
    "  s_ut[IDX3(i, j, k)] = 0;                                            \n"
    "  if (k == 0)                                                         \n"
    "    s_D[IDX2(i, j)] = D[IDX2(i, j)];                                  \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  for (uint l = 0; l < nx1; l++) {                                    \n"
    "    s_ur[IDX3(i, j, k)] += s_D[IDX2(l, i)] * u[ebase + IDX3(l, j, k)];\n"
    "    s_us[IDX3(i, j, k)] += s_D[IDX2(l, j)] * u[ebase + IDX3(i, l, k)];\n"
    "    s_ut[IDX3(i, j, k)] += s_D[IDX2(l, k)] * u[ebase + IDX3(i, j, l)];\n"
    "  }                                                                   \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  const uint gbase = 6 * (ebase + IDX3(i, j, k));                     \n"
    "  scalar r_G00 = G[gbase + 0];                                        \n"
    "  scalar r_G01 = G[gbase + 1];                                        \n"
    "  scalar r_G02 = G[gbase + 2];                                        \n"
    "  scalar r_G11 = G[gbase + 3];                                        \n"
    "  scalar r_G12 = G[gbase + 4];                                        \n"
    "  scalar r_G22 = G[gbase + 5];                                        \n"
    "                                                                      \n"
    "  scalar wr = r_G00 * s_ur[IDX3(i, j, k)] +                           \n"
    "              r_G01 * s_us[IDX3(i, j, k)] +                           \n"
    "              r_G02 * s_ut[IDX3(i, j, k)];                            \n"
    "  scalar ws = r_G01 * s_ur[IDX3(i, j, k)] +                           \n"
    "              r_G11 * s_us[IDX3(i, j, k)] +                           \n"
    "              r_G12 * s_ut[IDX3(i, j, k)];                            \n"
    "  scalar wt = r_G02 * s_ur[IDX3(i, j, k)] +                           \n"
    "              r_G12 * s_us[IDX3(i, j, k)] +                           \n"
    "              r_G22 * s_ut[IDX3(i, j, k)];                            \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  s_ur[IDX3(i, j, k)] = wr;                                           \n"
    "  s_us[IDX3(i, j, k)] = ws;                                           \n"
    "  s_ut[IDX3(i, j, k)] = wt;                                           \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  scalar wo = 0;                                                      \n"
    "  for (uint l = 0; l < nx1; l++) {                                    \n"
    "    wo += s_D[IDX2(i, l)] * s_ur[IDX3(l, j, k)] +                     \n"
    "          s_D[IDX2(j, l)] * s_us[IDX3(i, l, k)] +                     \n"
    "          s_D[IDX2(k, l)] * s_ut[IDX3(i, j, l)];                      \n"
    "  }                                                                   \n"
    "  w[ebase + IDX3(i, j, k)] = wo;                                      \n"
    "}                                                                     \n";

// OpenCL device, context, queue and program.
static cl_device_id ocl_device;
static cl_command_queue ocl_queue;
static cl_context ocl_ctx;

static void opencl_device_init(const struct bp5_t *bp5) {
  // Setup OpenCL platform.
  bp5_debug(bp5->verbose, "opencl_init: initialize platform ...\n");
  cl_uint num_platforms = 0;
  check(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs");
  if (bp5->platform_id < 0 | bp5->platform_id >= num_platforms)
    bp5_error("opencl_init: platform id is invalid: %d", bp5->platform_id);

  cl_platform_id *cl_platforms = bp5_calloc(cl_platform_id, num_platforms);
  check(clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms),
        "clGetPlatformIDs");
  cl_platform_id platform = cl_platforms[bp5->platform_id];
  bp5_free(&cl_platforms);

  // Setup OpenCL device.
  bp5_debug(bp5->verbose, "opencl_init: initialize device ...\n");
  cl_uint num_devices = 0;
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices),
        "clGetDeviceIDs");
  if (bp5->device_id >= num_devices)
    bp5_error("opencl_init: device id is invalid: %d", bp5->device_id);

  cl_device_id *cl_devices = bp5_calloc(cl_device_id, num_devices);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices),
        "clGetDeviceIDs");
  ocl_device = cl_devices[bp5->device_id];
  bp5_free(&cl_devices);

  // Setup OpenCL context and queue.
  bp5_debug(bp5->verbose, "opencl_init: initialize context and queue ...\n");
  cl_int err;
  ocl_ctx = clCreateContext(NULL, 1, &ocl_device, NULL, NULL, &err);
  check(err, "clCreateContext");
  ocl_queue = clCreateCommandQueueWithProperties(ocl_ctx, ocl_device, 0, &err);
  check(err, "clCreateCommandQueueWithProperties");

  bp5_debug(bp5->verbose, "opencl_init: done.\n");
}

// OpenCL device buffers.
static cl_mem r_mem, x_mem, z_mem, p_mem, w_mem;
static cl_mem c_mem, g_mem, D_mem;
static cl_mem gs_off_mem, gs_idx_mem;
static cl_mem wrk_mem;
static scalar *wrk;

static void opencl_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "opencl_mem_init: copy problem data to device ...\n");

  const uint n = bp5_get_local_dofs(bp5);

  // Allocate device buffers and copy problem data to device.
  cl_int err;
  r_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                         &err);
  check(err, "clCreateBuffer(r)");
  x_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                         &err);
  check(err, "clCreateBuffer(x)");
  z_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                         &err);
  check(err, "clCreateBuffer(z)");
  p_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                         &err);
  check(err, "clCreateBuffer(p)");
  w_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                         &err);
  check(err, "clCreateBuffer(w)");

  // Copy multiplicity array.
  c_mem =
      clCreateBuffer(ocl_ctx, CL_MEM_READ_ONLY, n * sizeof(scalar), NULL, &err);
  check(err, "clCreateBuffer(c)");
  check(clEnqueueWriteBuffer(ocl_queue, c_mem, CL_TRUE, 0, n * sizeof(scalar),
                             bp5->c, 0, NULL, NULL),
        "clEnqueueWriteBuffer(c)");

  // Copy geometric factors and derivative matrix.
  g_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_ONLY, 6 * n * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(g)");
  check(clEnqueueWriteBuffer(ocl_queue, g_mem, CL_TRUE, 0,
                             6 * n * sizeof(scalar), bp5->g, 0, NULL, NULL),
        "clEnqueueWriteBuffer(g)");

  D_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_ONLY,
                         bp5->nx1 * bp5->nx1 * sizeof(scalar), NULL, &err);
  check(err, "clCreateBuffer(D)");
  check(clEnqueueWriteBuffer(ocl_queue, D_mem, CL_TRUE, 0,
                             bp5->nx1 * bp5->nx1 * sizeof(scalar), bp5->D, 0,
                             NULL, NULL),
        "clEnqueueWriteBuffer(D)");

  // Copy gather-scatter offsets and indices.
  gs_off_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_ONLY,
                              (bp5->gs_n + 1) * sizeof(uint), NULL, &err);
  check(err, "clCreateBuffer(gs_off)");
  check(clEnqueueWriteBuffer(ocl_queue, gs_off_mem, CL_TRUE, 0,
                             (bp5->gs_n + 1) * sizeof(uint), bp5->gs_off, 0,
                             NULL, NULL),
        "clEnqueueWriteBuffer(gs_off)");

  // We add +1 to the actual buffer size in order to avoid
  // CL_INVALID_BUFFER_SIZE in case of there are zero gather-scatter dofs (for
  // example a single element with order = 1).
  gs_idx_mem =
      clCreateBuffer(ocl_ctx, CL_MEM_READ_ONLY,
                     (bp5->gs_off[bp5->gs_n] + 1) * sizeof(uint), NULL, &err);
  check(err, "clCreateBuffer(gs_idx)");
  check(clEnqueueWriteBuffer(ocl_queue, gs_idx_mem, CL_TRUE, 0,
                             bp5->gs_off[bp5->gs_n] * sizeof(uint), bp5->gs_idx,
                             0, NULL, NULL),
        "clEnqueueWriteBuffer(gs_idx)");

  // Work array.
  wrk = bp5_calloc(scalar, n);
  wrk_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, n * sizeof(scalar), NULL,
                           &err);
  check(err, "clCreateBuffer(wrk)");

  bp5_debug(bp5->verbose, "opencl_mem_init: done.\n");
}

// OpenCL kernels.
static cl_program ocl_program;
static cl_kernel mask_kernel, zero_kernel, copy_kernel;
static cl_kernel glsc3_kernel, add2s1_kernel, add2s2_kernel;
static cl_kernel ax_kernel, gs_kernel;
static const size_t local_size = 512;

static void print_program_build_log() {}

static void opencl_kernels_init(const uint verbose) {
  // Build OpenCL kernels.
  bp5_debug(verbose, "opencl_kernels_init: compile kernels ...\n");

  size_t size =
      strlen(header_src) + strlen(stream_knl_src) + strlen(ax_knl_src);
  char *knl_src = bp5_calloc(char, size + 1);
  strcpy(knl_src, header_src);
  strcpy(knl_src + strlen(header_src), stream_knl_src);
  strcpy(knl_src + strlen(header_src) + strlen(stream_knl_src), ax_knl_src);

  cl_int err;
  ocl_program = clCreateProgramWithSource(ocl_ctx, 1, (const char **)&knl_src,
                                          NULL, &err);
  check(err, "clCreateProgramWithSource");
  bp5_free(&knl_src);

  err = clBuildProgram(ocl_program, 1, &ocl_device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(ocl_program, ocl_device, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);
    char *log = bp5_calloc(char, log_size);
    clGetProgramBuildInfo(ocl_program, ocl_device, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    bp5_debug(verbose,
              "opencl_kernels_init: clBuildProgram failed with error:\n %s.\n",
              log);
    bp5_free(&log);
    bp5_error("clBuildProgram failed.");
  }
  bp5_debug(verbose, "opencl_kernels_init: done.\n");

  bp5_debug(verbose, "opencl_kernels_init: create kernels ...\n");
  mask_kernel = clCreateKernel(ocl_program, "mask", &err);
  check(err, "clCreateKernel(mask)");
  zero_kernel = clCreateKernel(ocl_program, "zero", &err);
  check(err, "clCreateKernel(zero)");
  copy_kernel = clCreateKernel(ocl_program, "copy", &err);
  check(err, "clCreateKernel(copy)");
  glsc3_kernel = clCreateKernel(ocl_program, "glsc3", &err);
  check(err, "clCreateKernel(glsc3)");
  add2s1_kernel = clCreateKernel(ocl_program, "add2s1", &err);
  check(err, "clCreateKernel(add2s1)");
  add2s2_kernel = clCreateKernel(ocl_program, "add2s2", &err);
  check(err, "clCreateKernel(add2s2)");
  ax_kernel = clCreateKernel(ocl_program, "ax_kernel_v00", &err);
  check(err, "clCreateKernel(ax)");
  gs_kernel = clCreateKernel(ocl_program, "gs", &err);
  check(err, "clCreateKernel(gs)");

  bp5_debug(verbose, "opencl_kernels_init: done.\n");
}

static void mask(cl_mem *mem, const uint n) {
  check(clSetKernelArg(mask_kernel, 0, sizeof(cl_mem), mem),
        "clSetKernelArg(mask, 0)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, mask_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(mask)");
}

static void zero(cl_mem *mem, const uint n) {
  check(clSetKernelArg(zero_kernel, 0, sizeof(cl_mem), mem),
        "clSetKernelArg(zero, 0)");
  check(clSetKernelArg(zero_kernel, 1, sizeof(uint), &n),
        "clSetKernelArg(zero, 1)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, zero_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(zero)");
}

static scalar glsc3(cl_mem *a, cl_mem *b, cl_mem *c, const uint n) {
  check(clSetKernelArg(glsc3_kernel, 0, sizeof(cl_mem), &wrk_mem),
        "clSetKernelArg(glsc3, 0)");
  check(clSetKernelArg(glsc3_kernel, 1, sizeof(cl_mem), a),
        "clSetKernelArg(glsc3, 1)");
  check(clSetKernelArg(glsc3_kernel, 2, sizeof(cl_mem), b),
        "clSetKernelArg(glsc3, 2)");
  check(clSetKernelArg(glsc3_kernel, 3, sizeof(cl_mem), c),
        "clSetKernelArg(glsc3, 3)");
  check(clSetKernelArg(glsc3_kernel, 4, sizeof(scalar) * local_size, NULL),
        "clSetKernelArg(glsc3, 4)");
  check(clSetKernelArg(glsc3_kernel, 5, sizeof(uint), &n),
        "clSetKernelArg(glsc3, 5)");

  const size_t block_size = (n + local_size - 1) / local_size;
  const size_t global_size = block_size * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, glsc3_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(glsc3)");
  check(clFinish(ocl_queue), "clFinish(glsc3)");

  check(clEnqueueReadBuffer(ocl_queue, wrk_mem, CL_TRUE, 0,
                            sizeof(scalar) * block_size, wrk, 0, NULL, NULL),
        "clEnqueueReadBuffer(glsc3, wrk)");

  for (uint i = 1; i < block_size; i++)
    wrk[0] += wrk[i];

  return wrk[0];
}

static void copy(cl_mem *a, cl_mem *b, const uint n) {
  check(clSetKernelArg(copy_kernel, 0, sizeof(cl_mem), a),
        "clSetKernelArg(copy, 0)");
  check(clSetKernelArg(copy_kernel, 1, sizeof(cl_mem), b),
        "clSetKernelArg(copy, 1)");
  check(clSetKernelArg(copy_kernel, 2, sizeof(uint), &n),
        "clSetKernelArg(copy, 2)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, copy_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(copy)");
}

static void add2s1(cl_mem *a, cl_mem *b, const scalar c, const uint n) {
  check(clSetKernelArg(add2s1_kernel, 0, sizeof(cl_mem), a),
        "clSetKernelArg(add2s1, 0)");
  check(clSetKernelArg(add2s1_kernel, 1, sizeof(cl_mem), b),
        "clSetKernelArg(add2s1, 1)");
  check(clSetKernelArg(add2s1_kernel, 2, sizeof(scalar), &c),
        "clSetKernelArg(add2s1, 2)");
  check(clSetKernelArg(add2s1_kernel, 3, sizeof(uint), &n),
        "clSetKernelArg(add2s1, 3)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, add2s1_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(add2s1)");
}

static void add2s2(cl_mem *a, cl_mem *b, const scalar c, const uint n) {
  check(clSetKernelArg(add2s2_kernel, 0, sizeof(cl_mem), a),
        "clSetKernelArg(add2s2, 0)");
  check(clSetKernelArg(add2s2_kernel, 1, sizeof(cl_mem), b),
        "clSetKernelArg(add2s2, 1)");
  check(clSetKernelArg(add2s2_kernel, 2, sizeof(scalar), &c),
        "clSetKernelArg(add2s2, 2)");
  check(clSetKernelArg(add2s2_kernel, 3, sizeof(uint), &n),
        "clSetKernelArg(add2s2, 3)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, add2s2_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(add2s2)");
}

static void ax(cl_mem *w, cl_mem *p, cl_mem *g, cl_mem *D, const uint nelt,
               const uint nx1) {
  check(clSetKernelArg(ax_kernel, 0, sizeof(cl_mem), w),
        "clSetKernelArg(ax, 0)");
  check(clSetKernelArg(ax_kernel, 1, sizeof(cl_mem), p),
        "clSetKernelArg(ax, 1)");
  check(clSetKernelArg(ax_kernel, 2, sizeof(cl_mem), g),
        "clSetKernelArg(ax, 2)");
  check(clSetKernelArg(ax_kernel, 3, sizeof(cl_mem), D),
        "clSetKernelArg(ax, 3)");
  check(clSetKernelArg(ax_kernel, 4, sizeof(uint), &nx1),
        "clSetKernelArg(ax, 5)");
  const size_t shared_size = (3 * nx1 * nx1 * nx1 + nx1 * nx1) * sizeof(scalar);
  check(clSetKernelArg(ax_kernel, 5, shared_size, NULL),
        "clSetKernelArg(ax, 6)");

  const size_t local_work[3] = {nx1, nx1, nx1};
  const size_t global_work[3] = {nelt * nx1, nx1, nx1};
  check(clEnqueueNDRangeKernel(ocl_queue, ax_kernel, 3, NULL, global_work,
                               local_work, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(ax)");
}

static void gs(cl_mem *x, const cl_mem *gs_off, const cl_mem *gs_idx,
               const uint gs_n) {
  check(clSetKernelArg(gs_kernel, 0, sizeof(cl_mem), x),
        "clSetKernelArg(gs, 0)");
  check(clSetKernelArg(gs_kernel, 1, sizeof(cl_mem), gs_off),
        "clSetKernelArg(gs, 1)");
  check(clSetKernelArg(gs_kernel, 2, sizeof(cl_mem), gs_idx),
        "clSetKernelArg(gs, 2)");
  check(clSetKernelArg(gs_kernel, 3, sizeof(uint), &gs_n),
        "clSetKernelArg(gs, 3)");

  const size_t global_size =
      ((gs_n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, gs_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(gs)");
}

static void opencl_init(const struct bp5_t *bp5) {
  if (initialized)
    return;
  bp5_debug(bp5->verbose, "opencl_init: initializing OpenCL backend ...\n");

  opencl_device_init(bp5);
  opencl_kernels_init(bp5->verbose);
  opencl_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "opencl_init: done.\n");
}

static scalar opencl_run(const struct bp5_t *bp5, const scalar *r) {
  if (!initialized)
    bp5_error("opencl_run: OpenCL backend is not initialized.\n");

  const uint n = bp5_get_local_dofs(bp5);
  bp5_debug(bp5->verbose, "opencl_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  check(clEnqueueWriteBuffer(ocl_queue, r_mem, CL_TRUE, 0, n * sizeof(scalar),
                             r, 0, NULL, NULL),
        "clEnqueueWriteBuffer(r)");

  scalar pap = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(&x_mem, n);

  // Apply Dirichlet BCs to RHS.
  mask(&r_mem, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(&r_mem, &c_mem, &r_mem, n)), r0 = rnorm;
  for (uint i = 0; i < bp5->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(&z_mem, &r_mem, n);

    rtz2 = rtz1;
    rtz1 = glsc3(&r_mem, &c_mem, &z_mem, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(&p_mem, &z_mem, beta, n);

    ax(&w_mem, &p_mem, &g_mem, &D_mem, bp5->nelt, bp5->nx1);
    gs(&w_mem, &gs_off_mem, &gs_idx_mem, bp5->gs_n);
    add2s2(&w_mem, &p_mem, 0.1, n);
    mask(&w_mem, n);

    pap = glsc3(&w_mem, &c_mem, &p_mem, n);

    scalar alpha = rtz1 / pap;
    add2s2(&x_mem, &p_mem, alpha, n);
    add2s2(&r_mem, &w_mem, -alpha, n);

    scalar rtr = glsc3(&r_mem, &c_mem, &r_mem, n);
    rnorm = sqrt(rtr);
    bp5_debug(bp5->verbose, "opencl_run: iteration %d, rnorm = %e\n", i, rnorm);
  }
  check(clFinish(ocl_queue), "clFinish(cg)");
  clock_t t1 = clock();

  bp5_debug(bp5->verbose, "opencl_run: done.\n");
  bp5_debug(bp5->verbose, "opencl_run: iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "opencl_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void opencl_finalize(void) {
  if (!initialized)
    return;

  check(clReleaseProgram(ocl_program), "clReleaseProgram");
  check(clReleaseKernel(mask_kernel), "clReleaseKernel(mask)");
  check(clReleaseKernel(zero_kernel), "clReleaseKernel(zero)");
  check(clReleaseKernel(copy_kernel), "clReleaseKernel(copy)");
  check(clReleaseKernel(glsc3_kernel), "clReleaseKernel(glsc3)");
  check(clReleaseKernel(add2s1_kernel), "clReleaseKernel(add2s1)");
  check(clReleaseKernel(add2s2_kernel), "clReleaseKernel(add2s2)");
  check(clReleaseKernel(ax_kernel), "clReleaseKernel(ax)");
  check(clReleaseKernel(gs_kernel), "clReleaseKernel(gs)");
  check(clReleaseMemObject(x_mem), "clReleaseMemObject(x)");
  check(clReleaseMemObject(z_mem), "clReleaseMemObject(z)");
  check(clReleaseMemObject(p_mem), "clReleaseMemObject(p)");
  check(clReleaseMemObject(w_mem), "clReleaseMemObject(w)");
  check(clReleaseMemObject(r_mem), "clReleaseMemObject(r)");
  check(clReleaseMemObject(c_mem), "clReleaseMemObject(c)");
  check(clReleaseMemObject(g_mem), "clReleaseMemObject(g)");
  check(clReleaseMemObject(D_mem), "clReleaseMemObject(D)");
  check(clReleaseMemObject(gs_off_mem), "clReleaseMemObject(gs_off)");
  check(clReleaseMemObject(gs_idx_mem), "clReleaseMemObject(gs_idx)");
  check(clReleaseMemObject(wrk_mem), "clReleaseMemObject(wrk)");
  bp5_free(&wrk);
  check(clReleaseCommandQueue(ocl_queue), "clReleaseCommandQueue");
  check(clReleaseContext(ocl_ctx), "clReleaseContext");

  initialized = 0;
}

void bp5_opencl_init(void) {
  bp5_register_backend("OPENCL", opencl_init, opencl_run, opencl_finalize);
}

#undef check
#undef FOR_EACH_ERROR
#undef CASE
