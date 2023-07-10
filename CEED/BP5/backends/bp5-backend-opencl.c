#include "bp5-backend.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

static uint initialized = 0;

static const char *ERR_STR_OPENCL_FAILURE = "%s failed with error code: %d.";

#define check(call, msg)                                                       \
  {                                                                            \
    cl_int err_ = (call);                                                      \
    if (err_ != CL_SUCCESS)                                                    \
      bp5_error(ERR_STR_OPENCL_FAILURE, msg, err_);                            \
  }

static const char *knl_src =
    "#define scalar double                                                 \n"
    "#define uint uint32_t                                                 \n"
    "                                                                      \n"
    "__kernel void mask(__global scalar *v) {                              \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i == 0)                                                         \n"
    "    v[i] = 0.0;                                                       \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void zero(__global scalar *v, const uint n) {                \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n)                                                          \n"
    "    v[i] = 0.0;                                                       \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void copy(__global scalar *dst, __global const scalar *src,  \n"
    "                   const uint n) {                                    \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n)                                                          \n"
    "    dst[i] = src[i];                                                  \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void add2s1(__global scalar *a, __global const scalar *b,    \n"
    "                     const scalar c, const uint n) {                  \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n)                                                          \n"
    "    a[i] += c * a[i] + b[i];                                          \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void add2s2(__global scalar *a, __global const scalar *b,    \n"
    "                     const scalar c, const uint n) {                  \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n)                                                          \n"
    "    a[i] += c * b[i];                                                 \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void glsc3(__global scalar *out,                             \n"
    "                    __global const scalar *a,                         \n"
    "                    __global const scalar *b,                         \n"
    "                    __global cosnt scalar *c,                         \n"
    "                    __local scalar *s_abc,                            \n"
    "                    const uint n) {                                   \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n)                                                          \n"
    "    s_abc[i] = a[i] * b[i] * c[i];                                    \n"
    "  else                                                                \n"
    "   s_abc[i] = 0.0;                                                    \n"
    "                                                                      \n"
    "  for (uint s = get_local_size(0) / 2; s > 0; s >>= 1) {              \n"
    "    barrier(CLK_LOCAL_MEM_FENCE);                                     \n"
    "    if (get_local_id(0) < s)                                          \n"
    "      s_abc[i] += s_abc[i + s];                                       \n"
    "  }                                                                   \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  if (get_local_id(0) == 0)                                           \n"
    "    out[get_group_id(0)] = s_abc[0];                                  \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void gs(__global scalar *v, __global const uint *gs_off,     \n"
    "                 __global const uint *gs_idx, const uint n) {         \n"
    "  int i = get_global_id(0);                                           \n"
    "  if (i < n) {                                                        \n"
    "    scalar s = 0.0;                                                   \n"
    "    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j)                  \n"
    "      s += v[gs_idx[j]];                                              \n"
    "    for (uint j = gs_off[i]; j < gs_off[i + 1]; ++j)                  \n"
    "      v[gs_idx[j]] = s;                                               \n"
    "  }                                                                   \n"
    "}                                                                     \n"
    "                                                                      \n"
    "__kernel void ax_v00(__global scalar *w, __global const scalar *u,    \n"
    "                     __global const scalar *g,                        \n"
    "                     __global const scalar *D, const uint nelt,       \n"
    "                     const uint nx1, const uint ngeo,                 \n"
    "                     __local scalar *smem)) {                         \n"
    "  const uint ebase = get_group_id(0) * nx1 * nx1 * nx1;               \n"
    "  const uint i = get_local_id(0);                                     \n"
    "  const uint j = get_local_id(1);                                     \n"
    "  const uint k = get_local_id(2);                                     \n"
    "                                                                      \n"
    "  scalar *s_D = smem;                                                 \n"
    "  scalar *s_u = (scalar *)&s_D[nx1 * nx1];                            \n"
    "  scalar *s_ur = (scalar *)&s_u[nx1 * nx1 * nx1];                     \n"
    "  scalar *s_us = (scalar *)&s_ur[nx1 * nx1 * nx1];                    \n"
    "  scalar *s_ut = (scalar *)&s_us[nx1 * nx1 * nx1];                    \n"
    "                                                                      \n"
    "  s_u[IDX3(i, j, k)] = u[ebase + IDX3(i, j, k)];                      \n"
    "  s_ur[IDX3(i, j, k)] = 0;                                            \n"
    "  s_us[IDX3(i, j, k)] = 0;                                            \n"
    "  s_ut[IDX3(i, j, k)] = 0;                                            \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  for (uint l = 0; l < nx1; l++) {                                    \n"
    "    s_ur[IDX3(i, j, k)] += s_D[IDX2(i, l) * s_u[IDX3(k, j, l)];       \n"
    "    s_us[IDX3(i, j, k)] += s_D[IDX2(j, l) * s_u[IDX3(k, l, i)];       \n"
    "    s_ut[IDX3(i, j, k)] += s_D[IDX2(k, l) * s_u[IDX3(l, j, i)];       \n"
    "  }                                                                   \n"
    "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
    "                                                                      \n"
    "  const uint gbase = ngeo * (ebase + IDX3(i, j, k));                  \n"
    "  scalar r_G00 = g[gbase + 0];                                        \n"
    "  scalar r_G01 = g[gbase + 1];                                        \n"
    "  scalar r_G02 = g[gbase + 2];                                        \n"
    "  scalar r_G11 = g[gbase + 3];                                        \n"
    "  scalar r_G12 = g[gbase + 4];                                        \n"
    "  scalar r_G22 = g[gbase + 5];                                        \n"
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
    "    wo += s_D[IDX2(l, i)] * s_ur[IDX3(l, j, k)] +                     \n"
    "          s_D[IDX2(l, j)] * s_us[IDX3(i, l, k)] +                     \n"
    "          s_D[IDX2(l, k)] * s_ut[IDX3(i, j, l)];                      \n"
    "  }                                                                   \n"
    "  w[ebase + IDX3(i, j, k)] = wo;                                      \n"
    "}                                                                     \n";

// OpenCL device, context, queue and program.
static cl_device_id ocl_device_id;
static cl_command_queue ocl_queue;
static cl_context ocl_ctx;

static void opencl_device_init(const struct bp5_t *bp5) {
  // Setup OpenCL platform.
  bp5_debug(bp5->verbose, "opencl_init: Initialize platform ...\n");
  cl_uint num_platforms = 0;
  check(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs");
  if (bp5->platform_id < 0 | bp5->platform_id >= num_platforms)
    bp5_error("opencl_init: Platform ID is invalid: %d", bp5->platform_id);

  cl_platform_id *cl_platforms = bp5_calloc(cl_platform_id, num_platforms);
  check(clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms),
        "clGetPlatformIDs");
  cl_platform_id platform = cl_platforms[bp5->platform_id];
  bp5_free(&cl_platforms);
  bp5_debug(bp5->verbose, "opencl_init: done.\n");

  // Setup OpenCL device.
  bp5_debug(bp5->verbose, "opencl_init: Initialize device ...\n");
  cl_uint num_devices = 0;
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices),
        "clGetDeviceIDs");
  if (bp5->device_id >= num_devices)
    bp5_error("opencl_init: Device ID is invalid: %d", bp5->device_id);

  cl_device_id *cl_devices = bp5_calloc(cl_device_id, num_devices);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices),
        "clGetDeviceIDs");
  ocl_device_id = cl_devices[bp5->device_id];
  bp5_free(&cl_devices);
  bp5_debug(bp5->verbose, "opencl_init: done.\n");

  // Setup OpenCL context and queue.
  cl_int err;
  bp5_debug(bp5->verbose, "opencl_init: Initialize context and queue ...\n");
  ocl_ctx = clCreateContext(NULL, 1, &ocl_device_id, NULL, NULL, &err);
  check(err, "clCreateContext");
  ocl_queue =
      clCreateCommandQueueWithProperties(ocl_ctx, ocl_device_id, 0, &err);
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
  bp5_debug(bp5->verbose, "opencl_mem_init: Copy problem data to device ...\n");

  // Allocate device buffers and copy problem data to device.
  ulong dofs = bp5_get_local_dofs(bp5);
  cl_int err;
  r_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(r)");
  x_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(x)");
  z_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(z)");
  p_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(p)");
  w_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(w)");

  // Copy multiplicity array.
  c_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                         NULL, &err);
  check(clEnqueueWriteBuffer(ocl_queue, c_mem, CL_TRUE, 0,
                             dofs * sizeof(scalar), bp5->c, 0, NULL, NULL),
        "clEnqueueWriteBuffer(c)");

  // Copy geometric factors and derivative matrix.
  g_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, 6 * dofs * sizeof(scalar),
                         NULL, &err);
  check(err, "clCreateBuffer(g)");
  check(clEnqueueWriteBuffer(ocl_queue, g_mem, CL_TRUE, 0,
                             6 * dofs * sizeof(scalar), bp5->g, 0, NULL, NULL),
        "clEnqueueWriteBuffer(g)");

  D_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE,
                         bp5->nx1 * bp5->nx1 * sizeof(scalar), NULL, &err);
  check(err, "clCreateBuffer(D)");
  check(clEnqueueWriteBuffer(ocl_queue, D_mem, CL_TRUE, 0,
                             bp5->nx1 * bp5->nx1 * sizeof(scalar), bp5->D, 0,
                             NULL, NULL),
        "clEnqueueWriteBuffer(D)");

  // Copy gather-scatter offsets and indices.
  gs_off_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE,
                              (bp5->gs_n + 1) * sizeof(uint), NULL, &err);
  check(err, "clCreateBuffer(gs_off)");
  check(clEnqueueWriteBuffer(ocl_queue, gs_off_mem, CL_TRUE, 0,
                             (bp5->gs_n + 1) * sizeof(uint), bp5->gs_off, 0,
                             NULL, NULL),
        "clEnqueueWriteBuffer(gs_off)");

  gs_idx_mem =
      clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE,
                     bp5->gs_off[bp5->gs_n] * sizeof(uint), NULL, &err);
  check(err, "clCreateBuffer(gs_idx)");
  check(clEnqueueWriteBuffer(ocl_queue, gs_idx_mem, CL_TRUE, 0,
                             bp5->gs_off[bp5->gs_n] * sizeof(uint), bp5->gs_idx,
                             0, NULL, NULL),
        "clEnqueueWriteBuffer(gs_idx)");

  // Work array.
  wrk = bp5_calloc(scalar, dofs);
  wrk_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, dofs * sizeof(scalar),
                           NULL, &err);
  check(err, "clCreateBuffer(wrk)");

  bp5_debug(bp5->verbose, "opencl_mem_init: done.\n");
}

// OpenCL kernels.
static cl_program ocl_program;
static cl_kernel mask_kernel, zero_kernel, copy_kernel;
static cl_kernel glsc3_kernel, add2s1_kernel, add2s2_kernel;
static cl_kernel ax_kernel, gs_kernel;
static const size_t local_size = 512;

static void opencl_kernels_init(const uint verbose) {
  // Build OpenCL kernels.
  bp5_debug(verbose, "opencl_kernels_init: Build kernels ...\n");
  cl_int err;
  ocl_program = clCreateProgramWithSource(ocl_ctx, 1, (const char **)&knl_src,
                                          NULL, &err);
  check(err, "clCreateProgramWithSource");
  err = clBuildProgram(ocl_program, 1, &ocl_device_id, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(ocl_program, ocl_device_id, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);
    char *log = bp5_calloc(char, log_size);
    clGetProgramBuildInfo(ocl_program, ocl_device_id, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    bp5_debug(verbose, "clBuildProgram failed with error:\n %s.\n", log);
    bp5_free(&log);
    bp5_error("clBuildProgram failed.");
  }
  bp5_debug(verbose, "opencl_kernels_init: done.\n");

  bp5_debug(verbose, "opencl_kernels_init: Create kernels ...");
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
  ax_kernel = clCreateKernel(ocl_program, "ax", &err);
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
  check(clSetKernelArg(glsc3_kernel, 0, sizeof(cl_mem), a),
        "clSetKernelArg(glsc3, 0)");
  check(clSetKernelArg(glsc3_kernel, 1, sizeof(cl_mem), b),
        "clSetKernelArg(glsc3, 1)");
  check(clSetKernelArg(glsc3_kernel, 2, sizeof(cl_mem), c),
        "clSetKernelArg(glsc3, 2)");
  check(clSetKernelArg(glsc3_kernel, 3, sizeof(uint), &n),
        "clSetKernelArg(glsc3, 3)");

  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
  check(clEnqueueNDRangeKernel(ocl_queue, glsc3_kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(glsc3)");

  // FIXME: Do the host side reduction.
  return 0;
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
  check(clSetKernelArg(ax_kernel, 4, sizeof(uint), &nelt),
        "clSetKernelArg(ax, 4)");
  check(clSetKernelArg(ax_kernel, 5, sizeof(uint), &nx1),
        "clSetKernelArg(ax, 5)");

  const size_t local_size[2] = {nx1, nx1};
  const size_t global_size[2] = {nelt * nx1, nx1};
  check(clEnqueueNDRangeKernel(ocl_queue, ax_kernel, 2, NULL, global_size,
                               local_size, 0, NULL, NULL),
        "clEnqueueNDRangeKernel(ax)");
}

static void gs(cl_mem *x, cl_mem *gs_off, cl_mem *gs_idx, const uint gs_n) {
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
  bp5_debug(bp5->verbose, "opencl_init: Initializing OpenCL backend ...\n");
  if (initialized)
    return;

  opencl_device_init(bp5);
  opencl_kernels_init(bp5->verbose);
  opencl_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "opencl_init: done.\n");
}

static scalar opencl_run(const struct bp5_t *bp5, const scalar *r) {
  if (!initialized)
    bp5_error("opencl_run: OpenCL backend is not initialized.");

  bp5_debug(bp5->verbose, "opencl_run: ... ");

  clock_t t0 = clock();

  // Copy rhs to device buffer r_mem.
  const uint n = bp5_get_local_dofs(bp5);
  check(clEnqueueWriteBuffer(ocl_queue, r_mem, CL_TRUE, 0, n * sizeof(scalar),
                             r, 0, NULL, NULL),
        "clEnqueueWriteBuffer(r)");

  // Run CG on the device.
  scalar rtz1 = 1, rtz2 = 0;
  mask(&r_mem, n);
  zero(&x_mem, n);
  scalar r0 = glsc3(&r_mem, &r_mem, &c_mem, n);
  for (uint i = 0; i < bp5->max_iter; ++i) {
    copy(&z_mem, &r_mem, n);

    rtz2 = rtz1;
    rtz1 = glsc3(&r_mem, &z_mem, &c_mem, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(&p_mem, &z_mem, beta, n);

    ax(&w_mem, &p_mem, &g_mem, &D_mem, bp5->nelt, bp5->nx1);
    gs(&w_mem, &gs_off_mem, &gs_idx_mem, bp5->gs_n);
    add2s2(&w_mem, &p_mem, 0.1, n);
    mask(&w_mem, n);

    scalar pap = glsc3(&w_mem, &p_mem, &c_mem, n);
    scalar alpha = rtz1 / pap;
    add2s2(&x_mem, &p_mem, alpha, n);
    add2s2(&r_mem, &w_mem, -alpha, n);
  }
  clFinish(ocl_queue);
  clock_t t1 = clock() - t0;

  bp5_debug(bp5->verbose, "done.\n");
  bp5_debug(bp5->verbose, "opencl_run: Iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "opencl_run: Residual = %e %e.\n", r0, rtz2);

  return ((double)t1) / CLOCKS_PER_SEC;
}

static void opencl_finalize(void) {
  if (!initialized)
    return;

  check(clReleaseCommandQueue(ocl_queue), "clReleaseCommandQueue");
  check(clReleaseContext(ocl_ctx), "clReleaseContext");
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

  initialized = 0;
}

void bp5_opencl_init(void) {
  bp5_register_backend("OPENCL", opencl_init, opencl_run, opencl_finalize);
}
