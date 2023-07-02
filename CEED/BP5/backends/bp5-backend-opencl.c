#include "bp5-impl.h"
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

static const char *ERR_STR_OPENCL_FAILURE = "%s failed with error code: %d.";

#define check(call, msg)                                                       \
  {                                                                            \
    cl_int err_ = (call);                                                      \
    if (err_ != CL_SUCCESS)                                                    \
      bp5_error(ERR_STR_OPENCL_FAILURE, msg, err_);                            \
  }

// OpenCL device, context, queue and program.
static uint initialized = 0;
static cl_device_id ocl_device_id;
static cl_command_queue ocl_queue;
static cl_context ocl_ctx;

static void opencl_init_device(const struct bp5_t *bp5) {
  // Setup OpenCL platform.
  bp5_debug(bp5->verbose, "opencl_init: Initialize platform ...");
  cl_uint num_platforms = 0;
  check(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs");
  if (bp5->platform_id < 0 | bp5->platform_id >= num_platforms)
    bp5_error("opencl_init: Platform ID is invalid: %d", bp5->platform_id);

  cl_platform_id *cl_platforms = bp5_calloc(cl_platform_id, num_platforms);
  check(clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms),
        "clGetPlatformIDs");
  cl_platform_id platform = cl_platforms[bp5->platform_id];
  bp5_free(&cl_platforms);
  bp5_debug(bp5->verbose, "done.\n");

  // Setup OpenCL device.
  bp5_debug(bp5->verbose, "opencl_init: Initialize device ...");
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
  bp5_debug(bp5->verbose, "done.\n");

  // Setup OpenCL context and queue.
  cl_int err;
  bp5_debug(bp5->verbose, "opencl_init: Initialize context and queue ...");
  ocl_ctx = clCreateContext(NULL, 1, &ocl_device_id, NULL, NULL, &err);
  check(err, "clCreateContext");
  ocl_queue =
      clCreateCommandQueueWithProperties(ocl_ctx, ocl_device_id, 0, &err);
  check(err, "clCreateCommandQueueWithProperties");
  bp5_debug(bp5->verbose, "done.\n");
}

// OpenCL device buffers.
static cl_mem r_mem, x_mem, z_mem, p_mem, w_mem;
static cl_mem c_mem, g_mem, D_mem;
static cl_mem gs_off_mem, gs_idx_mem;

static void opencl_init_mem(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "opencl_init_mem: Copy problem data to device ...");

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

  bp5_debug(bp5->verbose, "done.\n");
}

// OpenCL kernels.
static cl_program ocl_program;
static cl_kernel mask_kernel, zero_kernel, copy_kernel;
static cl_kernel glsc3_kernel, add2s1_kernel, add2s2_kernel;
static cl_kernel ax_kernel, gs_kernel;
static const size_t local_size = 512;

static void opencl_init_kernels(const uint verbose) {
  bp5_debug(verbose, "opencl_init_kernels: Read kernel source ...");
  // FIXME: Don't hardcode path for the kernel file.
  FILE *fp = fopen("./backends/bp5-backend-opencl.cl", "r");
  if (!fp)
    bp5_error("opencl_init_kernels: Failed to open kernel source file.");

  fseek(fp, 0, SEEK_END);
  size_t knl_src_size = ftell(fp);
  char *knl_src = bp5_calloc(char, knl_src_size + 1);
  rewind(fp);

  fread(knl_src, sizeof(char), knl_src_size, fp);
  knl_src[knl_src_size] = '\0';
  fclose(fp);
  bp5_debug(verbose, "done.\n");

  // Build OpenCL kernels.
  bp5_debug(verbose, "opencl_init_kernels: Build kernels ...");
  cl_int err;
  ocl_program = clCreateProgramWithSource(ocl_ctx, 1, (const char **)&knl_src,
                                          NULL, &err);
  check(err, "clCreateProgramWithSource");
  bp5_free(&knl_src);
  check(clBuildProgram(ocl_program, 1, &ocl_device_id, NULL, NULL, NULL),
        "clBuildProgram");
  bp5_debug(verbose, "done.\n");

  bp5_debug(verbose, "opencl_init_kernels: Create kernels ...");
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
  bp5_debug(verbose, "done.\n");
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
  bp5_debug(bp5->verbose, "opencl_init: Initializing OpenCL backend... ");
  if (initialized)
    return;

  opencl_init_device(bp5);
  opencl_init_kernels(bp5->verbose);
  opencl_init_mem(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "done.\n");
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

  initialized = 0;
}

BP5_INTERN void bp5_opencl_init(void) {
  bp5_register_backend("OPENCL", opencl_init, opencl_run, opencl_finalize);
}
