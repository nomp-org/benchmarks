#include <cstring>
#include <occa.hpp>

#include "nekbone-backend.h"

using namespace occa;

static uint initialized = 0;

static occa::device dev;

static memory d_r, d_x, d_z, d_p, d_w;
static memory d_c, d_g, d_D;
static memory d_gs_off, d_gs_idx;
static memory d_wrk;

static size_t  num_blocks;
static scalar *wrk;

static const size_t local_size = 512;

static void occa_device_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "occa_device_init: initialize device ...\n");

  char backend[BUFSIZ];
  strncpy(backend, nekbone->backend, BUFSIZ);

  char *token = strtok(backend, ":");
  token       = strtok(NULL, ":");

  dev.setup({{"mode", token}, {"device_id", 0}, {"platform_id", 0}});
}

static void occa_mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "occa_mem_init: copy problem data to device ...\n");
  const uint n = nekbone_get_local_dofs(nekbone);

  // Allocate device buffers.
  d_r = dev.malloc<scalar>(n);
  d_x = dev.malloc<scalar>(n);
  d_z = dev.malloc<scalar>(n);
  d_p = dev.malloc<scalar>(n);
  d_w = dev.malloc<scalar>(n);

  // Copy multiplicity array.
  d_c = dev.malloc<scalar>(n);
  d_c.copyFrom(nekbone->c);

  // Copy geometric factors and derivative matrix.
  const uint ng = 6 * n;
  d_g           = dev.malloc<scalar>(ng);
  d_g.copyFrom(nekbone->g);

  const uint nx1 = nekbone->nx1, nx2 = nx1 * nx1;
  d_D = dev.malloc<scalar>(nx2);
  d_D.copyFrom(nekbone->D);

  // Copy gather-scatter offsets and indices.
  d_gs_off = dev.malloc<uint>(nekbone->gs_n + 1);
  d_gs_off.copyFrom(nekbone->gs_off);

  d_gs_idx = dev.malloc<uint>(nekbone->gs_off[nekbone->gs_n]);
  d_gs_idx.copyFrom(nekbone->gs_idx);

  // Work array.
  num_blocks = (n + local_size - 1) / local_size;
  wrk        = nekbone_calloc(scalar, num_blocks);
  d_wrk      = dev.malloc<scalar>(num_blocks);
}

static kernel k_zero, k_mask, k_glsc3, k_copy, k_add2s1, k_add2s2, k_gs, k_ax;

static void occa_kernel_init(const struct nekbone_t *nekbone) {
  // Path is relative the installation directory.
  char okl_path[BUFSIZ];
  strncpy(okl_path, nekbone->scripts_dir, BUFSIZ);
  strncat(okl_path, "/nekbone.okl", 32);

  json kernel_info({{"defines/NX1", nekbone->nx1},
                    {"defines/BLOCK_SIZE", local_size},
                    {"defines/scalar", "double"},
                    {"defines/uint", "unsigned int"}});

  k_zero   = dev.buildKernel(okl_path, "zero", kernel_info);
  k_mask   = dev.buildKernel(okl_path, "mask", kernel_info);
  k_glsc3  = dev.buildKernel(okl_path, "glsc3", kernel_info);
  k_copy   = dev.buildKernel(okl_path, "copy", kernel_info);
  k_add2s1 = dev.buildKernel(okl_path, "add2s1", kernel_info);
  k_add2s2 = dev.buildKernel(okl_path, "add2s2", kernel_info);
  k_gs     = dev.buildKernel(okl_path, "gs", kernel_info);
  k_ax     = dev.buildKernel(okl_path, "ax", kernel_info);
}

inline static scalar glsc3(const memory &a, const memory &b, const memory &c,
                           const uint n) {
  k_glsc3(d_wrk, a, b, c, n);
  d_wrk.copyTo(wrk);
  for (size_t i = 1; i < num_blocks; i++) wrk[0] += wrk[i];
  return wrk[0];
}

static void occa_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "occa_init: initializing occa backend ...\n");

  occa_device_init(nekbone);
  occa_mem_init(nekbone);
  occa_kernel_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "occa_init: done.\n");
}

static scalar occa_run(const struct nekbone_t *nekbone, const scalar *r) {
  if (!initialized)
    nekbone_error("occa_run: occa backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "occa_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  d_r.copyFrom(r);

  scalar pap = 0, rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  k_zero(d_x, n);

  // Apply Dirichlet BCs to RHS.
  k_mask(d_r);

  // Run CG on the device.
  scalar rnorm = std::sqrt(glsc3(d_r, d_c, d_r, n));
  scalar r0    = rnorm;
  nekbone_debug(nekbone->verbose, "0: occa_run: iteration 0, rnorm = %e\n",
                rnorm);
  for (uint i = 0; i < nekbone->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    k_copy(d_z, d_r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(d_r, d_c, d_z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0) beta = 0;
    k_add2s1(d_p, d_z, beta, n);

    k_ax(d_w, d_p, d_g, d_D, nekbone->nelt, nekbone->nx1, n);
    k_gs(d_w, d_gs_off, d_gs_idx, nekbone->gs_n);
    k_add2s2(d_w, d_p, 0.1, n);
    k_mask(d_w);

    pap = glsc3(d_w, d_c, d_p, n);

    scalar alpha = rtz1 / pap;
    k_add2s2(d_x, d_p, alpha, n);
    k_add2s2(d_r, d_w, -alpha, n);

    rnorm = std::sqrt(glsc3(d_r, d_c, d_r, n));
    nekbone_debug(nekbone->verbose, "occa_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }

  dev.finish();
  clock_t t1 = clock();

  nekbone_debug(nekbone->verbose, "occa_run: done.\n");
  nekbone_debug(nekbone->verbose, "occa_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "occa_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void occa_finalize(void) {
  if (!initialized) return;

  nekbone_free(&wrk);

  initialized = 0;
}

NEKBONE_INTERN void nekbone_occa_init(void) {
  nekbone_register_backend("OCCA", occa_init, occa_run, occa_finalize);
}
