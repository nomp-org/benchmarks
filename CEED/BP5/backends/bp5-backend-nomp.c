#include "bp5-impl.h"

static uint initialized = 0;

static scalar *r, *x, *z, *p, *w;
static const scalar *c, *g, *D;
static const uint *gs_off, *gs_idx;

static void nomp_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "nomp_mem_init: Copy problem data to device ... ");

  // We allocate following arrays used in CG on both host and device.
  // Techinically we don't need the host arrays if we always run on the device.
  // But in the case nomp is not enabled, we need these arrays on host.
  uint dofs = bp5_get_local_dofs(bp5);
  r = bp5_calloc(scalar, dofs);
  x = bp5_calloc(scalar, dofs);
  z = bp5_calloc(scalar, dofs);
  p = bp5_calloc(scalar, dofs);
  w = bp5_calloc(scalar, dofs);
#pragma nomp update(alloc                                                      \
                    : r [0:dofs], x [0:dofs], z [0:dofs], p [0:dofs],          \
                      w [0:dofs])

  // There is no need to allcoate following arrays on host. We just copy them
  // into the device.
  c = bp5->c, g = bp5->g, D = bp5->D;
  gs_off = bp5->gs_off, gs_idx = bp5->gs_idx;
#pragma nomp update(to : c [0:dofs], g [0:6 * dofs], D [0:bp5->nx1 * bp5->nx1])

  bp5_debug(bp5->verbose, "done.\n");
}

inline static void mask(scalar *v, const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++) {
    if (i == 0)
      v[i] = 0;
  }
}

inline static void zero(scalar *v, const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++)
    v[i] = 0;
}

inline static void copy(scalar *a, const scalar *b, const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++)
    a[i] = b[i];
}

inline static void add2s1(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++)
    a[i] = c * a[i] + b[i];
}

inline static void add2s2(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++)
    a[i] += c * b[i];
}

inline static scalar glsc3(const scalar *a, const scalar *b, const scalar *c,
                           const uint n) {
  scalar wrk[1] = {0};
#pragma nomp for
  for (uint i = 0; i < n; i++)
    wrk[0] += a[i] * b[i] * c[i];
  return wrk[0];
}

inline static void gs(scalar *v, const uint gs_n) {
#pragma nomp for
  for (uint i = 0; i < gs_n; i++) {
    scalar s = 0;
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++)
      s += v[gs_idx[j]];
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++)
      v[gs_idx[j]] = s;
  }
}

inline static void ax(scalar *w, const scalar *p, const uint nelt,
                      const uint nx1) {
  // TODO: Implement ax_kernel.
  return;
}

static void nomp_init(const struct bp5_t *bp5) {
  if (initialized)
    return;

  bp5_debug(bp5->verbose, "nomp_init: Initializing NOMP backend ... ");
  const int argc = 6;
  char *argv[] = {"--nomp-device-id", "0", "--nomp-backend", "cuda",
                  "--nomp-verbose",   "0"};
#pragma nomp init(argc, argv)

  nomp_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "done.\n");
}

static scalar nomp_run(const struct bp5_t *bp5, const scalar *ri) {
  if (!initialized)
    bp5_error("nomp_run: NOMP backend is not initialized.\n");

  bp5_debug(bp5->verbose, "nomp_run: ... ");

  clock_t t0 = clock();
  // Copy rhs to device buffer
  const uint n = bp5_get_local_dofs(bp5);
  for (uint i = 0; i < n; i++)
    r[i] = ri[i];
#pragma nomp update(to : r [0:n])

  // Run CG on the device.
  scalar rtz1 = 1, rtz2 = 0;
  mask(r, n);
  zero(x, n);
  scalar r0 = glsc3(r, r, c, n);
  for (uint i = 0; i < bp5->max_iter; ++i) {
    copy(z, r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(r, z, c, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(p, z, beta, n);

    ax(w, p, bp5->nelt, bp5->nx1);
    gs(w, bp5->gs_n);
    add2s2(w, p, 0.1, n);
    mask(w, n);

    scalar pap = glsc3(w, p, c, n);
    scalar alpha = rtz1 / pap;
    add2s2(x, p, alpha, n);
    add2s2(r, w, -alpha, n);
  }
#pragma nomp sync
  clock_t t1 = clock() - t0;

  bp5_debug(bp5->verbose, "done.\n");
  bp5_debug(bp5->verbose, "nomp_run: Iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "nomp_run: Residual = %e %e.\n", r0, rtz2);

  return ((double)t1) / CLOCKS_PER_SEC;
}

static void nomp_finalize(void) {
  if (!initialized)
    return;

#pragma nomp update(free                                                       \
                    : r [0:dofs], x [0:dofs], z [0:dofs], p [0:dofs],          \
                      w [0:dofs], c [0:dofs], g [0:6 * dofs], D [0:nx1 * nx1])
  bp5_free(&r);
  bp5_free(&x);
  bp5_free(&z);
  bp5_free(&p);
  bp5_free(&w);
}

BP5_INTERN void bp5_nomp_init(void) {
  bp5_register_backend("NOMP", nomp_init, NULL, NULL);
}
