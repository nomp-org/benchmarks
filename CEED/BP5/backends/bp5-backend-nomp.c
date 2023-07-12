#include "bp5-backend.h"

static uint initialized = 0;

static scalar *r, *x, *z, *p, *w;
static const scalar *c, *g, *D;
static const uint *gs_off, *gs_idx;
static scalar *wrk;

static void nomp_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "nomp_mem_init: Copy problem data to device ...\n");

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

  // Work array on device and host.
  wrk = bp5_calloc(scalar, 6 * bp5_get_elem_dofs(bp5));
#pragma nomp update(alloc : wrk [0:6 * bp5_get_elem_dofs(bp5)])

  bp5_debug(bp5->verbose, "nomp_mem_init: done.\n");
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

inline static void mask(scalar *v, const uint n) {
#pragma nomp for
  for (uint i = 0; i < n; i++) {
    if (i == 0)
      v[i] = 0;
  }
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

inline static void ax(scalar *w, const scalar *u, const scalar *G,
                      const scalar *D, const uint nelt, const uint nx1,
                      const uint ngeo) {
  scalar *ur = wrk;
  scalar *us = ur + nx1 * nx1 * nx1;
  scalar *ut = us + nx1 * nx1 * nx1;
  for (uint e = 0; e < nelt; e++) {
    const uint ebase = e * nx1 * nx1 * nx1;
    for (uint k = 0; k < nx1; k++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint i = 0; i < nx1; i++) {
          ur[BP5_IDX3(i, j, k)] = 0;
          us[BP5_IDX3(i, j, k)] = 0;
          ut[BP5_IDX3(i, j, k)] = 0;
          for (uint l = 0; l < nx1; l++) {
            ur[BP5_IDX3(i, j, k)] +=
                D[BP5_IDX2(l, i)] * u[ebase + BP5_IDX3(l, j, k)];
            us[BP5_IDX3(i, j, k)] +=
                D[BP5_IDX2(l, j)] * u[ebase + BP5_IDX3(i, l, k)];
            ut[BP5_IDX3(i, j, k)] +=
                D[BP5_IDX2(l, k)] * u[ebase + BP5_IDX3(i, j, l)];
          }
        }
      }
    }

    for (uint k = 0; k < nx1; k++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint i = 0; i < nx1; i++) {
          const uint gbase = ngeo * (ebase + BP5_IDX3(i, j, k));
          scalar r_G00 = G[gbase + 0];
          scalar r_G01 = G[gbase + 1];
          scalar r_G02 = G[gbase + 2];
          scalar r_G11 = G[gbase + 3];
          scalar r_G12 = G[gbase + 4];
          scalar r_G22 = G[gbase + 5];
          scalar wr = r_G00 * ur[BP5_IDX3(i, j, k)] +
                      r_G01 * us[BP5_IDX3(i, j, k)] +
                      r_G02 * ut[BP5_IDX3(i, j, k)];
          scalar ws = r_G01 * ur[BP5_IDX3(i, j, k)] +
                      r_G11 * us[BP5_IDX3(i, j, k)] +
                      r_G12 * ut[BP5_IDX3(i, j, k)];
          scalar wt = r_G02 * ur[BP5_IDX3(i, j, k)] +
                      r_G12 * us[BP5_IDX3(i, j, k)] +
                      r_G22 * ut[BP5_IDX3(i, j, k)];
          ur[BP5_IDX3(i, j, k)] = wr;
          us[BP5_IDX3(i, j, k)] = ws;
          ut[BP5_IDX3(i, j, k)] = wt;
        }
      }
    }

    for (uint k = 0; k < nx1; k++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint i = 0; i < nx1; i++) {
          scalar wo = 0;
          for (uint l = 0; l < nx1; l++) {
            wo += D[BP5_IDX2(i, l)] * ur[BP5_IDX3(l, j, k)] +
                  D[BP5_IDX2(j, l)] * us[BP5_IDX3(i, l, k)] +
                  D[BP5_IDX2(k, l)] * ut[BP5_IDX3(i, j, l)];
          }
          w[ebase + BP5_IDX3(i, j, k)] = wo;
        }
      }
    }
  }
}

static void nomp_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "nomp_init: Initializing NOMP backend ...\n");
  if (initialized)
    return;

  const int argc = 6;
  char *argv[] = {"--nomp-device-id", "0", "--nomp-backend", "cuda",
                  "--nomp-verbose",   "0"};
#pragma nomp init(argc, argv)

  nomp_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "nomp_init: done.\n");
}

static scalar nomp_run(const struct bp5_t *bp5, const scalar *f) {
  if (!initialized)
    bp5_error("nomp_run: NOMP backend is not initialized.\n");

  bp5_debug(bp5->verbose, "nomp_run: ... \n");

  const uint n = bp5_get_local_dofs(bp5);

  clock_t t0 = clock();

  scalar pap = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(x, n);

  // Copy rhs to device buffer
  for (uint i = 0; i < n; i++)
    r[i] = f[i];
#pragma nomp update(to : r [0:n])

  mask(r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(r, r, c, n)), r0 = rnorm;
  for (uint i = 0; i < bp5->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(z, r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(r, c, z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(p, z, beta, n);

    ax(w, p, g, D, bp5->nelt, bp5->nx1, 6);
    gs(w, bp5->gs_n);
    add2s2(w, p, 0.1, n);
    mask(w, n);

    pap = glsc3(w, c, p, n);

    scalar alpha = rtz1 / pap;
    add2s2(x, p, alpha, n);
    add2s2(r, w, -alpha, n);

    scalar rtr = glsc3(r, c, r, n);
    rnorm = sqrt(rtr);
  }
#pragma nomp sync
  clock_t t1 = clock();

  bp5_debug(bp5->verbose, "nomp_run: done.\n");
  bp5_debug(bp5->verbose, "nomp_run: Iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "nomp_run: Residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void nomp_finalize(void) {
  if (!initialized)
    return;

#pragma nomp update(free                                                       \
                    : r [0:dofs], x [0:dofs], z [0:dofs], p [0:dofs],          \
                      w [0:dofs])
  bp5_free(&r);
  bp5_free(&x);
  bp5_free(&z);
  bp5_free(&p);
  bp5_free(&w);

#pragma nomp update(free : c [0:dofs], g [0:6 * dofs], D [0:nx1 * nx1])

#pragma nomp update(free : wrk [0:3 * dofs])
  bp5_free(&wrk);

  initialized = 0;
}

void bp5_nomp_init(void) {
  bp5_register_backend("NOMP", nomp_init, nomp_run, nomp_finalize);
}
