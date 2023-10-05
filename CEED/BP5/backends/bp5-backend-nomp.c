#include "bp5-backend.h"

static uint initialized = 0;

static scalar *r, *x, *z, *p, *w;
static const scalar *c, *g, *D;
static const uint *gs_off, *gs_idx;
static scalar *wrk;
static uint dofs, nx1, edofs, gs_n;

static void mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "mem_init: copy problem data to device ...\n");

  // We allocate following arrays used in CG on both host and device.
  // Techinically we don't need the host arrays if we always run on the device.
  // But in the case nomp is not enabled, we need these arrays on host.
  dofs = bp5_get_local_dofs(bp5);
  r = bp5_calloc(scalar, dofs);
  x = bp5_calloc(scalar, dofs);
  z = bp5_calloc(scalar, dofs);
  p = bp5_calloc(scalar, dofs);
  w = bp5_calloc(scalar, dofs);
#pragma nomp update(alloc : r[0, dofs], x[0, dofs], z[0, dofs], p[0, dofs],    \
                        w[0, dofs])

  // There is no need to allcoate following arrays on host. We just copy them
  // into the device.
  c = bp5->c, g = bp5->g, D = bp5->D;
  nx1 = bp5->nx1;
#pragma nomp update(to : c[0, dofs], g[0, 6 * dofs], D[0, nx1 * nx1])

  gs_n = bp5->gs_n;
  gs_off = bp5->gs_off, gs_idx = bp5->gs_idx;
#pragma nomp update(to : gs_off[0, gs_n + 1], gs_idx[0, gs_off[gs_n]])

  // Work array on device and host.
  edofs = bp5_get_elem_dofs(bp5);
  wrk = bp5_calloc(scalar, 6 * edofs);
#pragma nomp update(alloc : wrk[0, 6 * edofs])

  bp5_debug(bp5->verbose, "mem_init: done.\n");
}

static void _nomp_init(const struct bp5_t *bp5) {
  if (initialized)
    return;
  bp5_debug(bp5->verbose, "nomp_init: initializing nomp backend ...\n");

  char verbose[BUFSIZ], device[BUFSIZ], platform[BUFSIZ];
  snprintf(verbose, BUFSIZ, "%u", bp5->verbose);
  snprintf(device, BUFSIZ, "%u", bp5->device);
  snprintf(platform, BUFSIZ, "%u", bp5->platform);

  const int argc = 10;
  char *argv[] = {
      "--nomp-device",      device,         "--nomp-backend",  "hip",
      "--nomp-verbose",     verbose,        "--nomp-platform", platform,
      "--nomp-scripts-dir", BP5_SCRIPTS_DIR};

#pragma nomp init(argc, argv)

  mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "nomp_init: done.\n");
}

inline static void zero(scalar *v, const uint n) {
#pragma nomp for transform("bp5", "grid_loop")
  for (uint i = 0; i < n; i++)
    v[i] = 0;
}

inline static void copy(scalar *a, const scalar *b, const uint n) {
#pragma nomp for transform("bp5", "grid_loop")
  for (uint i = 0; i < n; i++)
    a[i] = b[i];
}

inline static void mask(scalar *v, const uint n) {
#pragma nomp for transform("bp5", "grid_loop")
  for (uint i = 0; i < n; i++) {
    if (i == 0)
      v[i] = 0;
  }
}

inline static void add2s1(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma nomp for transform("bp5", "grid_loop")
  for (uint i = 0; i < n; i++)
    a[i] = c * a[i] + b[i];
}

inline static void add2s2(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma nomp for transform("bp5", "grid_loop")
  for (uint i = 0; i < n; i++)
    a[i] += c * b[i];
}

inline static scalar glsc3(const scalar *a, const scalar *b, const scalar *c,
                           const uint n) {
  // FIXME: This doesn't work with nompcc: scalar wrk[1] = {0};
  scalar wrk[1];
  wrk[0] = 0;
#pragma nomp for reduce("wrk", "+")
  for (uint i = 0; i < n; i++)
    wrk[0] += a[i] * b[i] * c[i];
  return wrk[0];
}

inline static void gs(scalar *v, const uint *gs_off, const uint *gs_idx,
                      const uint gs_n) {
#pragma nomp for transform("bp5", "gs")
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
                      scalar *ur, scalar *us, scalar *ut) {
#pragma nomp for transform("bp5", "ax")
  for (uint e = 0; e < nelt; e++) {
    for (uint k = 0; k < nx1; k++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint i = 0; i < nx1; i++) {
          ur[BP5_IDX4(i, j, k, e)] = 0;
          us[BP5_IDX4(i, j, k, e)] = 0;
          ut[BP5_IDX4(i, j, k, e)] = 0;
          for (uint l = 0; l < nx1; l++) {
            ur[BP5_IDX4(i, j, k, e)] +=
                D[BP5_IDX2(l, i)] * u[BP5_IDX4(l, j, k, e)];
            us[BP5_IDX4(i, j, k, e)] +=
                D[BP5_IDX2(l, j)] * u[BP5_IDX4(i, l, k, e)];
            ut[BP5_IDX4(i, j, k, e)] +=
                D[BP5_IDX2(l, k)] * u[BP5_IDX4(i, j, l, e)];
          }
        }
      }
    }

    for (uint k1 = 0; k1 < nx1; k1++) {
      for (uint j1 = 0; j1 < nx1; j1++) {
        for (uint i1 = 0; i1 < nx1; i1++) {
          const uint gbase = 6 * BP5_IDX4(i1, j1, k1, e);
          scalar r_G00 = G[gbase + 0];
          scalar r_G01 = G[gbase + 1];
          scalar r_G02 = G[gbase + 2];
          scalar r_G11 = G[gbase + 3];
          scalar r_G12 = G[gbase + 4];
          scalar r_G22 = G[gbase + 5];
          scalar wr = r_G00 * ur[BP5_IDX4(i1, j1, k1, e)] +
                      r_G01 * us[BP5_IDX4(i1, j1, k1, e)] +
                      r_G02 * ut[BP5_IDX4(i1, j1, k1, e)];
          scalar ws = r_G01 * ur[BP5_IDX4(i1, j1, k1, e)] +
                      r_G11 * us[BP5_IDX4(i1, j1, k1, e)] +
                      r_G12 * ut[BP5_IDX4(i1, j1, k1, e)];
          scalar wt = r_G02 * ur[BP5_IDX4(i1, j1, k1, e)] +
                      r_G12 * us[BP5_IDX4(i1, j1, k1, e)] +
                      r_G22 * ut[BP5_IDX4(i1, j1, k1, e)];
          ur[BP5_IDX4(i1, j1, k1, e)] = wr;
          us[BP5_IDX4(i1, j1, k1, e)] = ws;
          ut[BP5_IDX4(i1, j1, k1, e)] = wt;
        }
      }
    }

    for (uint k2 = 0; k2 < nx1; k2++) {
      for (uint j2 = 0; j2 < nx1; j2++) {
        for (uint i2 = 0; i2 < nx1; i2++) {
          scalar wo = 0;
          for (uint l1 = 0; l1 < nx1; l1++) {
            wo += D[BP5_IDX2(i2, l1)] * ur[BP5_IDX4(l1, j2, k2, e)] +
                  D[BP5_IDX2(j2, l1)] * us[BP5_IDX4(i2, l1, k2, e)] +
                  D[BP5_IDX2(k2, l1)] * ut[BP5_IDX4(i2, j2, l1, e)];
          }
          w[BP5_IDX4(i2, j2, k2, e)] = wo;
        }
      }
    }
  }
}

static scalar _nomp_run(const struct bp5_t *bp5, const scalar *f) {
  if (!initialized)
    bp5_error("nomp_run: nomp backend is not initialized.\n");

  const uint n = bp5_get_local_dofs(bp5);
  bp5_debug(bp5->verbose, "nomp_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  for (uint i = 0; i < n; i++)
    r[i] = f[i];
#pragma nomp update(to : r[0, n])

  scalar pap = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(x, n);

  // Apply Dirichlet BCs to RHS.
  mask(r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(r, c, r, n));
  scalar r0 = rnorm;
  for (uint i = 0; i < bp5->max_iter; ++i) {
    copy(z, r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(r, c, z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0)
      beta = 0;
    add2s1(p, z, beta, n);

#if 0
    ax(w, p, g, D, bp5->nelt, nx1);
#else
    copy(w, p, n);
#endif
    gs(w, gs_off, gs_idx, gs_n);
    add2s2(w, p, 0.1, n);
    mask(w, n);

    pap = glsc3(w, c, p, n);

    scalar alpha = rtz1 / pap;
    add2s2(x, p, alpha, n);
    add2s2(r, w, -alpha, n);

    scalar rtr = glsc3(r, c, r, n);
    rnorm = sqrt(rtr);
    bp5_debug(bp5->verbose, "nomp_run: iteration %d, rnorm = %e\n", i, rnorm);
  }

#pragma nomp sync
  clock_t t1 = clock();

  bp5_debug(bp5->verbose, "nomp_run: done.\n");
  bp5_debug(bp5->verbose, "nomp_run: iterations = %d.\n", bp5->max_iter);
  bp5_debug(bp5->verbose, "nomp_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void _nomp_finalize(void) {
  if (!initialized)
    return;

#pragma nomp update(free : r[0, dofs], x[0, dofs], z[0, dofs], p[0, dofs],     \
                        w[0, dofs])
  bp5_free(&r);
  bp5_free(&x);
  bp5_free(&z);
  bp5_free(&p);
  bp5_free(&w);

#pragma nomp update(free : c[0, dofs], g[0, 6 * dofs], D[0, nx1 * nx1])

#pragma nomp update(free : wrk[0, 6 * edofs])
  bp5_free(&wrk);

  initialized = 0;
}

void bp5_nomp_init(void) {
  bp5_register_backend("NOMP", _nomp_init, _nomp_run, _nomp_finalize);
}
