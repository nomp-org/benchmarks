#include "nekbone-backend.h"

#include <string.h>

static uint initialized = 0;

static scalar       *r, *x, *z, *p, *w;
scalar              *wrk;
static const scalar *c, *g, *D;
static const uint   *gs_off, *gs_idx;
// FIXME: This doesn't work with nompcc: static scalar *ur, *us, *ut;
static uint dofs, nelt, nx1, gs_n;

static void mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "mem_init: copy problem data to device ...\n");

  dofs = nekbone_get_local_dofs(nekbone);
  r    = nekbone_calloc(scalar, dofs);
  x    = nekbone_calloc(scalar, dofs);
  z    = nekbone_calloc(scalar, dofs);
  p    = nekbone_calloc(scalar, dofs);
  w    = nekbone_calloc(scalar, dofs);
#pragma nomp update(alloc : r[0, dofs], x[0, dofs], z[0, dofs], p[0, dofs],    \
                        w[0, dofs])

  c    = nekbone->c;
  g    = nekbone->g;
  D    = nekbone->D;
  nelt = nekbone->nelt;
  nx1  = nekbone->nx1;
#pragma nomp update(to : c[0, dofs], g[0, 6 * dofs], D[0, nx1 * nx1])

  gs_n   = nekbone->gs_n;
  gs_off = nekbone->gs_off;
  gs_idx = nekbone->gs_idx;
#pragma nomp update(to : gs_off[0, gs_n + 1], gs_idx[0, gs_off[gs_n]])

  wrk = nekbone_calloc(scalar, 1);
  nekbone_debug(nekbone->verbose, "mem_init: done.\n");
}

static void _nomp_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "nomp_init: initializing nomp backend ...\n");

  char verbose[BUFSIZ], device[BUFSIZ], platform[BUFSIZ];
  snprintf(verbose, BUFSIZ, "%u", nekbone->verbose);
  snprintf(device, BUFSIZ, "%u", nekbone->device);
  snprintf(platform, BUFSIZ, "%u", nekbone->platform);

  char backend[BUFSIZ];
  strncpy(backend, nekbone->backend, BUFSIZ);

  char *token = strtok(backend, ":");
  token = strtok(NULL, ":");
  nekbone_debug(nekbone->verbose, "nomp_init: backend = %s\n", token);

  const int argc   = 10;
  char     *argv[] = {"--nomp-device",      device,
                      "--nomp-backend",     token,
                      "--nomp-verbose",     verbose,
                      "--nomp-platform",    platform,
                      "--nomp-scripts-dir", NEKBONE_SCRIPTS_DIR};

#pragma nomp init(argc, argv)

  mem_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "nomp_init: done.\n");
}

inline static void zero(scalar *v, const int n) {
#pragma nomp for transform("nekbone", "grid_loop") name("zero")
  for (int i = 0; i < n; i++) v[i] = 0;
}

inline static void copy(scalar *a, const scalar *b, const int n) {
#pragma nomp for transform("nekbone", "grid_loop") name("copy")
  for (int i = 0; i < n; i++) a[i] = b[i];
}

inline static void mask(scalar *v, const int n) {
#pragma nomp for transform("nekbone", "grid_loop") name("mask")
  for (int i = 0; i < n; i++) {
    if (i == 0) v[i] = 0;
  }
}

inline static void add2s1(scalar *a, const scalar *b, const scalar c,
                          const int n) {
#pragma nomp for transform("nekbone", "grid_loop") name("add2s1")
  for (int i = 0; i < n; i++) a[i] = c * a[i] + b[i];
}

inline static void add2s2(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma nomp for transform("nekbone", "grid_loop") name("add2s2")
  for (uint i = 0; i < n; i++) a[i] += c * b[i];
}

inline static scalar glsc3(const scalar *a, const scalar *b, const scalar *c,
                           const int n) {
  // FIXME: This doesn't work with nompcc: scalar wrk[1] = {0};
  // FIXME: This doesn't work witn libnomp: scalar wrk[1];
  wrk[0] = 0;
#pragma nomp for reduce("wrk", "+") name("glsc3")
  for (int i = 0; i < n; i++) wrk[0] += a[i] * b[i] * c[i];
  return wrk[0];
}

inline static void gs(scalar *v, const uint *gs_off, const uint *gs_idx,
                      const int gs_n) {
#pragma nomp for transform("nekbone", "gs") name("gs")
  for (int i = 0; i < gs_n; i++) {
    scalar s = 0;
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++) s += v[gs_idx[j]];
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++) v[gs_idx[j]] = s;
  }
}

inline static void ax(const int nelt, const int nx1,
                      scalar       w[nelt][nx1][nx1][nx1],
                      const scalar u[nelt][nx1][nx1][nx1],
                      const scalar G[nelt][nx1][nx1][nx1][6],
                      const scalar D[nx1][nx1]) {
#pragma nomp for transform("nekbone", "ax") name("ax") jit("nx1")
  for (int e = 0; e < nelt; e++) {
    scalar ur[nx1][nx1][nx1];
    scalar us[nx1][nx1][nx1];
    scalar ut[nx1][nx1][nx1];
    for (int k = 0; k < nx1; k++) {
      for (int j = 0; j < nx1; j++) {
        for (int i = 0; i < nx1; i++) {
          ur[k][j][i] = 0;
          us[k][j][i] = 0;
          ut[k][j][i] = 0;
          for (int l = 0; l < nx1; l++) {
            ur[k][j][i] += D[i][l] * u[e][k][j][l];
            us[k][j][i] += D[j][l] * u[e][k][l][i];
            ut[k][j][i] += D[k][l] * u[e][l][j][i];
          }
        }
      }
    }

    for (int k = 0; k < nx1; k++) {
      for (int j = 0; j < nx1; j++) {
        for (int i = 0; i < nx1; i++) {
          scalar r_G00 = G[e][k][j][i][0];
          scalar r_G01 = G[e][k][j][i][1];
          scalar r_G02 = G[e][k][j][i][2];
          scalar r_G11 = G[e][k][j][i][3];
          scalar r_G12 = G[e][k][j][i][4];
          scalar r_G22 = G[e][k][j][i][5];
          scalar wr =
              r_G00 * ur[k][j][i] + r_G01 * us[k][j][i] + r_G02 * ut[k][j][i];
          scalar ws =
              r_G01 * ur[k][j][i] + r_G11 * us[k][j][i] + r_G12 * ut[k][j][i];
          scalar wt =
              r_G02 * ur[k][j][i] + r_G12 * us[k][j][i] + r_G22 * ut[k][j][i];
          ur[k][j][i] = wr;
          us[k][j][i] = ws;
          ut[k][j][i] = wt;
        }
      }
    }

    for (int k = 0; k < nx1; k++) {
      for (int j = 0; j < nx1; j++) {
        for (int i = 0; i < nx1; i++) {
          scalar wo = 0;
          for (int l = 0; l < nx1; l++) {
            wo += D[l][i] * ur[k][j][l] + D[l][j] * us[k][l][i] +
                  D[l][k] * ut[l][j][i];
          }
          w[e][k][j][i] = wo;
        }
      }
    }
  }
}

static scalar _nomp_run(const struct nekbone_t *nekbone, const scalar *f) {
  if (!initialized)
    nekbone_error("nomp_run: nomp backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "nomp_run: ... n=%u\n", n);

  clock_t t0 = clock();

  for (uint i = 0; i < n; i++) r[i] = f[i];
#pragma nomp update(to : r[0, n])

  scalar pap  = 0;
  scalar rtz1 = 1, rtz2 = 0;

  zero(x, n);

  mask(r, n);

  scalar rnorm = sqrt(glsc3(r, c, r, n));
  scalar r0    = rnorm;
  nekbone_debug(nekbone->verbose, "nomp_run: iteration 0, rnorm = %e\n", rnorm);
  for (uint i = 0; i < nekbone->max_iter; ++i) {
    copy(z, r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(r, c, z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0) beta = 0;
    add2s1(p, z, beta, n);

    ax(nelt, nx1, (scalar(*)[nx1][nx1][nx1])w,
       (const scalar(*)[nx1][nx1][nx1])p, (const scalar(*)[nx1][nx1][nx1][6])g,
       (const scalar(*)[nx1])D);
    gs(w, gs_off, gs_idx, gs_n);
    add2s2(w, p, 0.1, n);
    mask(w, n);

    pap = glsc3(w, c, p, n);

    scalar alpha = rtz1 / pap;
    add2s2(x, p, alpha, n);
    add2s2(r, w, -alpha, n);

    scalar rtr = glsc3(r, c, r, n);
    rnorm      = sqrt(rtr);
    nekbone_debug(nekbone->verbose - 1, "nomp_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }
#pragma nomp sync
  clock_t t1 = clock();

  nekbone_debug(nekbone->verbose, "nomp_run: done.\n");
  nekbone_debug(nekbone->verbose, "nomp_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "nomp_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void _nomp_finalize(void) {
  if (!initialized) return;

#pragma nomp update(free : r[0, dofs], x[0, dofs], z[0, dofs], p[0, dofs],     \
                        w[0, dofs])
  nekbone_free(&r);
  nekbone_free(&x);
  nekbone_free(&z);
  nekbone_free(&p);
  nekbone_free(&w);

#pragma nomp update(free : c[0, dofs], g[0, 6 * dofs], D[0, nx1 * nx1])

  nekbone_free(&wrk);

  initialized = 0;
}

void nekbone_nomp_init(void) {
  nekbone_register_backend("NOMP", _nomp_init, _nomp_run, _nomp_finalize);
}
