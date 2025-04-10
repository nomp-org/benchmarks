#include "nekbone-backend.h"

#include <omp.h>

static uint initialized = 0;

static scalar       *r, *x, *z, *p, *w;
static scalar       *wrk;
static const scalar *c, *g, *D;
static const uint   *gs_off, *gs_idx;
static uint          dofs, nelt, nx1, gs_n;

static void mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "mem_init: copy problem data to device ...\n");

  dofs = nekbone_get_local_dofs(nekbone);
  r    = nekbone_calloc(scalar, dofs);
  x    = nekbone_calloc(scalar, dofs);
  z    = nekbone_calloc(scalar, dofs);
  p    = nekbone_calloc(scalar, dofs);
  w    = nekbone_calloc(scalar, dofs);
#pragma omp target enter data map(alloc : r[0 : dofs], x[0 : dofs],            \
                                      z[0 : dofs], p[0 : dofs], w[0 : dofs])

  c    = nekbone->c;
  g    = nekbone->g;
  D    = nekbone->D;
  nelt = nekbone->nelt;
  nx1  = nekbone->nx1;
#pragma omp target enter data map(to : c[0 : dofs], g[0 : 6 * dofs],           \
                                      D[0 : nx1 * nx1])

  gs_n   = nekbone->gs_n;
  gs_off = nekbone->gs_off;
  gs_idx = nekbone->gs_idx;
#pragma omp target enter data map(to : gs_off[0 : gs_n + 1],                   \
                                      gs_idx[0 : gs_off[gs_n]])

  wrk = nekbone_calloc(scalar, 1);
  nekbone_debug(nekbone->verbose, "mem_init: done.\n");
}

static void omp_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "omp_init: initializing omp backend ...\n");
  nekbone_debug(nekbone->verbose, "omp_init: devices = %d\n",
                omp_get_num_devices());

  mem_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "omp_init: done.\n");
}

inline static void                  zero(scalar *v, const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) v[i] = 0;
}

inline static void copy(scalar *a, const scalar *b, const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) a[i] = b[i];
}

inline static void                  mask(scalar *v, const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) {
    if (i == 0) v[i] = 0;
  }
}

inline static void add2s1(scalar *a, const scalar *b, const scalar c,
                          const int n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) a[i] = c * a[i] + b[i];
}

inline static void add2s2(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
#pragma omp target teams distribute parallel for
  for (uint i = 0; i < n; i++) a[i] += c * b[i];
}

inline static scalar glsc3(const scalar *a, const scalar *b, const scalar *c,
                           const int n) {
  wrk[0] = 0;
#pragma omp target teams distribute parallel for reduction(+ : wrk[0])
  for (int i = 0; i < n; i++) wrk[0] += a[i] * b[i] * c[i];
  return wrk[0];
}

inline static void gs(scalar *v, const uint *gs_off, const uint *gs_idx,
                      const int gs_n) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < gs_n; i++) {
    scalar s = 0;
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++) s += v[gs_idx[j]];
    for (uint j = gs_off[i]; j < gs_off[i + 1]; j++) v[gs_idx[j]] = s;
  }
}

#define NX1 2
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 3
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 4
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 5
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 6
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 7
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 8
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 9
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 10
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 11
#include "nekbone-backend-omp-ax.h"
#undef NX1

#define NX1 12
#include "nekbone-backend-omp-ax.h"
#undef NX1

inline static void ax(scalar *w, const scalar *u, const scalar *G,
                      const scalar *D, const int nelt, const int nx1) {
  switch (nx1) {
  case 2: ax_kernel_2(w, u, g, D, nelt); break;
  case 3: ax_kernel_3(w, u, g, D, nelt); break;
  case 4: ax_kernel_4(w, u, g, D, nelt); break;
  case 5: ax_kernel_5(w, u, g, D, nelt); break;
  case 6: ax_kernel_6(w, u, g, D, nelt); break;
  case 7: ax_kernel_7(w, u, g, D, nelt); break;
  case 8: ax_kernel_8(w, u, g, D, nelt); break;
  case 9: ax_kernel_9(w, u, g, D, nelt); break;
  case 10: ax_kernel_10(w, u, g, D, nelt); break;
  case 11: ax_kernel_11(w, u, g, D, nelt); break;
  case 12: ax_kernel_12(w, u, g, D, nelt); break;
  default: break;
  }
}

static scalar omp_run(const struct nekbone_t *nekbone, const scalar *f) {
  if (!initialized) nekbone_error("omp_run: omp backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "omp_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  for (uint i = 0; i < n; i++) r[i] = f[i];
#pragma omp target update to(r[0 : n])

  scalar pap  = 0;
  scalar rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(x, n);

  // Apply Dirichlet BCs to RHS.
  mask(r, n);

  // Run CG on the device.
  scalar rnorm = sqrt(glsc3(r, c, r, n));
  nekbone_debug(nekbone->verbose - 1, "omp_run: iteration 0, rnorm = %e\n", rnorm);
  scalar r0 = rnorm;
  for (uint i = 0; i < nekbone->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(z, r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(r, c, z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0) beta = 0;
    add2s1(p, z, beta, n);

    ax(w, p, g, D, nelt, nx1);
    gs(w, gs_off, gs_idx, gs_n);
    add2s2(w, p, 0.1, n);
    mask(w, n);

    pap = glsc3(w, c, p, n);

    scalar alpha = rtz1 / pap;
    add2s2(x, p, alpha, n);
    add2s2(r, w, -alpha, n);

    scalar rtr = glsc3(r, c, r, n);
    rnorm      = sqrt(rtr);
    nekbone_debug(nekbone->verbose, "omp_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }
  clock_t t1 = clock();

  nekbone_debug(nekbone->verbose, "omp_run: done.\n");
  nekbone_debug(nekbone->verbose, "omp_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "omp_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void omp_finalize(void) {
  if (!initialized) return;

#pragma omp target exit data map(delete : r[0 : dofs], x[0 : dofs],            \
                                     z[0 : dofs], p[0 : dofs], w[0 : dofs])
  nekbone_free(&r);
  nekbone_free(&x);
  nekbone_free(&z);
  nekbone_free(&p);
  nekbone_free(&w);

#pragma omp target exit data map(delete : c[0 : dofs], g[0 : 6 * dofs],        \
                                     D[0 : nx1 * nx1])

  nekbone_free(&wrk);

  initialized = 0;
}

void nekbone_omp_init(void) {
  nekbone_register_backend("OMP", omp_init, omp_run, omp_finalize);
}
