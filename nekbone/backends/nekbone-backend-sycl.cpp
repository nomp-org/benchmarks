#include <sycl/sycl.hpp>

#include "nekbone-backend.h"

#define NEKBONE_IDX2(i, j)       ((i) + nx1 * (j))
#define NEKBONE_IDX4(i, j, k, e) ((i) + nx1 * ((j) + nx1 * ((k) + nx1 * (e))))

using namespace sycl;

static uint initialized = 0;

static queue q;

static scalar *d_r, *d_x, *d_z, *d_p, *d_w;
static scalar *d_wrk, *wrk;
static scalar *d_c, *d_g, *d_D;
static uint   *d_gs_off, *d_gs_idx;

static void sycl_mem_init(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose,
                "sycl_mem_init: copy problem data to device ...\n");
  const uint n = nekbone_get_local_dofs(nekbone);

  // Allocate device buffers.
  d_r = malloc_device<scalar>(n, q);
  d_x = malloc_device<scalar>(n, q);
  d_z = malloc_device<scalar>(n, q);
  d_p = malloc_device<scalar>(n, q);
  d_w = malloc_device<scalar>(n, q);

  // Copy multiplicity array.
  d_c = malloc_device<scalar>(n, q);
  q.copy(nekbone->c, d_c, n);

  // Copy geometric factors and derivative matrix.
  const uint ng = 6 * n;
  d_g           = malloc_device<scalar>(ng, q);
  q.copy(nekbone->g, d_g, ng);

  const uint nx1 = nekbone->nx1, nx2 = nx1 * nx1;
  d_D = malloc_device<scalar>(nx2, q);
  q.copy(nekbone->D, d_D, nx2);

  // Copy gather-scatter offsets and indices.
  d_gs_off = malloc_device<uint>(nekbone->gs_n + 1, q);
  q.copy(nekbone->gs_off, d_gs_off, nekbone->gs_n + 1);

  d_gs_idx = malloc_device<uint>(nekbone->gs_off[nekbone->gs_n], q);
  q.copy(nekbone->gs_idx, d_gs_idx, nekbone->gs_off[nekbone->gs_n]);

  // Work array.
  wrk   = nekbone_calloc(scalar, n);
  d_wrk = malloc_device<scalar>(n, q);

  q.wait();
}

inline static void zero(scalar *x, const uint n) {
  q.parallel_for(n, [=](auto idx) { x[idx] = 0; });
}

inline static void mask(scalar *x) {
  q.single_task([=]() { x[0] = 0; });
}

inline static scalar glsc3(const scalar *a, const scalar *b, const scalar *c,
                           const uint n) {
  auto red = reduction(d_wrk, plus<scalar>{},
                       {property::reduction::initialize_to_identity{}});
  q.parallel_for(n, red, [=](auto id, auto &sum) {
    int    idx = static_cast<int>(id);
    scalar tmp = a[idx] * b[idx] * c[idx];
    sum += tmp;
  });
  q.copy(d_wrk, wrk, 1);
  return wrk[0];
}

inline static void copy(scalar *a, const scalar *b, const uint n) {
  q.copy(b, a, n);
}

inline static void add2s1(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
  q.parallel_for(n, [=](auto id) { a[id] = c * a[id] + b[id]; });
}

inline static void add2s2(scalar *a, const scalar *b, const scalar c,
                          const uint n) {
  q.parallel_for(n, [=](auto id) { a[id] += c * b[id]; });
}

static void gs(scalar *v, const uint *gs_off, const uint *gs_idx,
               const uint n) {
  q.parallel_for(n, [=](auto id) {
    scalar s = 0;
    for (uint j = gs_off[id]; j < gs_off[id + 1]; j++) s += v[gs_idx[j]];
    for (uint j = gs_off[id]; j < gs_off[id + 1]; j++) v[gs_idx[j]] = s;
  });
}

template <int nx1>
static void ax_(scalar *w, const scalar *u, const scalar *G, const scalar *D,
                const uint nelt) {
  auto cgh = [=](auto &h) {
    h.parallel_for(
        nd_range<3>{range<3>{nelt * nx1, nx1, nx1}, range<3>{nx1, nx1, nx1}},
        [=](auto id) {
          auto &s_ut = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                         scalar[nx1][nx1][nx1]>(id.get_group()));
          auto &s_ur = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                         scalar[nx1][nx1][nx1]>(id.get_group()));
          auto &s_us = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                         scalar[nx1][nx1][nx1]>(id.get_group()));
          auto &s_D  = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                        scalar[nx1][nx1]>(id.get_group()));

          const uint e = id.get_group(0);
          const uint i = id.get_local_id(0);
          const uint j = id.get_local_id(1);
          const uint k = id.get_local_id(2);

          s_ur[k][j][i] = 0;
          if (k == 0) s_D[j][i] = D[NEKBONE_IDX2(i, j)];
          s_us[k][j][i] = 0;
          s_ut[k][j][i] = 0;

          for (uint l = 0; l < nx1; l++) {
            s_ur[k][j][i] += s_D[i][l] * u[NEKBONE_IDX4(l, j, k, e)];
            s_us[k][j][i] += s_D[j][l] * u[NEKBONE_IDX4(i, l, k, e)];
            s_ut[k][j][i] += s_D[k][l] * u[NEKBONE_IDX4(i, j, l, e)];
          }

          const uint gbase = 6 * NEKBONE_IDX4(i, j, k, e);
          scalar     r_G00 = G[gbase + 0];
          scalar     r_G01 = G[gbase + 1];
          scalar     r_G02 = G[gbase + 2];
          scalar     r_G11 = G[gbase + 3];
          scalar     r_G12 = G[gbase + 4];
          scalar     r_G22 = G[gbase + 5];

          scalar wr = r_G00 * s_ur[k][j][i] + r_G01 * s_us[k][j][i] +
                      r_G02 * s_ut[k][j][i];
          scalar ws = r_G01 * s_ur[k][j][i] + r_G11 * s_us[k][j][i] +
                      r_G12 * s_ut[k][j][i];
          scalar wt = r_G02 * s_ur[k][j][i] + r_G12 * s_us[k][j][i] +
                      r_G22 * s_ut[k][j][i];
          id.barrier(access::fence_space::local_space);

          s_ur[k][j][i] = wr;
          s_us[k][j][i] = ws;
          s_ut[k][j][i] = wt;
          id.barrier(access::fence_space::local_space);

          scalar wo = 0;
          for (uint l = 0; l < nx1; l++) {
            wo += s_D[l][i] * s_ur[k][j][l] + s_D[l][j] * s_us[k][l][i] +
                  s_D[l][k] * s_ut[l][j][i];
          }
          w[NEKBONE_IDX4(i, j, k, e)] = wo;
        });
  };

  q.submit(cgh);
}

static void ax(scalar *w, const scalar *u, const scalar *G, const scalar *D,
               const uint nelt, const uint nx1) {
  switch (nx1) {
  case 1: ax_<1>(w, u, G, D, nelt); break;
  case 2: ax_<2>(w, u, G, D, nelt); break;
  case 3: ax_<3>(w, u, G, D, nelt); break;
  case 4: ax_<4>(w, u, G, D, nelt); break;
  case 5: ax_<5>(w, u, G, D, nelt); break;
  case 6: ax_<6>(w, u, G, D, nelt); break;
  case 7: ax_<7>(w, u, G, D, nelt); break;
  case 8: ax_<8>(w, u, G, D, nelt); break;
  case 9: ax_<9>(w, u, G, D, nelt); break;
  case 10: ax_<10>(w, u, G, D, nelt); break;
  }
}

static void sycl_init(const struct nekbone_t *nekbone) {
  if (initialized) return;
  nekbone_debug(nekbone->verbose, "sycl_init: initializing sycl backend ...\n");

  auto platforms = platform::get_platforms();
  if (nekbone->platform >= platforms.size())
    nekbone_error("sycl_init: platform id is invalid: %d", nekbone->platform);

  auto devices = platforms[nekbone->platform].get_devices();
  if (nekbone->device >= devices.size())
    nekbone_error("sycl_init: device id is invalid: %d", nekbone->device);

  property_list q_props{property::queue::in_order()};
  q = queue{devices[nekbone->device], q_props};

  sycl_mem_init(nekbone);

  initialized = 1;
  nekbone_debug(nekbone->verbose, "sycl_init: done.\n");
}

static scalar sycl_run(const struct nekbone_t *nekbone, const scalar *r) {
  if (!initialized)
    nekbone_error("sycl_run: sycl backend is not initialized.\n");

  const uint n = nekbone_get_local_dofs(nekbone);
  nekbone_debug(nekbone->verbose, "sycl_run: ... n=%u\n", n);

  clock_t t0 = clock();

  // Copy rhs to device buffer.
  q.copy(r, d_r, n);

  scalar pap = 0, rtz1 = 1, rtz2 = 0;

  // Zero out the solution.
  zero(d_x, n);

  // Apply Dirichlet BCs to RHS.
  mask(d_r);

  // Run CG on the device.
  scalar rnorm = std::sqrt(glsc3(d_r, d_c, d_r, n));
  scalar r0    = rnorm;
  nekbone_debug(nekbone->verbose, "0: sycl_run: iteration 0, rnorm = %e\n",
                rnorm);
  for (uint i = 0; i < nekbone->max_iter; ++i) {
    // Preconditioner (which is just a copy for now).
    copy(d_z, d_r, n);

    rtz2 = rtz1;
    rtz1 = glsc3(d_r, d_c, d_z, n);

    scalar beta = rtz1 / rtz2;
    if (i == 0) beta = 0;
    add2s1(d_p, d_z, beta, n);

    ax(d_w, d_p, d_g, d_D, nekbone->nelt, nekbone->nx1);
    gs(d_w, d_gs_off, d_gs_idx, nekbone->gs_n);
    add2s2(d_w, d_p, 0.1, n);
    mask(d_w);

    pap = glsc3(d_w, d_c, d_p, n);

    scalar alpha = rtz1 / pap;
    add2s2(d_x, d_p, alpha, n);
    add2s2(d_r, d_w, -alpha, n);

    rnorm = std::sqrt(glsc3(d_r, d_c, d_r, n));
    nekbone_debug(nekbone->verbose, "sycl_run: iteration %d, rnorm = %e\n",
                  i + 1, rnorm);
  }

  q.wait();
  clock_t t1 = clock();

  nekbone_debug(nekbone->verbose, "sycl_run: done.\n");
  nekbone_debug(nekbone->verbose, "sycl_run: iterations = %d.\n",
                nekbone->max_iter);
  nekbone_debug(nekbone->verbose, "sycl_run: residual = %e %e.\n", r0, rnorm);

  return ((double)t1 - t0) / CLOCKS_PER_SEC;
}

static void sycl_finalize(void) {
  if (!initialized) return;

  free(d_r, q);
  free(d_x, q);
  free(d_z, q);
  free(d_p, q);
  free(d_w, q);
  free(d_c, q);
  free(d_g, q);
  free(d_D, q);
  free(d_gs_off, q);
  free(d_gs_idx, q);
  free(d_wrk, q);
  nekbone_free(&wrk);

  initialized = 0;
}

NEKBONE_INTERN void nekbone_sycl_init(void) {
  nekbone_register_backend("SYCL", sycl_init, sycl_run, sycl_finalize);
}
