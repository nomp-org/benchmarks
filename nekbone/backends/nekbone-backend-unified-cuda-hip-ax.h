#define TOKEN_PASTE_(x, y) x##y
#define TOKEN_PASTE(x, y) TOKEN_PASTE_(x, y)

#ifndef NX1
#error "NX1 must be defined !"
#endif

#define NEKBONE_IDX2(i, j) ((i) + NX1 * (j))
#define NEKBONE_IDX3(i, j, k) ((i) + NX1 * ((j) + NX1 * (k)))
#define NEKBONE_IDX4(i, j, k, l) ((i) + NX1 * ((j) + NX1 * ((k) + NX1 * (l))))

#define ax_kernel TOKEN_PASTE(ax_kernel_v00_, NX1)

__global__ static void __launch_bounds__(NX1 *NX1 *NX1)
    ax_kernel(scalar *__restrict__ w, const scalar *__restrict__ u,
              const scalar *__restrict__ G, const scalar *__restrict__ D) {
  const uint ebase = blockIdx.x * NX1 * NX1 * NX1;
  const uint i = threadIdx.x;
  const uint j = threadIdx.y;
  const uint k = threadIdx.z;

  extern __shared__ scalar smem[];
  scalar *s_D = (scalar *)smem;
  scalar *s_ur = (scalar *)&s_D[NX1 * NX1];
  scalar *s_us = (scalar *)&s_ur[NX1 * NX1 * NX1];
  scalar *s_ut = (scalar *)&s_us[NX1 * NX1 * NX1];

  s_ur[NEKBONE_IDX3(i, j, k)] = 0;
  if (k == 0) s_D[NEKBONE_IDX2(i, j)] = D[NEKBONE_IDX2(i, j)];
  s_us[NEKBONE_IDX3(i, j, k)] = 0;
  s_ut[NEKBONE_IDX3(i, j, k)] = 0;
  __syncthreads();

  for (uint l = 0; l < NX1; ++l) {
    s_ur[NEKBONE_IDX3(i, j, k)] +=
        s_D[NEKBONE_IDX2(l, i)] * u[ebase + NEKBONE_IDX3(l, j, k)];
    s_us[NEKBONE_IDX3(i, j, k)] +=
        s_D[NEKBONE_IDX2(l, j)] * u[ebase + NEKBONE_IDX3(i, l, k)];
    s_ut[NEKBONE_IDX3(i, j, k)] +=
        s_D[NEKBONE_IDX2(l, k)] * u[ebase + NEKBONE_IDX3(i, j, l)];
  }
  __syncthreads();

  const uint gbase = 6 * (ebase + NEKBONE_IDX3(i, j, k));
  scalar r_G00 = G[gbase + 0];
  scalar r_G01 = G[gbase + 1];
  scalar r_G02 = G[gbase + 2];
  scalar r_G11 = G[gbase + 3];
  scalar r_G12 = G[gbase + 4];
  scalar r_G22 = G[gbase + 5];

  scalar wr = r_G00 * s_ur[NEKBONE_IDX3(i, j, k)] +
              r_G01 * s_us[NEKBONE_IDX3(i, j, k)] +
              r_G02 * s_ut[NEKBONE_IDX3(i, j, k)];
  scalar ws = r_G01 * s_ur[NEKBONE_IDX3(i, j, k)] +
              r_G11 * s_us[NEKBONE_IDX3(i, j, k)] +
              r_G12 * s_ut[NEKBONE_IDX3(i, j, k)];
  scalar wt = r_G02 * s_ur[NEKBONE_IDX3(i, j, k)] +
              r_G12 * s_us[NEKBONE_IDX3(i, j, k)] +
              r_G22 * s_ut[NEKBONE_IDX3(i, j, k)];
  __syncthreads();

  s_ur[NEKBONE_IDX3(i, j, k)] = wr;
  s_us[NEKBONE_IDX3(i, j, k)] = ws;
  s_ut[NEKBONE_IDX3(i, j, k)] = wt;
  __syncthreads();

  scalar wo = 0;
  for (uint l = 0; l < NX1; l++) {
    wo += s_D[NEKBONE_IDX2(i, l)] * s_ur[NEKBONE_IDX3(l, j, k)] +
          s_D[NEKBONE_IDX2(j, l)] * s_us[NEKBONE_IDX3(i, l, k)] +
          s_D[NEKBONE_IDX2(k, l)] * s_ut[NEKBONE_IDX3(i, j, l)];
  }
  w[ebase + NEKBONE_IDX3(i, j, k)] = wo;
}

#undef ax_kernel

#undef NEKBONE_IDX2
#undef NEKBONE_IDX3
#undef NEKBONE_IDX4

#undef TOKEN_PASTE
#undef TOKEN_PASTE_
