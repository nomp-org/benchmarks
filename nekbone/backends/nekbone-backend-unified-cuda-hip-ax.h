#define TOKEN_PASTE_(x, y) x##y
#define TOKEN_PASTE(x, y)  TOKEN_PASTE_(x, y)

#ifndef NX1
#error "NX1 must be defined !"
#endif

#define IDX2(i, j)       ((i) + NX1 * (j))
#define IDX4(i, j, k, e) ((i) + NX1 * ((j) + NX1 * ((k) + NX1 * (e))))

#define ax_kernel TOKEN_PASTE(ax_kernel_v00_, NX1)

__global__ static void __launch_bounds__(NX1 *NX1 *NX1)
    ax_kernel(scalar *__restrict__ w, const scalar *__restrict__ u,
              const scalar *__restrict__ G, const scalar *__restrict__ D) {
  const uint e = blockIdx.x;
  const uint i = threadIdx.x;
  const uint j = threadIdx.y;
  const uint k = threadIdx.z;

  __shared__ scalar s_D[NX1][NX1];
  __shared__ scalar s_ur[NX1][NX1][NX1];
  __shared__ scalar s_us[NX1][NX1][NX1];
  __shared__ scalar s_ut[NX1][NX1][NX1];

  if (k == 0) s_D[j][i] = D[IDX2(i, j)];
  scalar wr = 0;
  scalar ws = 0;
  scalar wt = 0;
  __syncthreads();

  for (uint l = 0; l < NX1; ++l) {
    wr += s_D[i][l] * u[IDX4(l, j, k, e)];
    ws += s_D[j][l] * u[IDX4(i, l, k, e)];
    wt += s_D[k][l] * u[IDX4(i, j, l, e)];
  }

  s_ur[k][j][i] = wr;
  s_us[k][j][i] = ws;
  s_ut[k][j][i] = wt;
  __syncthreads();

  const uint gbase = 6 * IDX4(i, j, k, e);
  scalar     r_G00 = G[gbase + 0];
  scalar     r_G01 = G[gbase + 1];
  scalar     r_G02 = G[gbase + 2];
  scalar     r_G11 = G[gbase + 3];
  scalar     r_G12 = G[gbase + 4];
  scalar     r_G22 = G[gbase + 5];

  wr = r_G00 * s_ur[k][j][i] + r_G01 * s_us[k][j][i] + r_G02 * s_ut[k][j][i];
  ws = r_G01 * s_ur[k][j][i] + r_G11 * s_us[k][j][i] + r_G12 * s_ut[k][j][i];
  wt = r_G02 * s_ur[k][j][i] + r_G12 * s_us[k][j][i] + r_G22 * s_ut[k][j][i];
  __syncthreads();

  s_ur[k][j][i] = wr;
  s_us[k][j][i] = ws;
  s_ut[k][j][i] = wt;
  __syncthreads();

  scalar wo = 0;
  for (uint l = 0; l < NX1; l++) {
    wo += s_D[l][i] * s_ur[k][j][l] + s_D[l][j] * s_us[k][l][i] +
          s_D[l][k] * s_ut[l][j][i];
  }
  w[IDX4(i, j, k, e)] = wo;
}

#undef ax_kernel

#undef NX1

#undef IDX2
#undef IDX4

#undef TOKEN_PASTE
#undef TOKEN_PASTE_
