#define TOKEN_PASTE_(x, y) x##y
#define TOKEN_PASTE(x, y)  TOKEN_PASTE_(x, y)

#ifndef NX1
#error "NX1 must be defined !"
#endif

#define NX2 (NX1 * NX1)
#define NX3 (NX1 * NX2)

#define IDX2(i, j)       ((i) + NX1 * (j))
#define IDX4(i, j, k, e) ((i) + NX1 * ((j) + NX1 * ((k) + NX1 * (e))))

#define ax_kernel TOKEN_PASTE(ax_kernel_v00_, NX1)

inline static void ax_kernel(scalar *w, const scalar *u, const scalar *G,
                             const scalar *D, const int nelt) {
  scalar s_D[NX1][NX1];
  scalar s_ur[NX1][NX1][NX1];
  scalar s_us[NX1][NX1][NX1];
  scalar s_ut[NX1][NX1][NX1];

  int    i, j, k, l, inner, e;
  scalar wr, ws, wt, wo;
  scalar r_G00, r_G01, r_G02, r_G11, r_G12, r_G22;

  // clang-format off
#pragma omp target teams distribute num_teams(nelt) thread_limit(NX3) \
  private(s_D, s_ur, s_us, s_ut)
  private(i, j, k, l, inner, e) \
  private(wr, ws, wt, wo) \
  private(r_G00, r_G01, r_G02, r_G11, r_G12,r_G22)
  // clang-format on
  for (e = 0; e < nelt; e++) {
#pragma omp parallel for private(i, j, k, inner)
    for (inner = 0; inner < NX3; inner++) {
      k = inner / NX2;
      j = (inner - k * NX2) / NX1;
      i = inner - k * NX2 - j * NX1;

      if (k == 0) s_D[j][i] = D[IDX2(i, j)];
    }

#pragma omp parallel for private(i, j, k, l, inner, e, wr, ws, wt)
    for (inner = 0; inner < NX3; inner++) {
      wr = 0;
      ws = 0;
      wt = 0;
      for (l = 0; l < NX1; l++) {
        wr += s_D[i][l] * u[IDX4(l, j, k, e)];
        ws += s_D[j][l] * u[IDX4(i, l, k, e)];
        wt += s_D[k][l] * u[IDX4(i, j, l, e)];
      }

      s_ur[k][j][i] = wr;
      s_us[k][j][i] = ws;
      s_ut[k][j][i] = wt;
    }

#pragma omp parallel for private(i, j, k, l, inner, e, wr, ws, wt) private(    \
        r_G00, r_G01, r_G02, r_G11, r_G12, r_G22)
    for (inner = 0; inner < NX3; inner++) {
      k = inner / NX2;
      j = (inner - k * NX2) / NX1;
      i = inner - k * NX2 - j * NX1;

      const uint gbase = 6 * IDX4(i, j, k, e);
      r_G00            = G[gbase + 0];
      r_G01            = G[gbase + 1];
      r_G02            = G[gbase + 2];
      r_G11            = G[gbase + 3];
      r_G12            = G[gbase + 4];
      r_G22            = G[gbase + 5];

      wr =
          r_G00 * s_ur[k][j][i] + r_G01 * s_us[k][j][i] + r_G02 * s_ut[k][j][i];
      ws =
          r_G01 * s_ur[k][j][i] + r_G11 * s_us[k][j][i] + r_G12 * s_ut[k][j][i];
      wt =
          r_G02 * s_ur[k][j][i] + r_G12 * s_us[k][j][i] + r_G22 * s_ut[k][j][i];

      s_ur[k][j][i] = wr;
      s_us[k][j][i] = ws;
      s_ut[k][j][i] = wt;
    }

#pragma omp parallel for private(i, j, k, l, inner, e, wo)
    for (inner = 0; inner < NX3; inner++) {
      k = inner / NX2;
      j = (inner - k * NX2) / NX1;
      i = inner - k * NX2 - j * NX1;

      wo = 0;
      for (l = 0; l < NX1; l++) {
        wo += s_D[l][i] * s_ur[k][j][l] + s_D[l][j] * s_us[k][l][i] +
              s_D[l][k] * s_ut[l][j][i];
      }
      // w[IDX4(i, j, k, e)] = u[IDX4(i, j, k, e)];
      w[IDX4(i, j, k, e)] = 1;
    }
  }
}

#undef ax_kernel

#undef NX1
#undef NX2
#undef NX3

#undef IDX2
#undef IDX4

#undef TOKEN_PASTE_
#undef TOKEN_PASTE
