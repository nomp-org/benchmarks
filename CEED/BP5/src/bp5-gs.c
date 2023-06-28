#include "bp5-impl.h"

void bp5_gs_setup(const struct bp5_t *bp5) {
  const uint nelt = bp5->nelt;

  uint nelx = cbrt(nelt + 1);
  while (nelt % nelx != 0)
    nelx--;

  uint nely = sqrt(nelt / nelx + 1);
  while (nelt % (nelx * nely) != 0)
    nely--;

  uint nelz = nelt / (nelx * nely);
  bp5_debug(bp5->verbose, "nelx = %u, nely = %u, nelz = %u\n", nelx, nely,
            nelz);
  bp5_assert(nelt == nelx * nely * nelz, "nelt = nelx * nely * nelz");

  const uint nx1 = bp5->nx1, p = nx1 - 1;
  const uint nx3 = nx1 * nx1 * nx1;
  const ulong ndof = nelt * nx3;

  slong *glo_num = bp5_calloc(slong, ndof);
  uint d = 0;
  for (uint e = 0; e < nelt; e++) {
    uint ex = e % nelx;
    uint ez = e / (nelx * nely);
    uint ey = (e - ex - ez * nelx * nely) / nelx;
    for (uint px = 0; px < nx1; px++) {
      for (uint py = 0; py < nx1; py++) {
        for (uint pz = 0; pz < nx1; pz++) {
          uint dx = p * ex + px;
          uint dy = p * ey + py;
          uint dz = p * ez + pz;
          glo_num[d++] =
              dx + (p * nelx + 1) * dy + (p * nelx + 1) * (p * nely + 1) * dz;
        }
      }
    }
  }

  bp5_free(&glo_num);
}

