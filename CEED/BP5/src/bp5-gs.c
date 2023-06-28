#include "bp5-impl.h"

struct dof_t {
  ulong id;
  uint idx;
};

static inline ulong get_local_dofs(const struct bp5_t *bp5) {
  const uint nx1 = bp5->nx1, nx3 = nx1 * nx1 * nx1;
  ulong ndof = bp5->nelt;
  ndof *= nx3;
  return ndof;
}

static int cmp_dof_t(const void *a, const void *b) {
  const struct dof_t *pa = (const struct dof_t *)a;
  const struct dof_t *pb = (const struct dof_t *)b;

  if (pa->id > pb->id)
    return 1;
  else if (pa->id < pb->id)
    return -1;

  // If pa->id == pb->id, so we look at the index.
  if (pa->idx > pb->idx)
    return 1;
  else
    return -1;
}

void bp5_gs_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_gs_setup: ...");

  // Calculate nelx, nelz, and nelz from nelt. nelx, nely, and nelz are the
  // number of elements in x, y and z directions, respectively.
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

  // Number the dofs based on the element location in x, y and z and the
  // polynomial order.
  const uint nx1 = bp5->nx1, p = nx1 - 1;
  const ulong ndof = get_local_dofs(bp5);
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

  // Sort the dofs based on the global number to find the unique dofs which are
  // repeated
  struct dof_t *dofs = bp5_calloc(struct dof_t, ndof);
  for (uint d = 0; d < ndof; d++) {
    dofs[d].id = glo_num[d];
    dofs[d].idx = d;
  }

  qsort(dofs, ndof, sizeof(struct dof_t), cmp_dof_t);

  uint rdof = 0, d0 = 0, gs_n = 0;
  for (uint d = 1; d < ndof; d++) {
    if (dofs[d].id != dofs[d0].id) {
      if (d - d0 > 1)
        gs_n++, rdof += d - d0;
      d0 = d;
    }
  }

  bp5->gs_n = gs_n;
  bp5->gs_off = bp5_calloc(uint, bp5->gs_n + 1);
  bp5->gs_idx = bp5_calloc(uint, rdof);

  bp5->gs_off[0] = gs_n = d0 = 0;
  for (uint d = 1; d < ndof; d++) {
    if (dofs[d].id != dofs[d0].id) {
      if (d - d0 > 1) {
        for (uint i = 0; i < d - d0; i++)
          bp5->gs_idx[bp5->gs_off[gs_n] + i] = dofs[d0 + i].idx;
        gs_n++;
        bp5->gs_off[gs_n] = bp5->gs_off[gs_n - 1] + d - d0;
      }
      d0 = d;
    }
  }

  bp5_free(&glo_num);

  bp5_debug(bp5->verbose, "done.\n");
}

void bp5_geom_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_geom_setup: ...");

  FILE *fp = fopen("zwgll.txt", "r");
  if (!fp)
    bp5_error("bp5_geom_setup: zwgll.txt not found.\n");

  size_t offset = 0;
  for (uint lines = 2; lines < bp5->nx1; lines++)
    offset += lines;

  char buf[BUFSIZ];
  for (uint i = 0; i < offset; i++) {
    if (!fgets(buf, BUFSIZ, fp))
      bp5_error("bp5_geom_setup: Order %u too large.\n", bp5->nx1 - 1);
  }

  bp5->z = bp5_calloc(scalar, bp5->nx1);
  bp5->w = bp5_calloc(scalar, bp5->nx1);
  for (uint i = 0; i < bp5->nx1; i++) {
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%lf %lf", &bp5->z[i], &bp5->w[i]);
  }

  fclose(fp);

  uint dof = 0;
  bp5->g = bp5_calloc(scalar, get_local_dofs(bp5));
  for (uint e = 0; e < bp5->nelt; e++) {
    for (uint i = 0; i < bp5->nx1; i++) {
      for (uint j = 0; j < bp5->nx1; j++) {
        for (uint k = 0; k < bp5->nx1; k++)
          bp5->g[dof++] = bp5->w[i] * bp5->w[j] * bp5->w[k];
      }
    }
  }

  bp5_debug(bp5->verbose, "done.\n");
}
