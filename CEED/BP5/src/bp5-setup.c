#include "bp5-impl.h"

struct dof_t {
  ulong id;
  uint idx;
};

uint bp5_get_elem_dofs(const struct bp5_t *bp5) {
  const uint nx1 = bp5->nx1;
  return nx1 * nx1 * nx1;
}

uint bp5_get_local_dofs(const struct bp5_t *bp5) {
  uint ndof = bp5->nelt;
  ndof *= bp5_get_elem_dofs(bp5);
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

static uint log_2(const uint x) {
  bp5_assert(x > 0, "x must be a positive interger.");
  uint l = 0;
  uint x_ = x;
  while (x_ >>= 1)
    l++;
  bp5_assert((1 << l) == x, "x must be a power of 2.");
  return l;
}

void bp5_gs_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_gs_setup: ...\n");

  // Calculate nelx, nelz, and nelz from nelt. nelx, nely, and nelz are the
  // number of elements in x, y and z directions, respectively. `nelt` has to be
  // a power of 2.
  const uint nelt = bp5->nelt, l = log_2(nelt);
  const uint nelx = 1 << (l + 2) / 3;
  const uint nely = 1 << (l - (l + 2) / 3 + 1) / 2;
  const uint nelz = nelt / (nelx * nely);
  bp5_debug(bp5->verbose, "nelx=%u, nely=%u, nelz=%u\n", nelx, nely, nelz);
  bp5_assert(nelx * nely * nelz == nelt, "nelt must equal nelx*nely*nelz.");

  // Number the dofs based on the element location in x, y and z and the
  // polynomial order.
  const uint nx1 = bp5->nx1, p = nx1 - 1;
  const ulong ndof = bp5_get_local_dofs(bp5);
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

  bp5_debug(bp5->verbose, "bp5_gs_setup: gs_n=%u, repeated dof=%u\n", gs_n,
            rdof);
  bp5_debug(bp5->verbose, "bp5_gs_setup: done.\n");
}

void bp5_read_zwgll(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_read_zwgll: ...\n");

  FILE *fp = fopen("data/zwgll.dat", "r");
  if (!fp)
    bp5_error("bp5_read_zwgll: zwgll.dat not found.\n");

  size_t offset = 0;
  const uint nx1 = bp5->nx1;
  for (uint lines = 2; lines < nx1; lines++)
    offset += lines;

  char buf[BUFSIZ];
  for (uint i = 0; i < offset; i++) {
    if (!fgets(buf, BUFSIZ, fp))
      bp5_error("bp5_read_zwgll: Order %u too large.\n", nx1 - 1);
  }

  bp5->z = bp5_calloc(scalar, nx1);
  bp5->w = bp5_calloc(scalar, nx1);
  for (uint i = 0; i < nx1; i++) {
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%lf %lf", &bp5->z[i], &bp5->w[i]);
  }

  fclose(fp);

  bp5_debug(bp5->verbose, "bp5_read_zwgll: done.\n");
}

void bp5_geom_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_geom_setup: ...\n");

  const uint nx1 = bp5->nx1, nelt = bp5->nelt;
  uint dof = 0;
  bp5->g = bp5_calloc(scalar, 6 * bp5_get_local_dofs(bp5));
  for (uint e = 0; e < nelt; e++) {
    for (uint i = 0; i < nx1; i++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint k = 0; k < nx1; k++) {
          // Set only the diagonal of geometric factors.
          bp5->g[6 * dof + 0] = bp5->w[i] * bp5->w[j] * bp5->w[k];
          bp5->g[6 * dof + 3] = bp5->w[i] * bp5->w[j] * bp5->w[k];
          bp5->g[6 * dof + 5] = bp5->w[i] * bp5->w[j] * bp5->w[k];
          dof++;
        }
      }
    }
  }

  bp5_debug(bp5->verbose, "bp5_geom_setup: done.\n");
}

void bp5_derivative_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_derivative_setup: ...\n");

  uint nx1 = bp5->nx1;
  scalar *z = bp5->z;
  scalar *a = bp5_calloc(scalar, nx1);
  for (uint i = 0; i < nx1; i++) {
    a[i] = 1;
    for (uint j = 0; j < i; j++)
      a[i] = a[i] * (z[i] - z[j]);
    for (uint j = i + 1; j < nx1; j++)
      a[i] = a[i] * (z[i] - z[j]);
    a[i] = 1 / a[i];
  }

  scalar *D = bp5->D = bp5_calloc(scalar, nx1 * nx1);
  uint k = 0;
  for (uint j = 0; j < nx1; j++) {
    for (uint i = 0; i < nx1; i++) {
      D[k] = 0;
      if (i != j)
        D[k] = a[j] / (a[i] * (z[i] - z[j]));
      k++;
    }
  }

  for (uint i = 0; i < nx1; i++) {
    k = i;
    scalar sum = 0;
    for (uint j = 0; j < nx1; j++, k += nx1)
      sum = sum + D[k];
    D[i + nx1 * i] = -sum;
  }

  bp5_free(&a);

  bp5_debug(bp5->verbose, "bp5_derivative_setup: done.\n");
}

static void gs(scalar *c, const struct bp5_t *bp5) {
  for (uint i = 0; i < bp5->gs_n; i++) {
    scalar sum = 0;
    for (uint j = bp5->gs_off[i]; j < bp5->gs_off[i + 1]; j++)
      sum += c[bp5->gs_idx[j]];
    for (uint j = bp5->gs_off[i]; j < bp5->gs_off[i + 1]; j++)
      c[bp5->gs_idx[j]] = sum;
  }
}

void bp5_inverse_multiplicity_setup(struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_inverse_multiplicity_setup: ...\n");

  uint ndof = bp5_get_local_dofs(bp5);
  scalar *c = bp5->c = bp5_calloc(scalar, ndof);
  for (uint i = 0; i < ndof; i++)
    c[i] = 1;

  gs(c, bp5);

  for (uint i = 0; i < ndof; i++)
    c[i] = 1 / c[i];

  bp5_debug(bp5->verbose, "bp5_inverse_multiplicity_setup: done.\n");
}
