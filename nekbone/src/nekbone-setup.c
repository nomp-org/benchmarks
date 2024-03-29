#include "nekbone-impl.h"

struct dof_t {
  ulong id;
  uint  idx;
};

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
  nekbone_assert(x > 0, "x must be a positive interger.");
  uint l  = 0;
  uint x_ = x;
  while (x_ >>= 1) l++;
  nekbone_assert((1 << l) == (int)x, "x must be a power of 2.");
  return l;
}

uint nekbone_get_local_dofs(const struct nekbone_t *nekbone) {
  uint ndof = nekbone->nelt;
  ndof *= nekbone->nx1 * nekbone->nx1 * nekbone->nx1;
  return ndof;
}

void nekbone_gs_setup(struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_gs_setup: ...\n");

  // Calculate nelx, nelz, and nelz from nelt. nelx, nely, and nelz are the
  // number of elements in x, y and z directions, respectively. `nelt` has to be
  // a power of 2.
  const uint nelt = nekbone->nelt, l = log_2(nelt);
  const uint nelx = 1 << (l + 2) / 3;
  const uint nely = 1 << (l - (l + 2) / 3 + 1) / 2;
  const uint nelz = nelt / (nelx * nely);
  nekbone_debug(nekbone->verbose, "nelx=%u, nely=%u, nelz=%u\n", nelx, nely,
                nelz);
  nekbone_assert(nelx * nely * nelz == nelt, "nelt must equal nelx*nely*nelz.");

  // Calculate the global element number for each local element.
  nekbone_debug(nekbone->verbose,
                "Calculating the global element number for each "
                "local element ...\n");
  uint      *lglel = nekbone_calloc(uint, nelt);
  uint       e     = 0;
  const uint nelxy = nelx * nely;
  for (uint k = 0; k < nelz; k++) {
    for (uint j = 0; j < nely; j++) {
      for (uint i = 0; i < nelx; i++) {
        lglel[e] = 1 + i + j * nelx + k * nelxy;
        e        = e + 1;
      }
    }
  }
  nekbone_assert(e == nelt, "e must equal nelt.");

  // Allocate memory for the global numbering of the dofs.
  const ulong ndof    = nekbone_get_local_dofs(nekbone);
  slong      *glo_num = nekbone_calloc(slong, ndof);

  // Number the dofs based on the element location in x, y and z and the
  // polynomial order.
  const uint nx1 = nekbone->nx1, p = nx1 - 1;
  for (uint e = 0; e < nelt; e++) {
    const uint eg = lglel[e];
    uint       ez = 1 + (eg - 1) / nelxy;
    uint       ex = (eg + nelx - 1) % nelx + 1;
    uint       ey = (eg + nelxy - 1) % nelxy + 1;
    ey            = 1 + (ey - 1) / nelx;
    ex--, ey--, ez--;
    for (uint k = 0; k < nx1; k++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint i = 0; i < nx1; i++) {
          uint dx = p * ex + i;
          uint dy = p * ey + j;
          uint dz = p * ez + k;
          uint ii = 1 + dx + dy * (p * nelx + 1) +
                    dz * (p * nelx + 1) * (p * nely + 1);
          uint ll = 1 + i + nx1 * j + nx1 * nx1 * k + nx1 * nx1 * nx1 * e;
          glo_num[ll - 1] = ii;
        }
      }
    }
  }
  nekbone_free(&lglel);

  // Sort the dofs based on the global number to find the unique dofs which are
  // repeated
  struct dof_t *dofs = nekbone_calloc(struct dof_t, ndof);
  for (uint d = 0; d < ndof; d++) {
    dofs[d].id  = glo_num[d];
    dofs[d].idx = d;
  }

  qsort(dofs, ndof, sizeof(struct dof_t), cmp_dof_t);

  uint rdof = 0, d0 = 0, gs_n = 0;
  for (uint d = 1; d < ndof; d++) {
    if (dofs[d].id != dofs[d0].id) {
      if (d - d0 > 1) gs_n++, rdof += d - d0;
      d0 = d;
    }
  }

  nekbone->gs_n   = gs_n;
  nekbone->gs_off = nekbone_calloc(uint, nekbone->gs_n + 1);
  nekbone->gs_idx = nekbone_calloc(uint, rdof);

  nekbone->gs_off[0] = gs_n = d0 = 0;
  for (uint d = 1; d < ndof; d++) {
    if (dofs[d].id != dofs[d0].id) {
      if (d - d0 > 1) {
        for (uint i = 0; i < d - d0; i++)
          nekbone->gs_idx[nekbone->gs_off[gs_n] + i] = dofs[d0 + i].idx;
        gs_n++;
        nekbone->gs_off[gs_n] = nekbone->gs_off[gs_n - 1] + d - d0;
      }
      d0 = d;
    }
  }

  nekbone_free(&glo_num);

  nekbone_debug(nekbone->verbose,
                "nekbone_gs_setup: gs_n=%u, repeated dof=%u\n", gs_n, rdof);
  nekbone_debug(nekbone->verbose, "nekbone_gs_setup: done.\n");
}

void nekbone_gs(scalar *c, const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_gs: ...\n");

  for (uint i = 0; i < nekbone->gs_n; i++) {
    scalar sum = 0;
    for (uint j = nekbone->gs_off[i]; j < nekbone->gs_off[i + 1]; j++)
      sum += c[nekbone->gs_idx[j]];
    for (uint j = nekbone->gs_off[i]; j < nekbone->gs_off[i + 1]; j++)
      c[nekbone->gs_idx[j]] = sum;
  }

  nekbone_debug(nekbone->verbose, "nekbone_gs: done.\n");
}

void nekbone_read_zwgll(struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_read_zwgll: ...\n");

  FILE *fp = fopen("data/zwgll.dat", "r");
  if (!fp) nekbone_error("nekbone_read_zwgll: zwgll.dat not found.\n");

  size_t     offset = 0;
  const uint nx1    = nekbone->nx1;
  for (uint lines = 2; lines < nx1; lines++) offset += lines;

  char buf[BUFSIZ];
  for (uint i = 0; i < offset; i++) {
    if (!fgets(buf, BUFSIZ, fp))
      nekbone_error("nekbone_read_zwgll: Order %u too large.\n", nx1 - 1);
  }

  nekbone->z = nekbone_calloc(scalar, nx1);
  nekbone->w = nekbone_calloc(scalar, nx1);
  for (uint i = 0; i < nx1; i++) {
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%lf %lf", &nekbone->z[i], &nekbone->w[i]);
  }

  fclose(fp);

  nekbone_debug(nekbone->verbose, "nekbone_read_zwgll: done.\n");
}

void nekbone_geom_setup(struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_geom_setup: ...\n");

  const uint nx1 = nekbone->nx1, nelt = nekbone->nelt;
  uint       dof = 0;
  nekbone->g     = nekbone_calloc(scalar, 6 * nekbone_get_local_dofs(nekbone));
  for (uint e = 0; e < nelt; e++) {
    for (uint i = 0; i < nx1; i++) {
      for (uint j = 0; j < nx1; j++) {
        for (uint k = 0; k < nx1; k++) {
          // Set only the diagonal of geometric factors.
          nekbone->g[6 * dof + 0] =
              nekbone->w[i] * nekbone->w[j] * nekbone->w[k];
          nekbone->g[6 * dof + 3] =
              nekbone->w[i] * nekbone->w[j] * nekbone->w[k];
          nekbone->g[6 * dof + 5] =
              nekbone->w[i] * nekbone->w[j] * nekbone->w[k];
          dof++;
        }
      }
    }
  }

  nekbone_debug(nekbone->verbose, "nekbone_geom_setup: done.\n");
}

void nekbone_derivative_setup(struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_derivative_setup: ...\n");

  const uint    nx1 = nekbone->nx1;
  const scalar *z   = nekbone->z;
  scalar       *a   = nekbone_calloc(scalar, nx1);
  for (uint i = 0; i < nx1; i++) {
    a[i] = 1;
    for (uint j = 0; j < i; j++) a[i] = a[i] * (z[i] - z[j]);
    for (uint j = i + 1; j < nx1; j++) a[i] = a[i] * (z[i] - z[j]);
    a[i] = 1 / a[i];
  }

  scalar *D = nekbone->D = nekbone_calloc(scalar, nx1 * nx1);
  for (uint i = 0; i < nx1; i++) {
    for (uint j = 0; j < nx1; j++) D[i * nx1 + j] = a[i] * (z[i] - z[j]);
    D[i * nx1 + i] = 1;
  }

  for (uint j = 0; j < nx1; j++) {
    for (uint i = 0; i < nx1; i++) D[j + i * nx1] /= a[j];
  }
  for (uint j = 0; j < nx1; j++) {
    for (uint i = 0; i < nx1; i++) D[i + nx1 * j] = 1.0 / D[i + nx1 * j];
  }

  for (uint i = 0; i < nx1; i++) {
    D[i + nx1 * i] = 0;
    scalar sum     = 0;
    for (uint j = 0; j < nx1; j++) sum = sum + D[i * nx1 + j];
    D[i + nx1 * i] = -sum;
  }

  nekbone_free(&a);

  nekbone_debug(nekbone->verbose, "nekbone_derivative_setup: done.\n");
}

void nekbone_inverse_multiplicity_setup(struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_inverse_multiplicity_setup: ...\n");

  uint    ndof = nekbone_get_local_dofs(nekbone);
  scalar *c = nekbone->c = nekbone_calloc(scalar, ndof);
  for (uint i = 0; i < ndof; i++) c[i] = 1;

  nekbone_gs(c, nekbone);

  for (uint i = 0; i < ndof; i++) c[i] = 1 / c[i];

  nekbone_debug(nekbone->verbose,
                "nekbone_inverse_multiplicity_setup: done.\n");
}

void nekbone_inverse_multiplicity(scalar *x, const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_inverse_multiplicity: ...\n");

  uint ndof = nekbone_get_local_dofs(nekbone);
  for (uint i = 0; i < ndof; i++) x[i] = x[i] * nekbone->c[i];

  nekbone_debug(nekbone->verbose, "nekbone_inverse_multiplicity: done.\n");
}
