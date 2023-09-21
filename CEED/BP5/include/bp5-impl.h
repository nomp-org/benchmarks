#if !defined(__LIBBP5_IMPL_H__)
#define __LIBBP5_IMPL_H__

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bp5-defs.h"
#include "bp5-types.h"
#include "bp5.h"

#define BP5_IDX2(i, j) ((i) + nx1 * (j))
#define BP5_IDX3(i, j, k) ((i) + nx1 * ((j) + nx1 * (k)))

// Dynamic memory allocation function.
#define bp5_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
BP5_INTERN void bp5_free_(void **p);
#define bp5_free(p) bp5_free_((void **)p)

// BP5 internal data structure.
struct bp5_t {
  // User input to define problem size and verbosity.
  sint nelt, nx1;
  uint verbose, max_iter, device_id, platform_id;
  char backend[BUFSIZ];
  // Internal data structure for gather-scatter.
  uint gs_n, *gs_off, *gs_idx;
  // Quadrature points and weights.
  scalar *z, *w;
  // Geometric factors array, derivative matrix, and inverse multiplicity.
  scalar *g, *D, *c;
};

//
// Setup BP5.
//
// Get the number of local DOFs.
BP5_INTERN uint bp5_get_local_dofs(const struct bp5_t *bp5);

// Get the number of elements DOFs.
BP5_INTERN uint bp5_get_elem_dofs(const struct bp5_t *bp5);

// Gather-scatter setup.
BP5_INTERN void bp5_gs_setup(struct bp5_t *bp5);

// Gather-scatter.
BP5_INTERN void bp5_gs(scalar *x, const struct bp5_t *bp5);

// Read quadrature points and weights.
BP5_INTERN void bp5_read_zwgll(struct bp5_t *bp5);

// Setup geometric factors.
BP5_INTERN void bp5_geom_setup(struct bp5_t *bp5);

// Setup derivative matrix.
BP5_INTERN void bp5_derivative_setup(struct bp5_t *bp5);

// Setup inverse multiplicity.
BP5_INTERN void bp5_inverse_multiplicity_setup(struct bp5_t *bp5);

// Apply inverse multiplicity.
BP5_INTERN void bp5_inverse_multiplicity(scalar *x, const struct bp5_t *bp5);

#endif // __LIBBP5_IMPL_H__
