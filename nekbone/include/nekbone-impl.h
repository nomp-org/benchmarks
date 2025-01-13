#if !defined(__LIBNEKBONE_IMPL_H__)
#define __LIBNEKBONE_IMPL_H__

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nekbone-defs.h"
#include "nekbone-types.h"
#include "nekbone.h"

// Dynamic memory allocation function.
#define nekbone_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
NEKBONE_INTERN void nekbone_free_(void **p);
#define nekbone_free(p) nekbone_free_((void **)p)

// NEKBONE internal data structure.
struct nekbone_t {
  // User input to define problem size and verbosity.
  sint nelt, nx1;
  uint verbose, max_iter, device, platform;
  char backend[BUFSIZ], scripts_dir[BUFSIZ];
  // Internal data structure for gather-scatter.
  uint gs_n, *gs_off, *gs_idx;
  // Quadrature points and weights.
  scalar *z, *w;
  // Geometric factors array, derivative matrix, and inverse multiplicity.
  scalar *g, *D, *c;
};

//
// Setup NEKBONE.
//
// Get the number of local DOFs.
NEKBONE_INTERN uint nekbone_get_local_dofs(const struct nekbone_t *nekbone);

// Gather-scatter setup.
NEKBONE_INTERN void nekbone_gs_setup(struct nekbone_t *nekbone);

// Gather-scatter.
NEKBONE_INTERN void nekbone_gs(scalar *x, const struct nekbone_t *nekbone);

// Read quadrature points and weights.
NEKBONE_INTERN void nekbone_read_zwgll(struct nekbone_t *nekbone);

// Setup geometric factors.
NEKBONE_INTERN void nekbone_geom_setup(struct nekbone_t *nekbone);

// Setup derivative matrix.
NEKBONE_INTERN void nekbone_derivative_setup(struct nekbone_t *nekbone);

// Setup inverse multiplicity.
NEKBONE_INTERN void
nekbone_inverse_multiplicity_setup(struct nekbone_t *nekbone);

// Apply inverse multiplicity.
NEKBONE_INTERN void
nekbone_inverse_multiplicity(scalar *x, const struct nekbone_t *nekbone);

#endif // __LIBNEKBONE_IMPL_H__
