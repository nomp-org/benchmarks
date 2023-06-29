#if !defined(__LIBBP5_IMPL_H__)
#define __LIBBP5_IMPL_H__

#include "bp5-types.h"
#include "bp5.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// BP5 internal data structure.
struct bp5_t {
  // User input to define problem size and verbosity.
  uint nelt, nx1, verbose;
  // Internal data structure for gather-scatter.
  uint gs_n, *gs_off, *gs_idx;
  // Quadrature points and weights.
  scalar *z, *w;
  // Geometric factors array.
  scalar *g;
  // Derivative matrix.
  scalar *D;
  // Inverse multiplicity of each DOF.
  scalar *c;
};

// Dynamic memory allocation function.
#define bp5_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
void bp5_free_(void **p);
#define bp5_free(p) bp5_free_((void **)p)

// Gather-scatter setup.
void bp5_gs_setup(struct bp5_t *bp5);

// Read quadrature points and weights.
void bp5_read_zwgll(struct bp5_t *bp5);

// Setup geometric factors.
void bp5_geom_setup(struct bp5_t *bp5);

// Setup derivative matrix.
void bp5_derivative_setup(struct bp5_t *bp5);

// Setup inverse multiplicity.
void bp5_inverse_multiplicity_setup(struct bp5_t *bp5);

#ifdef __cplusplus
}
#endif

#endif // __LIBBP5_IMPL_H__
