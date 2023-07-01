#if !defined(__LIBBP5_IMPL_H__)
#define __LIBBP5_IMPL_H__

#include "bp5-types.h"
#include "bp5.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Dynamic memory allocation function.
#define bp5_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
BP5_INTERN void bp5_free_(void **p);
#define bp5_free(p) bp5_free_((void **)p)

// BP5 internal data structure.
struct bp5_t {
  // User input to define problem size and verbosity.
  uint nelt, nx1, verbose, niter;
  char backend[BUFSIZ];
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

// Register a single GPU backend. This function is called by each backend.
BP5_INTERN void bp5_register_backend(const char *name, void (*initialize)(int),
                                     double (*run)(const struct bp5_t *),
                                     void (*finalize)(void));

// Register GPU backends.
BP5_INTERN void bp5_register_backends(void);

// Initialize a given backend to run BP5.
void bp5_init_backend(const struct bp5_t *bp5);

// Un-register GPU backends.
BP5_INTERN void bp5_unregister_backends(void);

// Gather-scatter setup.
BP5_INTERN void bp5_gs_setup(struct bp5_t *bp5);

// Read quadrature points and weights.
BP5_INTERN void bp5_read_zwgll(struct bp5_t *bp5);

// Setup geometric factors.
BP5_INTERN void bp5_geom_setup(struct bp5_t *bp5);

// Setup derivative matrix.
BP5_INTERN void bp5_derivative_setup(struct bp5_t *bp5);

// Setup inverse multiplicity.
BP5_INTERN void bp5_inverse_multiplicity_setup(struct bp5_t *bp5);

#endif // __LIBBP5_IMPL_H__
