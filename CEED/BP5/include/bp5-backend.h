#if !defined(__LIBBP5_BACKEND_H__)
#define __LIBBP5_BACKEND_H__

#include "bp5-impl.h"

//
// Register/Init Backends.
//
// Register a single GPU backend. This function is called by each backend.
BP5_INTERN void
bp5_register_backend(const char *name, void (*initialize)(const struct bp5_t *),
                     double (*run)(const struct bp5_t *, const scalar *),
                     void (*finalize)(void));

BP5_INTERN void bp5_opencl_init(void);

BP5_INTERN void bp5_cuda_init(void);

BP5_INTERN void bp5_hip_init(void);

BP5_INTERN void bp5_nomp_init(void);

// Register GPU backends.
BP5_INTERN void bp5_register_backends(int verbose);

// Initialize a user selected backend to run BP5.
BP5_INTERN void bp5_init_backend(const struct bp5_t *bp5);

BP5_INTERN scalar bp5_run_backend(const struct bp5_t *bp5, const scalar *rhs);

// Un-register GPU backends.
BP5_INTERN void bp5_unregister_backends(void);

#endif // __LIBBP5_BACKEND_H__
