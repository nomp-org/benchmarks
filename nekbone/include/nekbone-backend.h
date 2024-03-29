#if !defined(__LIBNEKBONE_BACKEND_H__)
#define __LIBNEKBONE_BACKEND_H__

#include "nekbone-impl.h"

//
// Register/Init Backends.
//
// Register a single GPU backend. This function is called by each backend.
NEKBONE_INTERN void nekbone_register_backend(
    const char *name, void (*initialize)(const struct nekbone_t *),
    double (*run)(const struct nekbone_t *, const scalar *),
    void (*finalize)(void));

NEKBONE_INTERN void nekbone_opencl_init(void);

NEKBONE_INTERN void nekbone_cuda_init(void);

NEKBONE_INTERN void nekbone_hip_init(void);

NEKBONE_INTERN void nekbone_nomp_init(void);

// Register GPU backends.
NEKBONE_INTERN void nekbone_register_backends(int verbose);

// Initialize a user selected backend to run NEKBONE.
NEKBONE_INTERN void nekbone_init_backend(const struct nekbone_t *nekbone);

NEKBONE_INTERN scalar nekbone_run_backend(const struct nekbone_t *nekbone,
                                          const scalar           *rhs);

// Un-register GPU backends.
NEKBONE_INTERN void nekbone_unregister_backends(void);

#endif // __LIBNEKBONE_BACKEND_H__
