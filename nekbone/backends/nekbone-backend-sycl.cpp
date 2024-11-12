#include "nekbone-backend.h"

static uint initialized = 0;

static void sycl_init(const struct nekbone_t *nekbone) {}

static scalar sycl_run(const struct nekbone_t *nekbone, const scalar *r) {
  return 1.0;
}

static void sycl_finalize(void) {}

NEKBONE_INTERN void nekbone_sycl_init(void) {
  nekbone_register_backend("SYCL", sycl_init, sycl_run, sycl_finalize);
}
