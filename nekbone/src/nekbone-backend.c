#include <ctype.h>
#include <string.h>

#include "nekbone-backend.h"

struct nekbone_backend_t {
  char name[BUFSIZ + 1];
  void (*initialize)(const struct nekbone_t *);
  double (*run)(const struct nekbone_t *, const scalar *);
  void (*finalize)(void);
};

static struct nekbone_backend_t **backends          = NULL;
static uint                       backends_count    = 0;
static uint                       backends_capacity = 0;
static int                        backend_active    = -1;

void nekbone_register_backend(const char *name,
                              void (*initialize)(const struct nekbone_t *),
                              double (*run)(const struct nekbone_t *,
                                            const scalar *),
                              void (*finalize)(void)) {
  struct nekbone_backend_t *backend =
      nekbone_calloc(struct nekbone_backend_t, 1);
  strncpy(backend->name, name, BUFSIZ);
  backend->name[BUFSIZ] = '\0';
  backend->initialize   = initialize;
  backend->run          = run;
  backend->finalize     = finalize;
  // HASH_ADD_STR(backends, name, backend);

  if (backends_count == backends_capacity) {
    backends_capacity += backends_capacity / 2 + 1;
    backends = realloc(backends, backends_capacity * sizeof(*backends));
  }

  backends[backends_count++] = backend;
}

void nekbone_register_backends(int verbose) {
  nekbone_debug(verbose, "nekbone_register_backends: ...\n");
#define NEKBONE_BACKEND(function) function();
#include "nekbone-backend-list.h"
#undef NEKBONE_BACKEND
  nekbone_debug(verbose, "nekbone_register_backends: backends_count=%d done.\n",
                backends_count);
  for (unsigned i = 0; i < backends_count; i++)
    nekbone_debug(verbose, "\tbackend %02d: %s\n", i, backends[i]->name);
}

void nekbone_init_backend(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_init_backend: ...\n");

  char         backend[BUFSIZ];
  const size_t len = strlen(nekbone->backend);
  for (size_t i = 0; i < len; i++) backend[i] = toupper(nekbone->backend[i]);
  backend[len] = '\0';

  for (uint i = 0; i < backends_count; i++) {
    const size_t len = strlen(backends[i]->name);
    if (strncmp(backends[i]->name, backend, len) != 0) continue;
    backends[i]->initialize(nekbone);
    nekbone_debug(nekbone->verbose, "nekbone_init_backend: %s done.\n",
                  backends[i]->name);
    backend_active = i;
    return;
  }
  nekbone_error("nekbone_init_backend: Unknown backend: %s\n", backend);
}

scalar nekbone_run_backend(const struct nekbone_t *nekbone, const scalar *rhs) {
  nekbone_debug(nekbone->verbose, "nekbone_run_backend: ...\n");

  nekbone_assert(backend_active >= 0 && backend_active < (int)backends_count,
                 "Invalid value for backend_active.");
  scalar elapased = backends[backend_active]->run(nekbone, rhs);
  nekbone_debug(nekbone->verbose, "nekbone_run_backend: %s done.\n",
                backends[backend_active]->name);
  return elapased;
}

void nekbone_unregister_backends(void) {
  if (backends_capacity == 0) return;

  for (uint i = 0; i < backends_count; i++) {
    backends[i]->finalize();
    nekbone_free(&backends[i]);
  }
  nekbone_free(&backends);
  backends_count = backends_capacity = 0;
}
