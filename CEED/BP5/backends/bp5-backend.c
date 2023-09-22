#include "bp5-backend.h"
#include <string.h>

struct bp5_backend_t {
  char name[BUFSIZ];
  void (*initialize)(const struct bp5_t *);
  double (*run)(const struct bp5_t *, const scalar *);
  void (*finalize)(void);
};

static struct bp5_backend_t **backends = NULL;
static uint backends_count = 0;
static uint backends_capacity = 0;
static int backend_active = -1;

void bp5_register_backend(const char *name,
                          void (*initialize)(const struct bp5_t *),
                          double (*run)(const struct bp5_t *, const scalar *),
                          void (*finalize)(void)) {
  struct bp5_backend_t *backend = bp5_calloc(struct bp5_backend_t, 1);
  strncpy(backend->name, name, BUFSIZ);
  backend->initialize = initialize;
  backend->run = run;
  backend->finalize = finalize;
  // HASH_ADD_STR(backends, name, backend);

  if (backends_count == backends_capacity) {
    backends_capacity += backends_capacity / 2 + 1;
    backends = realloc(backends, backends_capacity * sizeof(*backends));
  }

  backends[backends_count++] = backend;
}

void bp5_register_backends(int verbose) {
  bp5_debug(verbose, "bp5_register_backends: ...\n");
#define BP5_BACKEND(function) function();
#include "bp5-backend-list.h"
#undef BP5_BACKEND
  bp5_debug(verbose, "bp5_register_backends: backends_count=%d done.\n",
            backends_count);
  for (unsigned i = 0; i < backends_count; i++)
    bp5_debug(verbose, "\tbackend %02d: %s\n", i, backends[i]->name);
}

void bp5_init_backend(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "bp5_init_backend: ...\n");

  for (uint i = 0; i < backends_count; i++) {
    if (strncmp(backends[i]->name, bp5->backend, BUFSIZ) == 0) {
      backends[i]->initialize(bp5);
      bp5_debug(bp5->verbose, "bp5_init_backend: %s done.\n",
                backends[i]->name);
      backend_active = i;
      return;
    }
  }
  bp5_error("bp5_init_backend: Unknown backend: %s\n", bp5->backend);
}

scalar bp5_run_backend(const struct bp5_t *bp5, const scalar *rhs) {
  bp5_debug(bp5->verbose, "bp5_run_backend: ...\n");

  bp5_assert(backend_active >= 0 && backend_active < (int)backends_count,
             "Invalid value for backend_active.");
  scalar elapased = backends[backend_active]->run(bp5, rhs);
  bp5_debug(bp5->verbose, "bp5_run_backend: %s done.\n",
            backends[backend_active]->name);
  return elapased;
}

void bp5_unregister_backends(void) {
  if (backends_capacity == 0)
    return;

  for (uint i = 0; i < backends_count; i++) {
    backends[i]->finalize();
    bp5_free(&backends[i]);
  }
  bp5_free(&backends);
  backends_count = backends_capacity = 0;
}
