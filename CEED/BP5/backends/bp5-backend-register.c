#include "bp5-impl.h"
#include <string.h>

#define BP5_BACKEND(function) BP5_INTERN int function(void);
#include "bp5-backend-list.h"
#undef BP5_BACKEND

struct bp5_backend_t {
  char name[BUFSIZ];
  int (*initialize)(int);
  double (*run)(const struct bp5_t *);
};

static struct bp5_backend_t **backends = NULL;
static uint backends_count = 0;
static uint backends_capacity = 0;

void bp5_register_backend(const char *name, int (*initialize)(int),
                          double (*run)(const struct bp5_t *)) {
  struct bp5_backend_t *backend = bp5_calloc(struct bp5_backend_t, 1);
  strncpy(backend->name, name, BUFSIZ);
  backend->initialize = initialize;
  backend->run = run;
  // HASH_ADD_STR(backends, name, backend);

  if (backends_count == backends_capacity) {
    backends_capacity += backends_capacity / 2 + 1;
    backends = realloc(backends, backends_capacity * sizeof(*backends));
  }

  backends[backends_count++] = backend;
}

void bp5_register_backends(void) {
#define BP5_BACKEND(function) function();
#include "bp5-backend-list.h"
#undef BP5_BACKEND
}

void bp5_unregister_backends(void) {
  for (uint i = 0; i < backends_count; i++)
    bp5_free(&backends[i]);
  backends_count = 0;
}
