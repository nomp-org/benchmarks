#include "nekbone-backend.h"

#define NEKBONE_BACKEND(function)                                              \
  NEKBONE_INTERN void function(void) __attribute__((weak));                    \
  void function(void) { return; }

#include "nekbone-backend-list.h"

#undef NEKBONE_BACKEND
