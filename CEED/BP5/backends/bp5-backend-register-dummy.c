#include "bp5-impl.h"

#define BP5_BACKEND(function)                                                  \
  BP5_INTERN int function(void) __attribute__((weak));                         \
  int function(void) { return 0; }

#include "bp5-backend-list.h"

#undef BP5_BACKEND
