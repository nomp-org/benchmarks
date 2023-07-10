#include "bp5-backend.h"

#define BP5_BACKEND(function)                                                  \
  BP5_INTERN void function(void) __attribute__((weak));                        \
  void function(void) { return; }

#include "bp5-backend-list.h"

#undef BP5_BACKEND
