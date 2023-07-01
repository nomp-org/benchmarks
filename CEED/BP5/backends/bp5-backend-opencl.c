#include "bp5-impl.h"

BP5_INTERN void bp5_opencl_init(void) {
  bp5_register_backend("OPENCL", NULL, NULL, NULL);
}
