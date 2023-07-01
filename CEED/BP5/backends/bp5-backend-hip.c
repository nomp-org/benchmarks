#include "bp5-impl.h"

BP5_INTERN void bp5_hip_init(void) {
  bp5_register_backend("HIP", NULL, NULL, NULL);
}
