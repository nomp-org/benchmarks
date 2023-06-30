#include "bp5-impl.h"

BP5_INTERN void bp5_nomp_init(void) {
  bp5_register_backend("NOMP", NULL, NULL);
}
