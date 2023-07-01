#include "bp5-impl.h"

BP5_INTERN void bp5_cuda_init(void) {
  bp5_register_backend("CUDA", NULL, NULL, NULL);
}
