#include <bp5.h>
#include <stdio.h>

int main(int argc, char **argv) {
  struct bp5_t *bp5 = bp5_init(argc, argv);
  bp5_finalize(&bp5);

  return 0;
}
