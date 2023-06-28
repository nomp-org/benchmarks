#include <bp5.h>
#include <stdio.h>

int main(int argc, char **argv) {
  struct bp5_t *bp5 = bp5_init(&argc, &argv);

  for (int i = 0; i < argc; i++)
    printf("argv[%d] = %s\n", i, argv[i]);

  bp5_finalize(&bp5);

  return 0;
}
