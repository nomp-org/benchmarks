#include <nekbone.h>
#include <stdio.h>

int main(int argc, char **argv) {
  struct nekbone_t *nekbone = nekbone_init(&argc, &argv);

  nekbone_run(nekbone);

  nekbone_finalize(&nekbone);

  return 0;
}
