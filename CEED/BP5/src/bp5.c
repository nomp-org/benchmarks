#include <bp5-impl.h>
#include <getopt.h>

// Dynamic memory free function.
void bp5_free_(void **p) { free(*p), *p = NULL; }

static void print_help(const char *name) {
  printf("Usage: %s [OPTIONS]\n", name);
  printf("Options:\n");
  printf("  --verbose=<VERBOSITY>, Verbose level. Values: 0, 1, 2, ...\n");
  printf("  --help, Prints this help message.\n");
}

struct bp5_t *bp5_init(int argc, char **argv) {
  static struct option long_options[] = {{"verbose", optional_argument, 0, 10},
                                         {"help", no_argument, 0, 99},
                                         {0, 0, 0, 0}};

  struct bp5_t *bp5 = (struct bp5_t *)calloc(1, sizeof(struct bp5_t));
  bp5->verbose = 0;

  for (;;) {
    int c = getopt_long(argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      bp5->verbose = atoi(optarg);
      break;
    case 99:
      print_help(argv[0]);
      exit(EXIT_SUCCESS);
      break;
    default:
      print_help(argv[0]);
      exit(EXIT_FAILURE);
      break;
    }
  }

  return bp5;
}

void bp5_finalize(struct bp5_t **bp5) { bp5_free(bp5); }
