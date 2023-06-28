#include <bp5-impl.h>
#include <getopt.h>

// Dynamic memory free function.
void bp5_free_(void **p) { free(*p), *p = NULL; }

static void print_help(const char *name) {
  printf("Usage: %s [OPTIONS]\n", name);
  printf("Options:\n");
  printf("  --bp5-verbose=<VERBOSITY>, Verbose level. Values: 0, 1, 2, ...\n");
  printf("  --bp5-help, Prints this help message and exit.\n");
}

static void bp5_parse_opts(struct bp5_t *bp5, int *argc, char ***argv_in) {
  bp5->verbose = 0;

  static struct option long_options[] = {
      {"bp5-verbose", optional_argument, 0, 10},
      {"bp5-help", no_argument, 0, 99},
      {0, 0, 0, 0}};

  char **argv = *argv_in;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
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

  // Remove parsed arguments from argv. We just need to update the pointers
  // since command line arguments are not transient and available until the
  // end of the program.
  for (int i = optind; i < *argc; i++)
    argv[i - optind] = argv[i];
  *argc -= optind;

  return;
}

struct bp5_t *bp5_init(int *argc, char ***argv_in) {
  struct bp5_t *bp5 = bp5_calloc(struct bp5_t, 1);

  bp5_parse_opts(bp5, argc, argv_in);

  return bp5;
}

void bp5_finalize(struct bp5_t **bp5) { bp5_free(bp5); }
