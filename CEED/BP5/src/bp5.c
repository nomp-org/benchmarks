#include <bp5-impl.h>
#include <getopt.h>

extern void bp5_gs_setup(struct bp5_t *bp5);

// Dynamic memory free function.
void bp5_free_(void **p) { free(*p), *p = NULL; }

static void print_help(const char *name) {
  printf("Usage: %s [OPTIONS]\n", name);
  printf("Options:\n");
  printf("  --bp5-verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  printf("  --bp5-nelt <# of elements>, Number of elements. (1, 2, 3, ...)\n");
  printf("  --bp5-order <order>, Polynomial order. (1, 2, 3, ...)\n");
  printf("  --bp5-help, Prints this help message and exit.\n");
}

static void bp5_parse_opts(struct bp5_t *bp5, int *argc, char ***argv_in) {
  bp5->verbose = 0;

  static struct option long_options[] = {
      {"bp5-verbose", optional_argument, 0, 10},
      {"bp5-nelt", required_argument, 0, 20},
      {"bp5-order", required_argument, 0, 22},
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
    case 20:
      bp5->nelt = atoi(optarg);
      break;
    case 22:
      bp5->nx1 = atoi(optarg);
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
void bp5_debug(int verbose, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (verbose > 0)
    vprintf(fmt, args);
  va_end(args);
}

void bp5_assert_(int cond, const char *msg, const char *file,
                 const unsigned line) {
  if (!cond) {
    printf("%s:%d Assertion failure: %s", file, line, msg);
    exit(EXIT_FAILURE);
  }
}
