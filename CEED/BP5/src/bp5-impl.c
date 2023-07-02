#include "bp5-impl.h"
#include <ctype.h>
#include <getopt.h>
#include <string.h>

// Dynamic memory free function.
void bp5_free_(void **p) { free(*p), *p = NULL; }

static void print_help(const char *name) {
  printf("Usage: %s [OPTIONS]\n", name);
  printf("Options:\n");
  printf("  --bp5-verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  printf("  --bp5-device-id=<device id>, Device ID (0, 1, 2, ...).\n");
  printf("  --bp5-platform-id=<platform id>, Platform ID (0, 1, 2, ...).\n");
  printf("  --bp5-backend=<backend>, Backend (CUDA, OpenCL, nomp, etc.).\n");
  printf("  --bp5-nelt <# of elements>, Number of elements (1, 2, 3, ...).\n");
  printf("  --bp5-nx1 <order>, Polynomial order (1, 2, 3, ...).\n");
  printf("  --bp5-max-iter=<iters>, Number of CG iterations (1, 2, 3, ...).\n");
  printf("  --bp5-help, Prints this help message and exit.\n");
}

inline static void set_backend(struct bp5_t *bp5, const char *backend) {
  size_t len = strnlen(backend, BUFSIZ);
  for (uint i = 0; i < len; i++)
    bp5->backend[i] = toupper(backend[i]);
}

static void bp5_parse_opts(struct bp5_t *bp5, int *argc, char ***argv_) {
  static struct option long_options[] = {
      {"bp5-verbose", optional_argument, 0, 10},
      {"bp5-device-id", optional_argument, 0, 12},
      {"bp5-platform-id", optional_argument, 0, 14},
      {"bp5-backend", optional_argument, 0, 16},
      {"bp5-nelt", required_argument, 0, 20},
      {"bp5-nx1", required_argument, 0, 22},
      {"bp5-max-iter", optional_argument, 0, 24},
      {"bp5-help", no_argument, 0, 99},
      {0, 0, 0, 0}};

  // Default values for optional arguments.
  bp5->verbose = 0, bp5->device_id = 0, bp5->platform_id = 0;
  strncpy(bp5->backend, "NOMP", BUFSIZ);
  bp5->max_iter = 1000;

  // Set invalid values for required arguments so we can check if they were
  // initialized later.
  bp5->nelt = -1, bp5->nx1 = -1;

  char **argv = *argv_;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      bp5->verbose = atoi(optarg);
      break;
    case 12:
      bp5->device_id = atoi(optarg);
      break;
    case 14:
      bp5->platform_id = atoi(optarg);
      break;
    case 16:
      set_backend(bp5, optarg);
      break;
    case 20:
      bp5->nelt = atoi(optarg);
      break;
    case 22:
      bp5->nx1 = atoi(optarg) + 1;
      break;
    case 24:
      bp5->max_iter = atoi(optarg);
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

  // Check if the required arguments were provided.
  if (bp5->nelt < 1)
    bp5_error("bp5_parse_opts: --bp5-nelt is not provided or invalid.\n");
  if (bp5->nx1 < 2)
    bp5_error("bp5_parse_opts: --bp5-nx1 is not provided or invalid.\n");

  // Remove parsed arguments from argv. We just need to update the pointers
  // since command line arguments are not transient and available until the
  // end of the program.
  for (int i = optind; i < *argc; i++)
    argv[i - optind] = argv[i];
  *argc -= optind;

  return;
}

struct bp5_t *bp5_init(int *argc, char ***argv) {
  // Register available GPU backends.
  bp5_register_backends();

  struct bp5_t *bp5 = bp5_calloc(struct bp5_t, 1);
  bp5_parse_opts(bp5, argc, argv);

  // Initialize the backend.
  bp5_init_backend(bp5);

  // Setup the problem data on host.
  bp5_gs_setup(bp5);
  bp5_read_zwgll(bp5);
  bp5_geom_setup(bp5);
  bp5_derivative_setup(bp5);
  bp5_inverse_multiplicity_setup(bp5);

  return bp5;
}

void bp5_debug(int verbose, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (verbose > 0)
    vprintf(fmt, args);
  va_end(args);
}

void bp5_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  exit(EXIT_FAILURE);
}

void bp5_assert_(int cond, const char *msg, const char *file,
                 const unsigned line) {
  if (!cond) {
    fprintf(stderr, "%s:%d Assertion failure: %s", file, line, msg);
    exit(EXIT_FAILURE);
  }
}

void bp5_finalize(struct bp5_t **bp5_) {
  if (!bp5_ || !*bp5_)
    return;

  struct bp5_t *bp5 = *bp5_;
  bp5_debug(bp5->verbose, "bp5_finalize: ...");

  bp5_unregister_backends();

  bp5_free(&bp5->gs_off), bp5_free(&bp5->gs_idx);
  bp5_free(&bp5->w), bp5_free(&bp5->z);
  bp5_free(&bp5->g), bp5_free(&bp5->D), bp5_free(&bp5->c);

  int verbose = bp5->verbose;
  bp5_free(bp5_);

  bp5_debug(verbose, "done.\n");
}