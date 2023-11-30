#include "nekbone-backend.h"
#include "nekbone-impl.h"
#include <ctype.h>
#include <getopt.h>
#include <string.h>

// Dynamic memory free function.
void nekbone_free_(void **p) { free(*p), *p = NULL; }

static void print_help(const char *name) {
  printf("Usage: %s [OPTIONS]\n", name);
  printf("Options:\n");
  printf(
      "  --nekbone-verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  printf("  --nekbone-device-id=<device id>, Device ID (0, 1, 2, ...).\n");
  printf(
      "  --nekbone-platform-id=<platform id>, Platform ID (0, 1, 2, ...).\n");
  printf(
      "  --nekbone-backend=<backend>, Backend (CUDA, OpenCL, nomp, etc.).\n");
  printf(
      "  --nekbone-nelems <# elements>, Number of elements (1, 2, 3, ...).\n");
  printf("  --nekbone-order <order>, Polynomial order (1, 2, 3, ...).\n");
  printf("  --nekbone-max-iter=<iters>, Number of CG iterations (1, 2, 3, "
         "...).\n");
  printf("  --nekbone-help, Prints this help message and exit.\n");
}

inline static void set_backend(struct nekbone_t *nekbone, const char *backend) {
  size_t len = strnlen(backend, BUFSIZ);
  for (uint i = 0; i < len; i++)
    nekbone->backend[i] = toupper(backend[i]);
}

static void nekbone_parse_opts(struct nekbone_t *nekbone, int *argc,
                               char ***argv_) {
  static struct option long_options[] = {
      {"nekbone-verbose", optional_argument, 0, 10},
      {"nekbone-device-id", optional_argument, 0, 12},
      {"nekbone-platform-id", optional_argument, 0, 14},
      {"nekbone-backend", required_argument, 0, 16},
      {"nekbone-nelems", required_argument, 0, 20},
      {"nekbone-order", required_argument, 0, 22},
      {"nekbone-max-iter", optional_argument, 0, 24},
      {"nekbone-help", no_argument, 0, 99},
      {0, 0, 0, 0}};

  // Default values for optional arguments.
  nekbone->verbose = NEKBONE_VERBOSE;
  nekbone->device = NEKBONE_DEVICE;
  nekbone->platform = NEKBONE_PLATFORM;
  nekbone->max_iter = NEKBONE_MAX_ITER;

  // Set invalid values for required arguments so we can check if they were
  // initialized later.
  nekbone->nelt = -1, nekbone->nx1 = -1;
  strncpy(nekbone->backend, "", 1);

  char **argv = *argv_;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      nekbone->verbose = atoi(optarg);
      break;
    case 12:
      nekbone->device = atoi(optarg);
      break;
    case 14:
      nekbone->platform = atoi(optarg);
      break;
    case 16:
      set_backend(nekbone, optarg);
      break;
    case 20:
      nekbone->nelt = atoi(optarg);
      break;
    case 22:
      nekbone->nx1 = atoi(optarg) + 1;
      break;
    case 24:
      nekbone->max_iter = atoi(optarg);
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
  if (nekbone->nelt < 1)
    nekbone_error(
        "nekbone_parse_opts: --nekbone-nelems is not provided or invalid.\n");
  if (nekbone->nx1 < 2)
    nekbone_error(
        "nekbone_parse_opts: --nekbone-order is not provided or invalid.\n");
  if (strnlen(nekbone->backend, BUFSIZ) < 1)
    nekbone_error(
        "nekbone_parse_opts: --nekbone-backend is not provided or invalid.\n");

  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: verbose=%d\n",
                nekbone->verbose);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: device=%d\n",
                nekbone->device);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: platform=%d\n",
                nekbone->platform);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: backend=%s\n",
                nekbone->backend);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: nelems=%d\n",
                nekbone->nelt);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: order=%d\n",
                nekbone->nx1 - 1);
  nekbone_debug(nekbone->verbose, "nekbone_parse_opts: max_iter=%d\n",
                nekbone->max_iter);

  // Remove parsed arguments from argv. We just need to update the pointers
  // since command line arguments are not transient and available until the
  // end of the program.
  for (int i = optind; i < *argc; i++)
    argv[i - optind] = argv[i];
  *argc -= optind;

  return;
}

struct nekbone_t *nekbone_init(int *argc, char ***argv) {
  struct nekbone_t *nekbone = nekbone_calloc(struct nekbone_t, 1);
  nekbone_parse_opts(nekbone, argc, argv);

  // Register available GPU backends.
  nekbone_register_backends(nekbone->verbose);

  // Setup the problem data on host.
  nekbone_gs_setup(nekbone);
  nekbone_read_zwgll(nekbone);
  nekbone_geom_setup(nekbone);
  nekbone_derivative_setup(nekbone);
  nekbone_inverse_multiplicity_setup(nekbone);

  // Initialize the backend.
  nekbone_init_backend(nekbone);

  return nekbone;
}

void nekbone_debug(int verbose, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (verbose > 0)
    vprintf(fmt, args);
  va_end(args);
}

void nekbone_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  exit(EXIT_FAILURE);
}

void nekbone_assert_(int cond, const char *msg, const char *file,
                     const unsigned line) {
  if (!cond) {
    fprintf(stderr, "%s:%d Assertion failure: %s", file, line, msg);
    exit(EXIT_FAILURE);
  }
}

double nekbone_run(const struct nekbone_t *nekbone) {
  nekbone_debug(nekbone->verbose, "nekbone_run: ...\n");

  // Set RHS for the solver.
  const uint ldofs = nekbone_get_local_dofs(nekbone);
  scalar *r = nekbone_calloc(scalar, ldofs);
  for (uint i = 1; i <= ldofs; i++) {
    double arg = 1e9 * i * i;
    r[i - 1] = sin(1e9 * cos(arg));
  }

  // dssum
  nekbone_gs(r, nekbone);

  // Average the rhs by the inverse multiplicity.
  nekbone_inverse_multiplicity(r, nekbone);

  // Warmup the backend.
  nekbone_run_backend(nekbone, r);

  // Solve the system.
  double elapsed = nekbone_run_backend(nekbone, r);
  nekbone_free(&r);

  nekbone_debug(nekbone->verbose, "nekbone_run: done. Elapsed = %e\n", elapsed);

  return elapsed;
}

void nekbone_finalize(struct nekbone_t **nekbone_) {
  if (!nekbone_ || !*nekbone_)
    return;

  struct nekbone_t *nekbone = *nekbone_;
  nekbone_debug(nekbone->verbose, "nekbone_finalize: ...\n");

  nekbone_unregister_backends();

  nekbone_free(&nekbone->gs_off), nekbone_free(&nekbone->gs_idx);
  nekbone_free(&nekbone->w), nekbone_free(&nekbone->z);
  nekbone_free(&nekbone->g), nekbone_free(&nekbone->D),
      nekbone_free(&nekbone->c);

  int verbose = nekbone->verbose;
  nekbone_free(nekbone_);

  nekbone_debug(verbose, "nekbone_finalize: done.\n");
}
