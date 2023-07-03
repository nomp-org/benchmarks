#include "bp5-impl.h"

static uint initialized = 0;

static scalar *r, *x, *z, *p, *w;
static const scalar *c, *g, *D;
static const uint *gs_off, *gs_idx;
static scalar *wrk;

static void nomp_mem_init(const struct bp5_t *bp5) {
  bp5_debug(bp5->verbose, "nomp_mem_init: Copy problem data to device ... ");

  // We allocate following arrays used in CG on both host and device.
  // Techinically we don't need the host arrays if we always run on the device.
  // But in the case nomp is not enabled, we need these arrays on host.
  uint dofs = bp5_get_local_dofs(bp5);
  r = bp5_calloc(scalar, dofs);
  x = bp5_calloc(scalar, dofs);
  z = bp5_calloc(scalar, dofs);
  p = bp5_calloc(scalar, dofs);
  w = bp5_calloc(scalar, dofs);

#pragma nomp update(alloc                                                      \
                    : r [0:dofs], x [0:dofs], z [0:dofs], p [0:dofs],          \
                      w [0:dofs])

  // There is no need to allcoate following arrays on host. We just copy them
  // into the device.
  c = bp5->c, g = bp5->g, D = bp5->D;
  gs_off = bp5->gs_off, gs_idx = bp5->gs_idx;

#pragma nomp update(to : c [0:dofs], g [0:6 * dofs], D [0:bp5->nx1 * bp5->nx1])

  wrk = bp5_calloc(scalar, dofs);
#pragma nomp update(alloc : wrk [0:dofs])

  bp5_debug(bp5->verbose, "done.\n");
}

static void nomp_init(const struct bp5_t *bp5) {
  if (initialized)
    return;

  bp5_debug(bp5->verbose, "nomp_init: Initializing NOMP backend ... ");
  const int argc = 6;
  char *argv[] = {"--nomp-device-id", "0", "--nomp-backend", "cuda",
                  "--nomp-verbose",   "0"};
#pragma nomp init(argc, argv)

  nomp_mem_init(bp5);

  initialized = 1;
  bp5_debug(bp5->verbose, "done.\n");
}

BP5_INTERN void bp5_nomp_init(void) {
  bp5_register_backend("NOMP", nomp_init, NULL, NULL);
}
