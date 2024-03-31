static const char *ax_kernel_v00 =
    "#define NX1 %d\n"
    "#define scalar double\n"
    "#define uint unsigned\n"
    "\n"
    "#define NEKBONE_IDX2(i, j) ((i) + NX1 * (j))\n"
    "#define NEKBONE_IDX3(i, j, k) ((i) + NX1 * ((j) + NX1 * (k)))\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(NX1 *NX1 *NX1)\n"
    "    ax_kernel(scalar *__restrict__ w, const scalar *__restrict__ u,\n"
    "              const scalar *__restrict__ G, const scalar *__restrict__ D) "
    "{\n"
    "  const uint ebase = blockIdx.x * NX1 * NX1 * NX1;\n"
    "  const uint i = threadIdx.x;\n"
    "  const uint j = threadIdx.y;\n"
    "  const uint k = threadIdx.z;\n"
    "\n"
    "  __shared__ scalar s_D[NX1][NX1];\n"
    "  __shared__ scalar s_ur[NX1][NX1][NX1];\n"
    "  __shared__ scalar s_us[NX1][NX1][NX1];\n"
    "  __shared__ scalar s_ut[NX1][NX1][NX1];\n"
    "\n"
    "  s_ur[k][j][i] = 0;\n"
    "  if (k == 0) s_D[j][i] = D[NEKBONE_IDX2(i, j)];\n"
    "  s_us[k][j][i] = 0;\n"
    "  s_ut[k][j][i] = 0;\n"
    "  __syncthreads();\n"
    "\n"
    "  for (uint l = 0; l < NX1; ++l) {\n"
    "    s_ur[k][j][i] += s_D[i][l] * u[ebase + NEKBONE_IDX3(l, j, k)];\n"
    "    s_us[k][j][i] += s_D[j][l] * u[ebase + NEKBONE_IDX3(i, l, k)];\n"
    "    s_ut[k][j][i] += s_D[k][l] * u[ebase + NEKBONE_IDX3(i, j, l)];\n"
    "  }\n"
    "  __syncthreads();\n"
    "\n"
    "  const uint gbase = 6 * (ebase + NEKBONE_IDX3(i, j, k));\n"
    "  scalar r_G00 = G[gbase + 0];\n"
    "  scalar r_G01 = G[gbase + 1];\n"
    "  scalar r_G02 = G[gbase + 2];\n"
    "  scalar r_G11 = G[gbase + 3];\n"
    "  scalar r_G12 = G[gbase + 4];\n"
    "  scalar r_G22 = G[gbase + 5];\n"
    "\n"
    "  scalar wr =\n"
    "      r_G00 * s_ur[k][j][i] + r_G01 * s_us[k][j][i] + r_G02 * "
    "s_ut[k][j][i];\n"
    "  scalar ws =\n"
    "      r_G01 * s_ur[k][j][i] + r_G11 * s_us[k][j][i] + r_G12 * "
    "s_ut[k][j][i];\n"
    "  scalar wt =\n"
    "      r_G02 * s_ur[k][j][i] + r_G12 * s_us[k][j][i] + r_G22 * "
    "s_ut[k][j][i];\n"
    "  __syncthreads();\n"
    "\n"
    "  s_ur[k][j][i] = wr;\n"
    "  s_us[k][j][i] = ws;\n"
    "  s_ut[k][j][i] = wt;\n"
    "  __syncthreads();\n"
    "\n"
    "  scalar wo = 0;\n"
    "  for (uint l = 0; l < NX1; l++) {\n"
    "    wo += s_D[l][i] * s_ur[k][j][l] + s_D[l][j] * s_us[k][l][i] +\n"
    "          s_D[l][k] * s_ut[l][j][i];\n"
    "  }\n"
    "  w[ebase + NEKBONE_IDX3(i, j, k)] = wo;\n"
    "}\n";

static const char *ax_kernel_v01 =
    "#define bIdx(N) ((int) blockIdx.N)\n"
    "#define tIdx(N) ((int) threadIdx.N)\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(512) ax_kernel("
    "double *__restrict__ w, double const *__restrict__ u, double "
    "const *__restrict__ G, double const *__restrict__ D)\n"
    "{\n"
    "  __shared__ double D_fetch[8 * 8];\n"
    "  double r_G00;\n"
    "  double r_G01;\n"
    "  double r_G02;\n"
    "  double r_G11;\n"
    "  double r_G12;\n"
    "  double r_G22;\n"
    "  __shared__ double ur[8 * 8 * 8];\n"
    "  __shared__ double us[8 * 8 * 8];\n"
    "  __shared__ double ut[8 * 8 * 8];\n"
    "  double wo;\n"
    "  double wr;\n"
    "  double ws;\n"
    "  double wt;\n"
    "\n"
    "  ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = 0;\n"
    "  if (tIdx(y) == 0)\n"
    "    D_fetch[8 * tIdx(x) + tIdx(z)] = D[8 * tIdx(x) + tIdx(z)];\n"
    "  us[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = 0;\n"
    "  ut[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = 0;\n"
    "  __syncthreads() /* for D_fetch (_nomp_insn_4 depends on D_fetch_rule) "
    "*/;\n"
    "  for (int l = 0; l <= 7; ++l)\n"
    "  {\n"
    "    ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = ur[64 * tIdx(z) + 8 * "
    "tIdx(y) + tIdx(x)] + D_fetch[8 * tIdx(x) + l] * u[512 * bIdx(x) + 64 * "
    "tIdx(z) + 8 * tIdx(y) + l];\n"
    "    us[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = us[64 * tIdx(z) + 8 * "
    "tIdx(y) + tIdx(x)] + D_fetch[8 * tIdx(y) + l] * u[512 * bIdx(x) + 64 * "
    "tIdx(z) + 8 * l + tIdx(x)];\n"
    "    ut[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = ut[64 * tIdx(z) + 8 * "
    "tIdx(y) + tIdx(x)] + D_fetch[8 * tIdx(z) + l] * u[512 * bIdx(x) + 64 * l "
    "+ 8 * tIdx(y) + tIdx(x)];\n"
    "  }\n"
    "  r_G00 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * "
    "tIdx(x)];\n"
    "  r_G01 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * tIdx(x) + "
    "1];\n"
    "  r_G02 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * tIdx(x) + "
    "2];\n"
    "  r_G11 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * tIdx(x) + "
    "3];\n"
    "  r_G12 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * tIdx(x) + "
    "4];\n"
    "  r_G22 = G[3072 * bIdx(x) + 384 * tIdx(z) + 48 * tIdx(y) + 6 * tIdx(x) + "
    "5];\n"
    "  __syncthreads() /* for ur (_nomp_insn_11 depends on _nomp_insn_2) */;\n"
    "  wr = r_G00 * ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G01 * us[64 * "
    "tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G02 * ut[64 * tIdx(z) + 8 * tIdx(y) "
    "+ tIdx(x)];\n"
    "  ws = r_G01 * ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G11 * us[64 * "
    "tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G12 * ut[64 * tIdx(z) + 8 * tIdx(y) "
    "+ tIdx(x)];\n"
    "  wt = r_G02 * ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G12 * us[64 * "
    "tIdx(z) + 8 * tIdx(y) + tIdx(x)] + r_G22 * ut[64 * tIdx(z) + 8 * tIdx(y) "
    "+ tIdx(x)];\n"
    "  __syncthreads() /* for ur (_nomp_insn_14 depends on _nomp_insn_11) */;\n"
    "  ur[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = wr;\n"
    "  us[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = ws;\n"
    "  ut[64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = wt;\n"
    "  wo = 0;\n"
    "  __syncthreads() /* for ur (_nomp_insn_18 depends on _nomp_insn_14) */;\n"
    "  for (int l_ = 0; l_ <= 7; ++l_)\n"
    "    wo = wo + D_fetch[8 * l_ + tIdx(x)] * ur[64 * tIdx(z) + 8 * tIdx(y) + "
    "l_] + D_fetch[8 * l_ + tIdx(y)] * us[64 * tIdx(z) + 8 * l_ + tIdx(x)] + "
    "D_fetch[8 * l_ + tIdx(z)] * ut[64 * l_ + 8 * tIdx(y) + tIdx(x)];\n"
    "  w[512 * bIdx(x) + 64 * tIdx(z) + 8 * tIdx(y) + tIdx(x)] = wo;\n"
    "}\n";

static const char        *ax_kernel_template[] = {ax_kernel_v00, ax_kernel_v01};
static unified_module_t   module               = NULL;
static unified_function_t function             = NULL;
static size_t             global[3]            = {0};
static size_t             local[3]             = {0};
static int                ax_dynamic_initialized = 0;

inline static void ax_dynamic_setup(const int nelt, const int nx1) {
  if (ax_dynamic_initialized) return;

  int    template_id = (nx1 == 8) ? 1 : 0;
  size_t length      = strlen(ax_kernel_template[template_id]) + 100;
  char  *ax_kernel   = nekbone_calloc(char, length);
  if (template_id == 0)
    snprintf(ax_kernel, length, ax_kernel_template[template_id], nx1);
  if (template_id == 1)
    strncpy(ax_kernel, ax_kernel_template[template_id], length);

  unified_rtc_program program;
  check_rtc(
      unified_rtc_create_program(&program, ax_kernel, NULL, 0, NULL, NULL));
  nekbone_free(&ax_kernel);

  char *code = NULL;

  unified_rtc_result result = unified_rtc_compile_program(program, 0, NULL);
  if (result != UNIFIED_RTC_SUCCESS) goto rtc_error;

  size_t size;
  check_rtc(unified_rtc_get_code_size(program, &size));
  code = nekbone_calloc(char, size + 1);
  check_rtc(unified_rtc_get_code(program, code));
  check_rtc(unified_rtc_destroy_program(&program));

  check_runtime(unified_module_load_data(&module, code));
  check_runtime(unified_module_get_function(&function, module, "ax_kernel"));
  nekbone_free(&code);

  global[0] = nelt, global[1] = 1, global[2] = 1;
  local[0] = nx1, local[1] = nx1, local[2] = nx1;
  ax_dynamic_initialized = 1;

  return;

rtc_error:
  unified_rtc_get_program_log_size(program, &size);

  char *log = nekbone_calloc(char, size + 1);
  unified_rtc_get_program_log(program, log);

  const char *error_string = unified_rtc_get_error_string(result);
  size += strlen(error_string) + 2 + 1;

  char *msg = nekbone_calloc(char, size);
  snprintf(msg, size, "%s: %s", error_string, log);

  // Only cleaning up some resources in case of an error.
  nekbone_free(&ax_kernel), nekbone_free(&log);

  nekbone_error("%s\n", msg);
}

inline static void ax_dynamic(scalar *d_w, const scalar *d_u, const scalar *d_g,
                              const scalar *d_D, const uint nelt,
                              const uint nx1) {
  void *args[] = {(void *)&d_w, (void *)&d_u, (void *)&d_g, (void *)&d_D, 0};

  check_runtime(unified_module_launch_kernel(function, global[0], global[1],
                                             global[2], local[0], local[1],
                                             local[2], 0, NULL, args, NULL));
}

inline static void ax_dynamic_finalize() {
  if (!ax_dynamic_initialized) return;
  unified_module_unload(module);
  ax_dynamic_initialized = 0;
}
