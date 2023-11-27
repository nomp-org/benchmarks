#if !defined(__LIBNEKBONE_H__)
#define __LIBNEKBONE_H__

#define NEKBONE_VISIBILITY(mode) __attribute__((visibility(#mode)))

#if defined(__cplusplus)
#define NEKBONE_EXTERN extern "C" NEKBONE_VISIBILITY(default)
#else
#define NEKBONE_EXTERN extern NEKBONE_VISIBILITY(default)
#endif

#if defined(__cplusplus)
#define NEKBONE_INTERN extern "C" NEKBONE_VISIBILITY(hidden)
#else
#define NEKBONE_INTERN extern NEKBONE_VISIBILITY(hidden)
#endif

struct nekbone_t;
NEKBONE_EXTERN struct nekbone_t *nekbone_init(int *argc, char ***argv);

NEKBONE_EXTERN void nekbone_debug(int verbose, const char *fmt, ...);

NEKBONE_EXTERN void nekbone_error(const char *fmt, ...);

NEKBONE_INTERN void nekbone_assert_(int cond, const char *fmt, const char *file,
                                    const unsigned line);
#define nekbone_assert(COND, MSG) nekbone_assert_(COND, MSG, __FILE__, __LINE__)

NEKBONE_EXTERN double nekbone_run(const struct nekbone_t *nekbone);

NEKBONE_EXTERN void nekbone_finalize(struct nekbone_t **nekbone);

#endif // __LIBNEKBONE_H__
